import numpy as np
from ditk import logging
from ding.framework import task
from ding.utils import EasyTimer
from ding.torch_utils import to_ndarray, to_tensor, to_device
from ding.framework.middleware.functional.evaluator import VectorEvalMonitor
from lzero.mcts.buffer.game_segment import GameSegment
from lzero.mcts.utils import prepare_observation


class MuZeroEvaluator:

    def __init__(
            self,
            cfg,
            policy,
            env,
            eval_freq: int = 100,
    ) -> None:
        self._cfg = cfg.policy
        self._env = env
        self._env.seed(cfg.seed, dynamic_seed=False)
        self._n_episode = cfg.env.n_evaluator_episode
        self._policy = policy
        self._eval_freq = eval_freq
        self._max_eval_reward = float("-inf")
        self._last_eval_iter = 0

        self._timer = EasyTimer()
        self._stop_value = cfg.env.stop_value

    def __call__(self, ctx):
        if ctx.last_eval_iter != -1 and \
           (ctx.train_iter - ctx.last_eval_iter < self._eval_freq):
            return
        ctx.last_eval_iter = ctx.train_iter
        if self._env.closed:
            self._env.launch()
        else:
            self._env.reset()
        self._policy.reset()
        env_nums = self._env.env_num
        n_episode = self._n_episode
        eval_monitor = VectorEvalMonitor(env_nums, n_episode)
        assert env_nums == n_episode

        init_obs = self._env.ready_obs
        action_mask_dict = {i: to_ndarray(init_obs[i]['action_mask']) for i in range(env_nums)}
        to_play_dict = {i: to_ndarray(init_obs[i]['to_play']) for i in range(env_nums)}
        timestep_dict = {i: to_ndarray(init_obs[i].get('timestep', -1)) for i in range(env_nums)}

        game_segments = [
            GameSegment(self._env.action_space, game_segment_length=self._cfg.game_segment_length, config=self._cfg)
            for _ in range(env_nums)
        ]
        for i in range(env_nums):
            game_segments[i].reset(
                [to_ndarray(init_obs[i]['observation']) for _ in range(self._cfg.model.frame_stack_num)]
            )

        ready_env_id = set()
        remain_episode = n_episode

        with self._timer:
            while not eval_monitor.is_finished():
                obs = self._env.ready_obs
                new_available_env_id = set(obs.keys()).difference(ready_env_id)
                ready_env_id = ready_env_id.union(set(list(new_available_env_id)[:remain_episode]))
                remain_episode -= min(len(new_available_env_id), remain_episode)

                stack_obs = {env_id: game_segments[env_id].get_obs() for env_id in ready_env_id}
                stack_obs = list(stack_obs.values())

                action_mask_dict_ready = {env_id: action_mask_dict[env_id] for env_id in ready_env_id}
                to_play_dict_ready = {env_id: to_play_dict[env_id] for env_id in ready_env_id}
                timestep_dict_ready = {env_id: timestep_dict[env_id] for env_id in ready_env_id}
                
                action_mask = [action_mask_dict_ready[env_id] for env_id in ready_env_id]
                to_play = [to_play_dict_ready[env_id] for env_id in ready_env_id]
                timestep = [timestep_dict_ready[env_id] for env_id in ready_env_id]

                stack_obs = to_ndarray(stack_obs)
                stack_obs = prepare_observation(stack_obs, self._cfg.model.model_type)
                stack_obs = to_tensor(stack_obs)
                stack_obs = to_device(stack_obs, self._cfg.device)

                policy_output = self._policy.forward(stack_obs, action_mask, to_play, ready_env_id=ready_env_id, timestep=timestep)

                actions = {k: v['action'] for k, v in policy_output.items()}
                timesteps = self._env.step(actions)

                for env_id, t in timesteps.items():
                    i = env_id
                    game_segments[i].append(actions[i], t.obs['observation'], t.reward)

                    if t.done:
                        # Env reset is done by env_manager automatically.
                        self._policy.reset([env_id])
                        reward = t.info['eval_episode_return']
                        if 'episode_info' in t.info:
                            eval_monitor.update_info(env_id, t.info['episode_info'])
                        eval_monitor.update_reward(env_id, reward)
                        logging.info(
                            "[EVALUATOR]env {} finish episode, final episode_return: {}, current episode: {}".format(
                                env_id, eval_monitor.get_latest_reward(env_id), eval_monitor.get_current_episode()
                            )
                        )
                        ready_env_id.remove(env_id)
        episode_reward = eval_monitor.get_episode_return()
        eval_reward = np.mean(episode_reward)
        stop_flag = eval_reward >= self._stop_value and ctx.train_iter > 0
        if stop_flag:
            task.finish = True