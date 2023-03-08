import copy
import time
from collections import namedtuple
from typing import Optional, Callable, Tuple

import numpy as np
import torch
from ding.envs import BaseEnvManager
from ding.torch_utils import to_ndarray
from ding.utils import build_logger, EasyTimer
from ding.worker.collector.base_serial_evaluator import ISerialEvaluator, VectorEvalMonitor
from easydict import EasyDict

from lzero.mcts.utils import prepare_observation_list


class MuZeroEvaluator(ISerialEvaluator):
    """
    Overview:
        The Evaluator for MCTS+RL algorithms, including MuZero, EfficientZero, Sampled EfficientZero.
    Interfaces:
        __init__, reset, reset_policy, reset_env, close, should_eval, eval
    Property:
        env, policy
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        """
        Overview:
            Get evaluator's default config. We merge evaluator's default config with other default configs\
                and user's config to get the final config.
        Return:
            cfg (:obj:`EasyDict`): evaluator's default config
        """
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = dict(
        # Evaluate every "eval_freq" training iterations.
        eval_freq=50,
    )

    def __init__(
            self,
            cfg: dict,
            env: BaseEnvManager = None,
            policy: namedtuple = None,
            tb_logger: 'SummaryWriter' = None,  # noqa
            exp_name: Optional[str] = 'default_experiment',
            instance_name: Optional[str] = 'evaluator',
            game_config: 'game_config' = None,  # noqa
    ) -> None:
        """
        Overview:
            Init method. Load config and use ``self._cfg`` setting to build common serial evaluator components,
            e.g. logger helper, timer.
        Arguments:
            - cfg (:obj:`EasyDict`): Configuration EasyDict.
            - env (:obj:`BaseEnvManager`): the subclass of vectorized env_manager(BaseEnvManager)
            - policy (:obj:`namedtuple`): the api namedtuple of collect_mode policy
            - tb_logger (:obj:`SummaryWriter`): tensorboard handle
            - exp_name (:obj:`str`): Experiment name, which is used to indicate output directory.
            - instance_name (:obj:`Optional[str]`): Name of this instance.
            - game_config: Config of game.
        """
        self._cfg = cfg
        self._exp_name = exp_name
        self._instance_name = instance_name
        if tb_logger is not None:
            self._logger, _ = build_logger(
                path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name, need_tb=False
            )
            self._tb_logger = tb_logger
        else:
            self._logger, self._tb_logger = build_logger(
                path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name
            )
        self.reset(policy, env)

        self._timer = EasyTimer()
        self._default_n_episode = cfg.n_episode
        self._stop_value = cfg.stop_value

        # MuZero
        self.game_config = game_config



    def reset_env(self, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Reset evaluator's environment. In some case, we need evaluator use the same policy in different \
                environments. We can use reset_env to reset the environment.
            If _env is None, reset the old environment.
            If _env is not None, replace the old environment in the evaluator with the \
                new passed in environment and launch.
        Arguments:
            - env (:obj:`Optional[BaseEnvManager]`): instance of the subclass of vectorized \
                env_manager(BaseEnvManager)
        """
        if _env is not None:
            self._env = _env
            self._env.launch()
            self._env_num = self._env.env_num
        else:
            self._env.reset()

    def reset_policy(self, _policy: Optional[namedtuple] = None) -> None:
        """
        Overview:
            Reset evaluator's policy. In some case, we need evaluator work in this same environment but use\
                different policy. We can use reset_policy to reset the policy.
            If _policy is None, reset the old policy.
            If _policy is not None, replace the old policy in the evaluator with the new passed in policy.
        Arguments:
            - policy (:obj:`Optional[namedtuple]`): the api namedtuple of eval_mode policy
        """
        assert hasattr(self, '_env'), "please set env first"
        if _policy is not None:
            self._policy = _policy
        # self._action_shape = _policy.get_attribute('cfg').model.action_shape  # TODO
        self._policy.reset()

    def reset(self, _policy: Optional[namedtuple] = None, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Reset evaluator's policy and environment. Use new policy and environment to collect data.
            If _env is None, reset the old environment.
            If _env is not None, replace the old environment in the evaluator with the new passed in \
                environment and launch.
            If _policy is None, reset the old policy.
            If _policy is not None, replace the old policy in the evaluator with the new passed in policy.
        Arguments:
            - policy (:obj:`Optional[namedtuple]`): the api namedtuple of eval_mode policy
            - env (:obj:`Optional[BaseEnvManager]`): instance of the subclass of vectorized \
                env_manager(BaseEnvManager)
        """
        if _env is not None:
            self.reset_env(_env)
        if _policy is not None:
            self.reset_policy(_policy)
        self._max_eval_reward = float("-inf")
        self._last_eval_iter = 0
        self._end_flag = False

    def close(self) -> None:
        """
        Overview:
            Close the evaluator. If end_flag is False, close the environment, flush the tb_logger\
                and close the tb_logger.
        """
        if self._end_flag:
            return
        self._end_flag = True
        self._env.close()
        self._tb_logger.flush()
        self._tb_logger.close()

    def __del__(self):
        """
        Overview:
            Execute the close command and close the evaluator. __del__ is automatically called \
                to destroy the evaluator instance when the evaluator finishes its work
        """
        self.close()

    def should_eval(self, train_iter: int) -> bool:
        """
        Overview:
            Determine whether you need to start the evaluation mode, if the number of training has reached\
                the maximum number of times to start the evaluator, return True
        Arguments:
            - train_iter (:obj:`int`): Current training iteration.
        """
        if train_iter == self._last_eval_iter:
            return False
        if (train_iter - self._last_eval_iter) < self._cfg.eval_freq and train_iter != 0:
            return False
        self._last_eval_iter = train_iter
        return True

    def eval(
            self,
            save_ckpt_fn: Callable = None,
            train_iter: int = -1,
            envstep: int = -1,
            n_episode: Optional[int] = None,
            config: Optional[dict] = None,
    ) -> Tuple[bool, float]:
        """
        Overview:
            Evaluate policy and store the best policy based on whether it reaches the highest historical reward.
        Arguments:
            - save_ckpt_fn (:obj:`Callable`): Saving ckpt function, which will be triggered by getting the best reward.
            - train_iter (:obj:`int`): Current training iteration.
            - envstep (:obj:`int`): Current env interaction step.
            - n_episode (:obj:`int`): Number of evaluation episodes.
        Returns:
            - stop_flag (:obj:`bool`): Whether this training program can be ended.
            - eval_reward (:obj:`float`): Current eval_reward.
        """
        if self.game_config.sampled_algo:
            from lzero.mcts.tree_search.game_sampled_efficientzero import GameBlock
        else:
            from lzero.mcts.tree_search.game import GameBlock

        if n_episode is None:
            n_episode = self._default_n_episode
        assert n_episode is not None, "please indicate eval n_episode"
        envstep_count = 0
        eval_monitor = VectorEvalMonitor(self._env.env_num, n_episode)
        env_nums = self._env.env_num

        self._env.reset()
        self._policy.reset()

        # initializations
        init_obs = self._env.ready_obs

        retry_waiting_time = 0.1
        while len(init_obs.keys()) != self._env_num:
            # Wait for all envs to finish resetting.
            # self._logger.info('-----'*20)
            # print('init_obs.keys():', init_obs.keys())
            self._logger.info('Wait for all envs to finish resetting:')
            self._logger.info('self._env_states {}'.format(self._env._env_states))
            time.sleep(retry_waiting_time)
            self._logger.info('sleep {} s'.format(retry_waiting_time))
            self._logger.info('self._env_states {}'.format(self._env._env_states))
            init_obs = self._env.ready_obs

        # init_obs = to_tensor(init_obs, dtype=torch.float32)
        action_mask = [init_obs[i]['action_mask'] for i in range(env_nums)]
        action_mask_dict = {i: to_ndarray(init_obs[i]['action_mask']) for i in range(env_nums)}

        if 'to_play' in init_obs[0]:
            two_player_game = True
        else:
            two_player_game = False
        if two_player_game:
            to_play = [init_obs[i]['to_play'] for i in range(env_nums)]
            to_play_dict = {i: to_ndarray(init_obs[i]['to_play']) for i in range(env_nums)}

        dones = np.array([False for _ in range(env_nums)])


        game_blocks = [
            GameBlock(
                self._env.action_space,
                game_block_length=self.game_config.game_block_length,
                config=self.game_config
            ) for _ in range(env_nums)
        ]
        for i in range(env_nums):
            game_blocks[i].init(
                [to_ndarray(init_obs[i]['observation']) for _ in range(self.game_config.model.frame_stack_num)]
            )

        ready_env_id = set()
        remain_episode = n_episode

        with self._timer:
            while not eval_monitor.is_finished():
                # Get current ready env obs.
                # only for subprocess, to get the ready_env_id
                obs = self._env.ready_obs
                new_available_env_id = set(obs.keys()).difference(ready_env_id)
                ready_env_id = ready_env_id.union(set(list(new_available_env_id)[:remain_episode]))
                remain_episode -= min(len(new_available_env_id), remain_episode)

                stack_obs = {env_id: game_blocks[env_id].step_obs() for env_id in ready_env_id}
                stack_obs = list(stack_obs.values())

                action_mask_dict = {env_id: action_mask_dict[env_id] for env_id in ready_env_id}
                to_play_dict = {env_id: to_play_dict[env_id] for env_id in ready_env_id}
                action_mask = [action_mask_dict[env_id] for env_id in ready_env_id]
                to_play = [to_play_dict[env_id] for env_id in ready_env_id]

                stack_obs = to_ndarray(stack_obs)
                stack_obs = prepare_observation_list(stack_obs)
                stack_obs = torch.from_numpy(stack_obs).to(self.game_config.device).float()

                if two_player_game:
                    policy_output = self._policy.forward(stack_obs, action_mask, to_play)
                else:
                    policy_output = self._policy.forward(stack_obs, action_mask, None)

                actions_no_env_id = {k: v['action'] for k, v in policy_output.items()}
                distributions_dict_no_env_id = {k: v['distributions'] for k, v in policy_output.items()}
                if self.game_config.sampled_algo:
                    root_sampled_actions_dict_no_env_id = {k: v['root_sampled_actions'] for k, v in
                                                           policy_output.items()}

                value_dict_no_env_id = {k: v['value'] for k, v in policy_output.items()}
                pred_value_dict_no_env_id = {k: v['pred_value'] for k, v in policy_output.items()}
                visit_entropy_dict_no_env_id = {
                    k: v['visit_count_distribution_entropy']
                    for k, v in policy_output.items()
                }

                # TODO(pu): subprocess
                actions = {}
                distributions_dict = {}
                if self.game_config.sampled_algo:
                    root_sampled_actions_dict = {}

                value_dict = {}
                pred_value_dict = {}
                visit_entropy_dict = {}
                for index, env_id in enumerate(ready_env_id):
                    actions[env_id] = actions_no_env_id.pop(index)
                    distributions_dict[env_id] = distributions_dict_no_env_id.pop(index)
                    if self.game_config.sampled_algo:
                        root_sampled_actions_dict[env_id] = root_sampled_actions_dict_no_env_id.pop(index)
                    value_dict[env_id] = value_dict_no_env_id.pop(index)
                    pred_value_dict[env_id] = pred_value_dict_no_env_id.pop(index)
                    visit_entropy_dict[env_id] = visit_entropy_dict_no_env_id.pop(index)

                # Interact with env.
                timesteps = self._env.step(actions)

                for env_id, t in timesteps.items():
                    i = env_id
                    obs, reward, done, info = t.obs, t.reward, t.done, t.info
                    if self.game_config.sampled_algo:
                        game_blocks[env_id].store_search_stats(
                            distributions_dict[env_id], value_dict[env_id], root_sampled_actions_dict[env_id]
                        )
                    else:
                        game_blocks[i].store_search_stats(distributions_dict[i], value_dict[i])
                    if two_player_game:
                        # for two_player board games
                        # append a transition tuple, including a_t, o_{t+1}, r_{t}, action_mask_{t}, to_play_{t}
                        # in ``game_blocks[env_id].init``, we have append o_{t} in ``self.obs_history``
                        game_blocks[i].append(
                            actions[i], to_ndarray(obs['observation']), reward, action_mask_dict[i],
                            to_play_dict[i]
                        )
                    else:
                        game_blocks[i].append(actions[i], to_ndarray(obs['observation']), reward)

                    # NOTE: the position of code snippt is very important.
                    # the obs['action_mask'] and obs['to_play'] is corresponding to next action
                    if two_player_game:
                        action_mask_dict[i] = to_ndarray(obs['action_mask'])
                        to_play_dict[i] = to_ndarray(obs['to_play'])

                    dones[i] = done

                    if t.done:
                        # Env reset is done by env_manager automatically.
                        self._policy.reset([env_id])
                        reward = t.info['final_eval_reward']
                        if 'episode_info' in t.info:
                            eval_monitor.update_info(env_id, t.info['episode_info'])
                        eval_monitor.update_reward(env_id, reward)
                        self._logger.info(
                            "[EVALUATOR]env {} finish episode, final reward: {}, current episode: {}".format(
                                env_id, eval_monitor.get_latest_reward(env_id), eval_monitor.get_current_episode()
                            )
                        )
                        if n_episode > self._env_num:
                            # reset the finished env
                            init_obs = self._env.ready_obs

                            if len(init_obs.keys()) != self._env_num:
                                while env_id not in init_obs.keys():
                                    init_obs = self._env.ready_obs
                                    print(f'wait the {env_id} env to reset')

                            init_obs = init_obs[env_id]['observation']
                            init_obs = to_ndarray(init_obs)
                            action_mask_dict[env_id] = to_ndarray(init_obs[env_id]['action_mask'])
                            to_play_dict[env_id] = to_ndarray(init_obs[env_id]['to_play'])

                            game_blocks[i] = GameBlock(
                                self._env.action_space,
                                game_block_length=self.game_config.game_block_length,
                                config=self.game_config
                            )
                            # stack_obs_windows[env_id] = [init_obs for _ in range(self.game_config.model.frame_stack_num)]
                            # game_blocks[env_id].init(stack_obs_windows[env_id])
                            # last_game_blocks[env_id] = None
                            # last_game_priorities[env_id] = None

                            game_blocks[i].init(
                                [init_obs[i]['observation'] for _ in range(self.game_config.model.frame_stack_num)]
                            )

                        # TODO(pu): subprocess
                        ready_env_id.remove(env_id)

                    envstep_count += 1
        duration = self._timer.value
        episode_reward = eval_monitor.get_episode_return()
        info = {
            'train_iter': train_iter,
            'ckpt_name': 'iteration_{}.pth.tar'.format(train_iter),
            'episode_count': n_episode,
            'envstep_count': envstep_count,
            'avg_envstep_per_episode': envstep_count / n_episode,
            'evaluate_time': duration,
            'avg_envstep_per_sec': envstep_count / duration,
            'avg_time_per_episode': n_episode / duration,
            'reward_mean': np.mean(episode_reward),
            'reward_std': np.std(episode_reward),
            'reward_max': np.max(episode_reward),
            'reward_min': np.min(episode_reward),
            # 'each_reward': episode_reward,
        }
        episode_info = eval_monitor.get_episode_info()
        if episode_info is not None:
            info.update(episode_info)
        self._logger.info(self._logger.get_tabulate_vars_hor(info))
        # self._logger.info(self._logger.get_tabulate_vars(info))
        for k, v in info.items():
            if k in ['train_iter', 'ckpt_name', 'each_reward']:
                continue
            if not np.isscalar(v):
                continue
            self._tb_logger.add_scalar('{}_iter/'.format(self._instance_name) + k, v, train_iter)
            self._tb_logger.add_scalar('{}_step/'.format(self._instance_name) + k, v, envstep)
        eval_reward = np.mean(episode_reward)
        if eval_reward > self._max_eval_reward:
            if save_ckpt_fn:
                save_ckpt_fn('ckpt_best.pth.tar')
            self._max_eval_reward = eval_reward
        stop_flag = eval_reward >= self._stop_value and train_iter > 0
        if stop_flag:
            self._logger.info(
                "[DI-engine serial pipeline] " +
                "Current eval_reward: {} is greater than stop_value: {}".format(eval_reward, self._stop_value) +
                ", so your RL agent is converged, you can refer to 'log/evaluator/evaluator_logger.txt' for details."
            )
        return stop_flag, eval_reward
