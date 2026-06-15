import time
from collections import defaultdict
from typing import Optional, Callable, Dict, Any

import numpy as np
import torch
from ding.torch_utils import to_ndarray, to_tensor
from ding.utils import get_rank
from ding.worker.collector.base_serial_evaluator import VectorEvalMonitor

from lzero.mcts.buffer.game_segment import GameSegment
from lzero.mcts.utils import prepare_observation
from lzero.worker.muzero_evaluator import MuZeroEvaluator


class MuZeroPerLevelEvaluator(MuZeroEvaluator):
    """MuZeroEvaluator with per-level TensorBoard logging.

    Tracks `level_id` from episode info and logs per-level + aggregated
    reward metrics to TensorBoard with tags matching PriorZero exactly,
    enabling cross-method comparison on the same TB dashboard.
    """

    def _log_per_level_tb(self, per_level_results: dict, tag_prefix: str, global_step: int) -> None:
        if not per_level_results or self._tb_logger is None:
            return
        all_level_means = []
        for level_id in sorted(per_level_results.keys()):
            rewards = per_level_results[level_id]
            mean_r = np.mean(rewards)
            self._tb_logger.add_scalar(f'{tag_prefix}/level_{level_id}_reward', mean_r, global_step)
            all_level_means.append(mean_r)
        self._tb_logger.add_scalar(f'{tag_prefix}/level_mean', np.mean(all_level_means), global_step)
        self._tb_logger.add_scalar(f'{tag_prefix}/level_std', np.std(all_level_means), global_step)
        self._tb_logger.add_scalar(f'{tag_prefix}/level_min', np.min(all_level_means), global_step)
        self._tb_logger.add_scalar(f'{tag_prefix}/level_max', np.max(all_level_means), global_step)

    def _log_agg_tb(self, info: dict, tag_prefix: str, global_step: int) -> None:
        if self._tb_logger is None:
            return
        for k in ['avg_envstep_per_episode', 'reward_mean', 'reward_std', 'reward_max', 'reward_min']:
            if k in info:
                self._tb_logger.add_scalar(f'{tag_prefix}/{k}', info[k], global_step)

    def eval(
            self,
            save_ckpt_fn: Optional[Callable] = None,
            train_iter: int = -1,
            envstep: int = -1,
            n_episode: Optional[int] = None,
            return_trajectory: bool = False,
    ) -> Dict[str, Any]:
        if torch.cuda.is_available():
            torch.cuda.set_device(get_rank())

        episode_info = None
        stop_flag = False
        per_level_results = defaultdict(list)

        if get_rank() >= 0:
            if n_episode is None:
                n_episode = self._default_n_episode
            assert n_episode is not None
            envstep_count = 0
            eval_monitor = VectorEvalMonitor(self._env.env_num, n_episode)
            env_nums = self._env.env_num

            self._env.reset()
            self._policy.reset(task_id=self.task_id)

            init_obs = self._env.ready_obs
            retry_waiting_time = 0.001
            while len(init_obs.keys()) != self._env_num:
                time.sleep(retry_waiting_time)
                init_obs = self._env.ready_obs

            action_mask_dict = {i: to_ndarray(init_obs[i]['action_mask']) for i in range(env_nums)}
            to_play_dict = {i: to_ndarray(init_obs[i]['to_play']) for i in range(env_nums)}
            timestep_dict = {}
            for i in range(env_nums):
                timestep_dict[i] = to_ndarray(init_obs[i].get('timestep', -1))

            dones = np.array([False for _ in range(env_nums)])
            game_segments = [
                GameSegment(
                    self._env.action_space,
                    game_segment_length=self.policy_config.game_segment_length,
                    config=self.policy_config,
                    task_id=self.task_id,
                ) for _ in range(env_nums)
            ]
            for i in range(env_nums):
                game_segments[i].reset(
                    [to_ndarray(init_obs[i]['observation']) for _ in range(self.policy_config.model.frame_stack_num)]
                )

            ready_env_id = set()
            remain_episode = n_episode
            eps_steps_lst = np.zeros(env_nums)
            total_finishes = 0
            with self._timer:
                while not eval_monitor.is_finished() and total_finishes < n_episode:
                    if self.stop_event.is_set():
                        self._logger.info("[EVALUATOR]: Evaluation aborted due to timeout.")
                        break

                    obs = self._env.ready_obs
                    new_available_env_id = set(obs.keys()).difference(ready_env_id)
                    ready_env_id = ready_env_id.union(set(list(new_available_env_id)[:remain_episode]))
                    remain_episode -= min(len(new_available_env_id), remain_episode)

                    if not ready_env_id:
                        continue

                    stack_obs = {env_id: game_segments[env_id].get_obs() for env_id in ready_env_id}
                    stack_obs = list(stack_obs.values())
                    action_mask = [action_mask_dict[env_id] for env_id in ready_env_id]
                    to_play = [to_play_dict[env_id] for env_id in ready_env_id]
                    timestep = [timestep_dict[env_id] for env_id in ready_env_id]

                    stack_obs = to_ndarray(stack_obs)
                    stack_obs = prepare_observation(stack_obs, self.policy_config.model.model_type)
                    stack_obs = torch.from_numpy(stack_obs).to(self.policy_config.device).float()

                    if self.task_id is None:
                        policy_output = self._policy.forward(stack_obs, action_mask, to_play, ready_env_id=ready_env_id, timestep=timestep)
                    else:
                        policy_output = self._policy.forward(stack_obs, action_mask, to_play, ready_env_id=ready_env_id, timestep=timestep, task_id=self.task_id)

                    actions_with_env_id = {k: v['action'] for k, v in policy_output.items()}
                    distributions_dict_with_env_id = {k: v['visit_count_distributions'] for k, v in policy_output.items()}
                    if self.policy_config.sampled_algo:
                        root_sampled_actions_dict_with_env_id = {k: v['root_sampled_actions'] for k, v in policy_output.items()}
                    value_dict_with_env_id = {k: v['searched_value'] for k, v in policy_output.items()}
                    pred_value_dict_with_env_id = {k: v['predicted_value'] for k, v in policy_output.items()}
                    timestep_dict_with_env_id = {k: v.get('timestep', -1) for k, v in policy_output.items()}
                    visit_entropy_dict_with_env_id = {k: v['visit_count_distribution_entropy'] for k, v in policy_output.items()}

                    actions, distributions_dict, value_dict, pred_value_dict, timestep_dict, visit_entropy_dict = {}, {}, {}, {}, {}, {}
                    if self.policy_config.sampled_algo:
                        root_sampled_actions_dict = {}

                    for index, env_id in enumerate(ready_env_id):
                        actions[env_id] = actions_with_env_id.pop(env_id)
                        distributions_dict[env_id] = distributions_dict_with_env_id.pop(env_id)
                        if self.policy_config.sampled_algo:
                            root_sampled_actions_dict[env_id] = root_sampled_actions_dict_with_env_id.pop(env_id)
                        value_dict[env_id] = value_dict_with_env_id.pop(env_id)
                        pred_value_dict[env_id] = pred_value_dict_with_env_id.pop(env_id)
                        timestep_dict[env_id] = timestep_dict_with_env_id.pop(env_id)
                        visit_entropy_dict[env_id] = visit_entropy_dict_with_env_id.pop(env_id)
                    timesteps = self._env.step(actions)
                    timesteps = to_tensor(timesteps, dtype=torch.float32)
                    for env_id, episode_timestep in timesteps.items():
                        obs_t, reward, done, info = episode_timestep.obs, episode_timestep.reward, episode_timestep.done, episode_timestep.info

                        eps_steps_lst[env_id] += 1
                        if self._policy.get_attribute('cfg').type in ['unizero', 'sampled_unizero']:
                            self._policy.reset(env_id=env_id, current_steps=eps_steps_lst[env_id], reset_init_data=False, task_id=self.task_id)

                        game_segments[env_id].append(
                            actions[env_id], to_ndarray(obs_t['observation']), reward,
                            action_mask_dict[env_id], to_play_dict[env_id], timestep_dict[env_id],
                        )

                        action_mask_dict[env_id] = to_ndarray(obs_t['action_mask'])
                        to_play_dict[env_id] = to_ndarray(obs_t['to_play'])
                        timestep_dict[env_id] = to_ndarray(obs_t.get('timestep', -1))

                        dones[env_id] = done
                        if episode_timestep.done:
                            self._policy.reset([env_id])
                            reward = episode_timestep.info['score']
                            saved_info = {'eval_episode_return': episode_timestep.info['score']}
                            if 'episode_info' in episode_timestep.info:
                                saved_info.update(episode_timestep.info['episode_info'])
                            eval_monitor.update_info(env_id, saved_info)
                            eval_monitor.update_reward(env_id, reward)
                            total_finishes += 1

                            level_id = episode_timestep.info.get('level_id', None)
                            if level_id is not None:
                                per_level_results[int(level_id)].append(float(reward))

                            self._logger.info(
                                f"[EVALUATOR] env {env_id} finished episode (level {level_id}), "
                                f"reward: {reward}, count: {total_finishes}/{n_episode}"
                            )
                            if n_episode > self._env_num:
                                init_obs = self._env.ready_obs
                                while len(init_obs.keys()) != self._env_num:
                                    time.sleep(retry_waiting_time)
                                    init_obs = self._env.ready_obs

                                new_available_env_id = set(init_obs.keys()).difference(ready_env_id)
                                ready_env_id = ready_env_id.union(set(list(new_available_env_id)[:remain_episode]))
                                remain_episode -= min(len(new_available_env_id), remain_episode)

                                action_mask_dict[env_id] = to_ndarray(init_obs[env_id]['action_mask'])
                                to_play_dict[env_id] = to_ndarray(init_obs[env_id]['to_play'])
                                timestep_dict[env_id] = to_ndarray(init_obs[env_id].get('timestep', -1))

                                game_segments[env_id] = GameSegment(
                                    self._env.action_space,
                                    game_segment_length=self.policy_config.game_segment_length,
                                    config=self.policy_config,
                                    task_id=self.task_id,
                                )
                                game_segments[env_id].reset(
                                    [init_obs[env_id]['observation'] for _ in range(self.policy_config.model.frame_stack_num)]
                                )

                            eps_steps_lst[env_id] = 0
                            self._policy.reset([env_id])
                            ready_env_id.remove(env_id)

                        envstep_count += 1

            episode_return = eval_monitor.get_episode_return()
            mean_episode_return = np.mean(episode_return)
            if mean_episode_return >= self._max_episode_return:
                if save_ckpt_fn:
                    save_ckpt_fn('WM_ckpt_best.pth.tar')
                self._max_episode_return = mean_episode_return
            info = {
                'avg_envstep_per_episode': envstep_count / n_episode if n_episode > 0 else 0,
                'reward_mean': np.mean(episode_return),
                'reward_std': np.std(episode_return),
                'reward_max': np.max(episode_return),
                'reward_min': np.min(episode_return),
            }

            self._log_agg_tb(info, 'eval/wm_only/agg', envstep)
            self._log_per_level_tb(dict(per_level_results), 'eval/wm_only/per_level', envstep)

            self._log_agg_tb(info, 'eval/wm_only/agg_iter', train_iter)
            self._log_per_level_tb(dict(per_level_results), 'eval/wm_only/per_level_iter', train_iter)

            self._log_agg_tb(info, 'deprecated/eval/wm_mcts/agg_wm_iter', train_iter)
            self._log_per_level_tb(dict(per_level_results), 'deprecated/eval/wm_mcts/per_level_wm_iter', train_iter)

        return info
