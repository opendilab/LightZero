import time
from collections import deque, namedtuple
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import wandb
from ding.envs import BaseEnvManager
from ding.torch_utils import to_ndarray
from ding.utils import build_logger, EasyTimer, get_rank, get_world_size, broadcast_object_list
from ding.worker.collector.base_serial_evaluator import ISerialEvaluator, VectorEvalMonitor

from lzero.mcts.utils import prepare_observation


class MuZeroEvaluatorPPO(ISerialEvaluator):
    config = dict(
        eval_freq=50,
    )

    def __init__(
            self,
            eval_freq: int = 1000,
            n_evaluator_episode: int = 3,
            stop_value: int = 1e6,
            env: BaseEnvManager = None,
            policy: namedtuple = None,
            tb_logger: 'SummaryWriter' = None,  # noqa
            exp_name: Optional[str] = 'default_experiment',
            instance_name: Optional[str] = 'evaluator',
            policy_config: 'policy_config' = None,  # noqa
    ) -> None:
        self._eval_freq = eval_freq
        self._default_n_episode = n_evaluator_episode
        self._stop_value = stop_value
        self._exp_name = exp_name
        self._instance_name = instance_name
        if get_rank() == 0:
            if tb_logger is not None:
                self._logger, _ = build_logger(
                    './{}/log/{}'.format(self._exp_name, self._instance_name), self._instance_name, need_tb=False
                )
                self._tb_logger = tb_logger
            else:
                self._logger, self._tb_logger = build_logger(
                    './{}/log/{}'.format(self._exp_name, self._instance_name), self._instance_name
                )
        else:
            self._logger, self._tb_logger = None, None

        self.policy_config = policy_config
        self._timer = EasyTimer()
        self.reset(policy, env)

    def reset_env(self, _env: Optional[BaseEnvManager] = None) -> None:
        if _env is not None:
            self._env = _env
            self._env.launch()
            self._env_num = self._env.env_num
        else:
            self._env.reset()

    def reset_policy(self, _policy: Optional[namedtuple] = None) -> None:
        assert hasattr(self, '_env'), "please set env first"
        if _policy is not None:
            self._policy = _policy
        self._policy.reset()

    def reset(self, _policy: Optional[namedtuple] = None, _env: Optional[BaseEnvManager] = None) -> None:
        if _env is not None:
            self.reset_env(_env)
        if _policy is not None:
            self.reset_policy(_policy)
        self._last_eval_iter = 0
        self._max_episode_return = float('-inf')
        self._end_flag = False

    def close(self) -> None:
        if self._end_flag:
            return
        self._end_flag = True
        self._env.close()
        if self._tb_logger:
            self._tb_logger.flush()
            self._tb_logger.close()

    def __del__(self) -> None:
        self.close()

    def should_eval(self, train_iter: int) -> bool:
        if train_iter == self._last_eval_iter:
            return False
        if (train_iter - self._last_eval_iter) < self._eval_freq and train_iter != 0:
            return False
        self._last_eval_iter = train_iter
        return True

    def eval(
            self,
            save_ckpt_fn: Optional[callable] = None,
            train_iter: int = -1,
            envstep: int = -1,
            n_episode: Optional[int] = None,
            return_trajectory: bool = False,
    ) -> Tuple[bool, Dict[str, Any]]:
        episode_info = None
        stop_flag = False
        if get_rank() == 0:
            if n_episode is None:
                n_episode = self._default_n_episode
            self._env.reset()
            self._policy.reset()

            monitor = VectorEvalMonitor(self._env.env_num, n_episode)
            frame_stack = self.policy_config.model.frame_stack_num
            obs_stacks = {
                env_id: deque([to_ndarray(obs['observation']) for _ in range(frame_stack)], maxlen=frame_stack)
                for env_id, obs in self._env.ready_obs.items()
            }

            retry_wait = 0.01
            while len(obs_stacks) != self._env.env_num:
                time.sleep(retry_wait)
                obs_stacks = {
                    env_id: deque([to_ndarray(obs['observation']) for _ in range(frame_stack)], maxlen=frame_stack)
                    for env_id, obs in self._env.ready_obs.items()
                }

            ready_env_id = set()
            remain_episode = n_episode
            episode_returns: List[float] = []

            with self._timer:
                while not monitor.is_finished():
                    obs = self._env.ready_obs
                    new_available = set(obs.keys()).difference(ready_env_id)
                    ready_env_id = ready_env_id.union(set(list(new_available)[:remain_episode]))
                    remain_episode -= min(len(new_available), remain_episode)
                    if not ready_env_id:
                        time.sleep(0.01)
                        continue

                    stacked = [np.array(list(obs_stacks[env_id])) for env_id in ready_env_id]
                    action_masks = [to_ndarray(obs[env_id]['action_mask']) for env_id in ready_env_id]
                    timesteps = [obs[env_id].get('timestep', -1) for env_id in ready_env_id]

                    stacked_np = prepare_observation(stacked, self.policy_config.model.model_type)
                    stacked_tensor = torch.from_numpy(stacked_np).to(self.policy_config.device)

                    policy_output = self._policy.forward(
                        stacked_tensor,
                        action_mask=action_masks,
                        ready_env_id=ready_env_id,
                        timestep=timesteps,
                    )
                    actions = {env_id: policy_output[env_id]['action'] for env_id in ready_env_id}

                    timesteps_output = self._env.step(actions)

                    for env_id, timestep_data in timesteps_output.items():
                        episode_obs = timestep_data.obs
                        reward = float(timestep_data.reward)
                        done = bool(timestep_data.done)
                        next_obs = to_ndarray(episode_obs['observation'])

                        obs_stacks[env_id].append(next_obs)

                        if done:
                            eval_reward = timestep_data.info.get('eval_episode_return', reward)
                            monitor.update_reward(env_id, eval_reward)
                            monitor.update_info(env_id, timestep_data.info.get('episode_info', {}))
                            episode_returns.append(eval_reward)
                            ready_env_id.remove(env_id)
                            remain_episode += 1
                            if remain_episode > 0:
                                obs_reset = self._env.ready_obs
                                obs_stacks[env_id] = deque(
                                    [to_ndarray(obs_reset[env_id]['observation']) for _ in range(frame_stack)],
                                    maxlen=frame_stack
                                )
                                self._policy.reset([env_id])
                        else:
                            continue

            duration = self._timer.value
            reward_mean = float(np.mean(episode_returns)) if episode_returns else 0.0
            reward_std = float(np.std(episode_returns)) if episode_returns else 0.0
            reward_max = float(np.max(episode_returns)) if episode_returns else 0.0
            reward_min = float(np.min(episode_returns)) if episode_returns else 0.0

            info = dict(
                train_iter=train_iter,
                envstep_count=envstep,
                episode_count=len(episode_returns),
                evaluate_time=duration,
                reward_mean=reward_mean,
                reward_std=reward_std,
                reward_max=reward_max,
                reward_min=reward_min,
            )
            if self._logger is not None:
                self._logger.info(self._logger.get_tabulate_vars_hor(info))
            if self._tb_logger is not None:
                for k, v in info.items():
                    if isinstance(v, (int, float)):
                        self._tb_logger.add_scalar('{}_iter/'.format(self._instance_name) + k, v, train_iter)
                        self._tb_logger.add_scalar('{}_step/'.format(self._instance_name) + k, v, envstep)
                    if getattr(self.policy_config, 'use_wandb', False) and isinstance(v, (int, float)):
                        wandb.log({'{}_step/'.format(self._instance_name) + k: v}, step=envstep)

            episode_info = info
            if reward_mean > self._max_episode_return:
                if save_ckpt_fn:
                    save_ckpt_fn('ckpt_best.pth.tar')
                self._max_episode_return = reward_mean
            stop_flag = reward_mean >= self._stop_value and train_iter > 0

        if get_world_size() > 1:
            objects = [stop_flag, episode_info]
            broadcast_object_list(objects, src=0)
            stop_flag, episode_info = objects

        return stop_flag, episode_info
