import os
import time
from collections import deque, namedtuple
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from ding.envs import BaseEnvManager
from ding.torch_utils import to_ndarray
from ding.utils import build_logger, EasyTimer, SERIAL_COLLECTOR_REGISTRY, get_rank, get_world_size, allreduce_data
from ding.worker.collector.base_serial_collector import ISerialCollector

from lzero.mcts.utils import prepare_observation


@SERIAL_COLLECTOR_REGISTRY.register('episode_muzero_ppo')
class MuZeroCollectorPPO(ISerialCollector):
    """Collector that follows the original MuZeroCollector structure but gathers PPO rollouts."""

    config = dict()

    def __init__(
            self,
            collect_print_freq: int = 100,
            env: BaseEnvManager = None,
            policy: namedtuple = None,
            tb_logger: 'SummaryWriter' = None,  # noqa
            exp_name: Optional[str] = 'default_experiment',
            instance_name: Optional[str] = 'collector',
            policy_config: 'policy_config' = None,  # noqa
    ) -> None:
        self._exp_name = exp_name
        self._instance_name = instance_name
        self._collect_print_freq = collect_print_freq
        self._timer = EasyTimer()
        self._end_flag = False

        self._rank = get_rank()
        self._world_size = get_world_size()
        if self._rank == 0:
            if tb_logger is not None:
                self._logger, _ = build_logger(
                    path='./{}/log/{}'.format(self._exp_name, self._instance_name),
                    name=self._instance_name,
                    need_tb=False
                )
                self._tb_logger = tb_logger
            else:
                self._logger, self._tb_logger = build_logger(
                    path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name
                )
        else:
            self._logger, _ = build_logger(
                path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name, need_tb=False
            )
            self._tb_logger = None

        self.policy_config = policy_config
        self.rollout_length = self.policy_config.ppo.rollout_length

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
            self._default_n_episode = _policy.get_attribute('cfg').get('n_episode', None)
            self._logger.debug(
                'Set default n_episode mode(n_episode({}), env_num({}))'.format(self._default_n_episode, self._env_num)
            )
        self._policy.reset()

    def reset(self, _policy: Optional[namedtuple] = None, _env: Optional[BaseEnvManager] = None) -> None:
        if _env is not None:
            self.reset_env(_env)
        if _policy is not None:
            self.reset_policy(_policy)

        self._env_info = {env_id: {'time': 0., 'step': 0} for env_id in range(self._env_num)}
        self._episode_info: List[Dict[str, Any]] = []
        self._pending_buffers = {env_id: [] for env_id in range(self._env_num)}
        self._obs_stacks = {env_id: deque([], maxlen=self.policy_config.model.frame_stack_num) for env_id in range(self._env_num)}
        self._total_envstep_count = 0
        self._total_episode_count = 0
        self._total_duration = 0.
        self._last_train_iter = 0
        self._end_flag = False

    def _reset_stat(self, env_id: int) -> None:
        self._env_info[env_id] = {'time': 0., 'step': 0}
        self._pending_buffers[env_id] = []
        self._obs_stacks[env_id] = deque(maxlen=self.policy_config.model.frame_stack_num)

    @property
    def envstep(self) -> int:
        return self._total_envstep_count

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

    def collect(self,
                train_iter: int = 0,
                policy_kwargs: Optional[dict] = None) -> List[Any]:
        if policy_kwargs is None:
            policy_kwargs = {}

        env_num = self._env_num
        target_episode = policy_kwargs.get('n_episode', self._default_n_episode)
        target_episode = max(target_episode, env_num) if target_episode is not None else env_num
        transitions: List[Dict[str, Any]] = []
        collected_step = 0
        collected_episode = 0
        collected_duration = 0.0

        init_obs = self._env.ready_obs
        retry_waiting_time = 0.01
        while len(init_obs.keys()) != self._env_num:
            time.sleep(retry_waiting_time)
            init_obs = self._env.ready_obs

        frame_stack = self.policy_config.model.frame_stack_num
        obs_stacks = getattr(self, '_obs_stacks', None)
        if obs_stacks is None or not obs_stacks:
            obs_stacks = {
                env_id: deque([to_ndarray(init_obs[env_id]['observation']) for _ in range(frame_stack)], maxlen=frame_stack)
                for env_id in range(env_num)
            }
            self._obs_stacks = obs_stacks
        last_prev_action = {}
        for env_id in range(env_num):
            buffer = self._pending_buffers.get(env_id, [])
            if buffer:
                last_prev_action[env_id] = int(buffer[-1]['action'])
            else:
                last_prev_action[env_id] = -1

        while collected_episode < target_episode:
            ready_obs = self._env.ready_obs
            if not ready_obs:
                time.sleep(0.001)
                continue

            for env_id in ready_obs.keys():
                if len(obs_stacks[env_id]) < frame_stack:
                    obs_value = to_ndarray(ready_obs[env_id]['observation'])
                    obs_stacks[env_id] = deque([obs_value for _ in range(frame_stack)], maxlen=frame_stack)

            ready_env_list = list(ready_obs.keys())

            stacked = [np.array(list(obs_stacks[env_id])) for env_id in ready_env_list]
            action_masks = [to_ndarray(ready_obs[env_id]['action_mask']) for env_id in ready_env_list]
            timesteps = [ready_obs[env_id].get('timestep', -1) for env_id in ready_env_list]

            stacked_np = prepare_observation(stacked, self.policy_config.model.model_type)
            stacked_tensor = torch.from_numpy(stacked_np).to(self.policy_config.device)

            policy_output = self._policy.forward(
                stacked_tensor,
                action_mask=action_masks,
                ready_env_id=ready_env_list,
                timestep=timesteps,
            )

            actions = {env_id: policy_output[env_id]['action'] for env_id in ready_env_list}

            with self._timer:
                timesteps_output = self._env.step(actions)
            interaction_duration = self._timer.value / max(len(timesteps_output), 1)

            for env_id, timestep_data in timesteps_output.items():
                obs_dict = timestep_data.obs
                reward = float(timestep_data.reward)
                done = bool(timestep_data.done)
                next_obs = to_ndarray(obs_dict['observation'])

                info = policy_output[env_id]
                prev_stack = np.array(list(obs_stacks[env_id]))
                prev_action_value = last_prev_action[env_id]

                obs_stacks[env_id].append(next_obs)
                next_stack = np.array(list(obs_stacks[env_id]))
                last_prev_action[env_id] = info['action']

                step_record = dict(
                    prev_obs=prev_stack,
                    obs=next_stack,
                    action_mask=info['action_mask'],
                    action=np.array(info['action'], dtype=np.int64),
                    old_log_prob=np.array(info['log_prob'], dtype=np.float32),
                    value=np.array(info['predicted_value'], dtype=np.float32),
                    reward=np.array(reward, dtype=np.float32),
                    done=np.array(done, dtype=np.float32),
                    prev_action=np.array(prev_action_value, dtype=np.int64),
                    timestep=np.array(info['timestep'], dtype=np.int64),
                )
                self._pending_buffers[env_id].append(step_record)

                self._env_info[env_id]['time'] += interaction_duration
                self._env_info[env_id]['step'] += 1
                collected_step += 1

                if done:
                    episode_transitions, episode_reward = self._finalize_episode(env_id)
                    transitions.extend(episode_transitions)
                    collected_episode += 1
                    collected_duration += self._env_info[env_id]['time']
                    self._episode_info.append({
                        'reward': episode_reward,
                        'step': self._env_info[env_id]['step'],
                        'time': self._env_info[env_id]['time'],
                    })

                    self._policy.reset([env_id])
                    self._reset_stat(env_id)
                    if env_id in ready_env_list:
                        ready_env_list.remove(env_id)

                    last_prev_action[env_id] = -1

        self._obs_stacks = obs_stacks

        if self._world_size > 1:
            collected_step = allreduce_data(collected_step, 'sum')
            collected_episode = allreduce_data(collected_episode, 'sum')
            collected_duration = allreduce_data(collected_duration, 'sum')

        self._total_envstep_count += collected_step
        self._total_episode_count += collected_episode
        self._total_duration += collected_duration

        self._output_log(train_iter)

        return [transitions, {}]

    def _finalize_episode(self, env_id: int) -> Tuple[List[Dict[str, Any]], float]:
        buffer = self._pending_buffers[env_id]
        if not buffer:
            return [], 0.0

        gamma = self.policy_config.ppo.gamma
        gae_lambda = self.policy_config.ppo.gae_lambda

        rewards = np.array([step['reward'] for step in buffer], dtype=np.float32)
        values = np.array([step['value'] for step in buffer], dtype=np.float32)
        dones = np.array([step['done'] for step in buffer], dtype=np.float32)

        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(buffer))):
            next_value = 0.0 if t == len(buffer) - 1 or dones[t] else values[t + 1]
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        transitions: List[Dict[str, Any]] = []
        for t, step in enumerate(buffer):
            transitions.append({
                'prev_obs': step['prev_obs'],
                'obs': step['obs'],
                'action_mask': step['action_mask'],
                'action': step['action'],
                'old_log_prob': step['old_log_prob'],
                'advantage': advantages[t],
                'return': returns[t],
                'prev_action': step['prev_action'],
                'timestep': step['timestep'],
            })

        episode_reward = float(rewards.sum())
        self._pending_buffers[env_id] = []
        return transitions, episode_reward

    def _output_log(self, train_iter: int) -> None:
        if self._rank != 0:
            self._episode_info.clear()
            return
        if self._total_episode_count <= 0:
            return
        if (train_iter - self._last_train_iter) < self._collect_print_freq and train_iter != 0:
            return
        self._last_train_iter = train_iter

        reward_list = [info['reward'] for info in self._episode_info]
        step_list = [info['step'] for info in self._episode_info]
        time_list = [info['time'] for info in self._episode_info]

        avg_reward = float(np.mean(reward_list)) if reward_list else 0.0
        avg_steps = float(np.mean(step_list)) if step_list else 0.0
        avg_time = float(np.mean(time_list)) if time_list else 0.0

        log_str = f"collect iter({train_iter}) envstep({self._total_envstep_count}) episode({self._total_episode_count}) " \
                  f"avg_reward({avg_reward:.3f}) avg_step({avg_steps:.2f}) avg_time({avg_time:.2f})"
        self._logger.info(log_str)
        if self._tb_logger is not None:
            self._tb_logger.add_scalar('collect/avg_reward', avg_reward, train_iter)
            self._tb_logger.add_scalar('collect/avg_step', avg_steps, train_iter)
            self._tb_logger.add_scalar('collect/avg_time', avg_time, train_iter)

        self._episode_info.clear()
