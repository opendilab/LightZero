from collections import namedtuple
from typing import Optional, Any, List

import numpy as np
from ding.envs import BaseEnvManager
from ding.torch_utils import to_ndarray
from ding.utils import build_logger, EasyTimer, SERIAL_COLLECTOR_REGISTRY, get_rank, get_world_size, \
    allreduce_data
from ding.worker.collector.base_serial_collector import ISerialCollector, CachePool, TrajBuffer, INF, \
    to_tensor_transitions


@SERIAL_COLLECTOR_REGISTRY.register('episode_alphazero')
class AlphaZeroCollector(ISerialCollector):
    """
    Overview:
        AlphaZero collector for collecting episodes of experience during self-play or playing against an opponent.
        This collector is specifically designed for the AlphaZero algorithm.
    Interfaces:
        ``__init__``, ``reset``, ``reset_env``, ``reset_policy``, ``collect``, ``close``
    Property:
        envstep
    """

    # TO be compatible with ISerialCollector
    config = dict()

    def __init__(
            self,
            collect_print_freq: int = 100,
            env: BaseEnvManager = None,
            policy: namedtuple = None,
            tb_logger: 'SummaryWriter' = None,  # noqa
            exp_name: Optional[str] = 'default_experiment',
            instance_name: Optional[str] = 'collector',
            env_config=None,
    ) -> None:
        """
        Overview:
            Initialize the AlphaZero collector with the provided environment, policy, and configurations.
        Arguments:
            - collect_print_freq (:obj:`int`): Frequency of printing collection statistics (in training steps).
            - env (:obj:`Optional[BaseEnvManager]`): Environment manager for managing multiple environments.
            - policy (:obj:`Optional[namedtuple]`): Policy used for making decisions during collection.
            - tb_logger (:obj:`Optional[SummaryWriter]`): TensorBoard logger for logging statistics.
            - exp_name (:obj:`str`): Name of the experiment for logging purposes.
            - instance_name (:obj:`str`): Unique identifier for this collector instance.
            - env_config (:obj:`Optional[dict]`): Configuration for the environment.
        """
        self._exp_name = exp_name
        self._instance_name = instance_name
        self._collect_print_freq = collect_print_freq
        self._timer = EasyTimer()
        self._end_flag = False
        self._env_config = env_config

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

        self.reset(policy, env)

    def reset_env(self, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Reset or replace the environment in the collector.
            If _env is None, reset the old environment.
            If _env is not None, replace the old environment in the collector with the new passed \
                in environment and launch.
        Arguments:
            - _env (:obj:`Optional[BaseEnvManager]`): New environment to replace the existing one, if provided.
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
            Reset or replace the policy in the collector.
            If _policy is None, reset the old policy.
            If _policy is not None, replace the old policy in the collector with the new passed in policy.
        Arguments:
            - _policy (:obj:`Optional[namedtuple]`): New policy to replace the existing one, if provided.
        """
        assert hasattr(self, '_env'), "please set env first"
        if _policy is not None:
            self._policy = _policy
            self._default_n_episode = _policy.get_attribute('cfg').get('n_episode', None)
            self._on_policy = _policy.get_attribute('cfg').on_policy
            self._traj_len = INF
            self._logger.debug(
                'Set default n_episode mode(n_episode({}), env_num({}), traj_len({}))'.format(
                    self._default_n_episode, self._env_num, self._traj_len
                )
            )
        self._policy.reset()

    def reset(self, _policy: Optional[namedtuple] = None, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Reset the environment and policy within the collector.
            If _env is None, reset the old environment.
            If _env is not None, replace the old environment in the collector with the new passed \
                in environment and launch.
            If _policy is None, reset the old policy.
            If _policy is not None, replace the old policy in the collector with the new passed in policy.
        Arguments:
            - _policy (:obj:`Optional[namedtuple]`): New policy to replace the existing one, if provided.
            - _env (:obj:`Optional[BaseEnvManager]`): New environment to replace the existing one, if provided.
        """
        if _env is not None:
            self.reset_env(_env)
        if _policy is not None:
            self.reset_policy(_policy)

        self._obs_pool = CachePool('obs', self._env_num, deepcopy=False)
        self._policy_output_pool = CachePool('policy_output', self._env_num)
        # _traj_buffer is {env_id: TrajBuffer}, is used to store traj_len pieces of transitions
        self._traj_buffer = {env_id: TrajBuffer(maxlen=self._traj_len) for env_id in range(self._env_num)}
        self._env_info = {env_id: {'time': 0., 'step': 0} for env_id in range(self._env_num)}

        self._episode_info = []
        self._total_envstep_count = 0
        self._total_episode_count = 0
        self._total_duration = 0
        self._last_train_iter = 0
        self._end_flag = False

    def _reset_stat(self, env_id: int) -> None:
        """
        Overview:
            Reset the statistics for a specific environment.
            Including reset the traj_buffer, obs_pool, policy_output_pool and env_info.
            Reset these states according to env_id.
            You can refer to base_serial_collector to get more messages.
        Arguments:
            - env_id (:obj:`int`): the id where we need to reset the collector's state
        """
        self._traj_buffer[env_id].clear()
        self._obs_pool.reset(env_id)
        self._policy_output_pool.reset(env_id)
        self._env_info[env_id] = {'time': 0., 'step': 0}

    def close(self) -> None:
        """
        Overview:
            Close the collector. If end_flag is False, close the environment, flush the tb_logger
            and close the tb_logger.
        """
        if self._end_flag:
            return
        self._end_flag = True
        self._env.close()
        if self._tb_logger:
            self._tb_logger.flush()
            self._tb_logger.close()

    def collect(self,
                n_episode: Optional[int] = None,
                train_iter: int = 0,
                policy_kwargs: Optional[dict] = None) -> List[Any]:
        """
        Overview:
            Collect experience data for a specified number of episodes using the current policy.
        Arguments:
            - n_episode (:obj:`Optional[int]`): Number of episodes to collect. Defaults to a pre-set value if None.
            - train_iter (:obj:`int`): Current training iteration.
            - policy_kwargs (:obj:`Optional[dict]`): Additional keyword arguments for the policy.
        Returns:
            - return_data (:obj:`List[Any]`): A list of collected experience episodes.
        """
        if n_episode is None:
            if self._default_n_episode is None:
                raise RuntimeError("Please specify collect n_episode")
            else:
                n_episode = self._default_n_episode
        assert n_episode >= self._env_num, "Please make sure n_episode >= env_num{}/{}".format(n_episode, self._env_num)
        if policy_kwargs is None:
            policy_kwargs = {}
        temperature = policy_kwargs['temperature']
        collected_episode = 0
        collected_step = 0
        return_data = []
        ready_env_id = set()
        remain_episode = n_episode

        while True:
            with self._timer:
                # Get current env obs.
                obs = self._env.ready_obs
                new_available_env_id = set(obs.keys()).difference(ready_env_id)
                ready_env_id = ready_env_id.union(set(list(new_available_env_id)[:remain_episode]))
                remain_episode -= min(len(new_available_env_id), remain_episode)
                obs_ = {env_id: obs[env_id] for env_id in ready_env_id}
                # Policy forward.
                self._obs_pool.update(obs_)

                # ==============================================================
                # policy forward
                # ==============================================================
                policy_output = self._policy.forward(obs_, temperature)
                self._policy_output_pool.update(policy_output)
                # Interact with env.
                actions = {env_id: output['action'] for env_id, output in policy_output.items()}
                actions = to_ndarray(actions)
                # ==============================================================
                # Interact with env.
                # ==============================================================
                timesteps = self._env.step(actions)

            interaction_duration = self._timer.value / len(timesteps)
            for env_id, timestep in timesteps.items():
                with self._timer:
                    if timestep.info.get('abnormal', False):
                        # If there is an abnormal timestep, reset all the related variables(including this env).
                        # suppose there is no reset param, just reset this env
                        self._env.reset({env_id: None})
                        self._policy.reset([env_id])
                        self._reset_stat(env_id)
                        self._logger.info('Env{} returns a abnormal step, its info is {}'.format(env_id, timestep.info))
                        continue

                    transition = self._policy.process_transition(
                        self._obs_pool[env_id], self._policy_output_pool[env_id], timestep
                    )
                    transition['collect_iter'] = train_iter
                    self._traj_buffer[env_id].append(transition)
                    self._env_info[env_id]['step'] += 1
                    collected_step += 1

                    # prepare data
                    if timestep.done:
                        transitions = to_tensor_transitions(self._traj_buffer[env_id])
                        # reward_shaping
                        transitions = self.reward_shaping(transitions, timestep.info['eval_episode_return'])

                        return_data.append(transitions)
                        self._traj_buffer[env_id].clear()

                self._env_info[env_id]['time'] += self._timer.value + interaction_duration
                if timestep.done:
                    # the eval_episode_return is calculated from Player 1's perspective
                    reward = timestep.info['eval_episode_return']
                    info = {
                        'reward': reward,  # only means player1 reward
                        'time': self._env_info[env_id]['time'],
                        'step': self._env_info[env_id]['step'],
                    }
                    collected_episode += 1
                    self._episode_info.append(info)
                    self._policy.reset([env_id])
                    self._reset_stat(env_id)
                    ready_env_id.remove(env_id)

            if collected_episode >= n_episode:
                break

        collected_duration = sum([d['time'] for d in self._episode_info])
        # reduce data when enables DDP
        if self._world_size > 1:
            collected_step = allreduce_data(collected_step, 'sum')
            collected_episode = allreduce_data(collected_episode, 'sum')
            collected_duration = allreduce_data(collected_duration, 'sum')
        self._total_envstep_count += collected_step
        self._total_episode_count += collected_episode
        self._total_duration += collected_duration

        # log
        self._output_log(train_iter)
        return return_data

    @property
    def envstep(self) -> int:
        """
        Overview:
            Get the total number of environment steps taken by the collector.
        Returns:
            - envstep (:obj:`int`): Total count of environment steps.
        """
        return self._total_envstep_count

    def close(self) -> None:
        """
        Overview:
            Close the collector and clean up resources such as environment and logger.
        """
        if self._end_flag:
            return
        self._end_flag = True
        self._env.close()
        if self._tb_logger:
            self._tb_logger.flush()
            self._tb_logger.close()

    def __del__(self) -> None:
        """
        Overview:
            Destructor method that is called when the collector object is being destroyed.
        """
        self.close()

    def _output_log(self, train_iter: int) -> None:
        """
        Overview:
            Output logging information for the current collection phase.
        Arguments:
            - train_iter (:obj:`int`): Current training iteration for logging purposes.
        """
        if self._rank != 0:
            return
        if (train_iter - self._last_train_iter) >= self._collect_print_freq and len(self._episode_info) > 0:
            self._last_train_iter = train_iter
            episode_count = len(self._episode_info)
            envstep_count = sum([d['step'] for d in self._episode_info])
            duration = sum([d['time'] for d in self._episode_info])
            episode_reward = [d['reward'] for d in self._episode_info]
            self._total_duration += duration
            info = {
                'episode_count': episode_count,
                'envstep_count': envstep_count,
                'avg_envstep_per_episode': envstep_count / episode_count,
                'avg_envstep_per_sec': envstep_count / duration,
                'avg_episode_per_sec': episode_count / duration,
                'collect_time': duration,
                'reward_mean': np.mean(episode_reward),
                'reward_std': np.std(episode_reward),
                'reward_max': np.max(episode_reward),
                'reward_min': np.min(episode_reward),
                'total_envstep_count': self._total_envstep_count,
                'total_episode_count': self._total_episode_count,
                'total_duration': self._total_duration,
            }
            self._episode_info.clear()
            self._logger.info("collect end:\n{}".format('\n'.join(['{}: {}'.format(k, v) for k, v in info.items()])))
            for k, v in info.items():
                if k in ['each_reward']:
                    continue
                self._tb_logger.add_scalar('{}_iter/'.format(self._instance_name) + k, v, train_iter)
                if k in ['total_envstep_count']:
                    continue
                self._tb_logger.add_scalar('{}_step/'.format(self._instance_name) + k, v, self._total_envstep_count)

    def reward_shaping(self, transitions, eval_episode_return):
        """
        Overview:
            Shape the rewards in the collected transitions based on the outcome of the episode.
        Return:
            - transitions (:obj:`List[dict]`): List of data transitions.
        """
        reward = transitions[-1]['reward']
        to_play = transitions[-1]['obs'].get('to_play', -1)

        if to_play == -1:
            return self._play_with_bot_mode(transitions, eval_episode_return)
        else:
            return self._self_play_mode(transitions, eval_episode_return, to_play, reward)

    def _play_with_bot_mode(self, transitions, eval_episode_return):
        """
        Play with bot mode: All rewards are shaped based on eval_episode_return.
        """
        for t in transitions:
            t['reward'] = eval_episode_return
        return transitions

    def _self_play_mode(self, transitions, eval_episode_return, to_play, reward):
        """
        Self play mode: Reward shaping depends on the player's perspective.
        """
        for t in transitions:
            current_to_play = t['obs'].get('to_play', -1)
            if current_to_play == to_play:
                t['reward'] = int(reward)
            else:
                t['reward'] = int(-reward)
        return transitions
