import os
from collections import namedtuple
from ding.envs import BaseEnv
from ding.worker.collector.base_serial_collector import ISerialCollector, CachePool, TrajBuffer, INF, to_tensor_transitions
from typing import Optional, Any, List
from easydict import EasyDict
import numpy as np
from ding.envs import BaseEnvManager
from ding.utils.data import default_decollate
from ding.utils import build_logger, EasyTimer, SERIAL_COLLECTOR_REGISTRY, ENV_REGISTRY
from ding.torch_utils import to_tensor, to_ndarray


@SERIAL_COLLECTOR_REGISTRY.register('episode_alphazero')
class AlphaZeroCollector(ISerialCollector):
    """
    Overview:
        AlphaZero collector (n_episode).
    Interfaces:
        __init__, reset, reset_env, reset_policy, collect, close
    Property:
        envstep
    """
    config = dict(
        deepcopy_obs=False,
        transform_obs=False,
        collect_print_freq=100,
        get_train_sample=False,
        reward_shaping=True,
        augmentation=False
    )

    def __init__(
        self,
        cfg: EasyDict,
        env: BaseEnvManager = None,
        policy: namedtuple = None,
        tb_logger: 'SummaryWriter' = None,  # noqa
        exp_name: Optional[str] = 'default_experiment',
        instance_name: Optional[str] = 'collector',
        replay_buffer: 'replay_buffer' = None,  # noqa
        env_config=None,
    ):
        self._exp_name = exp_name
        self._instance_name = instance_name
        self._collect_print_freq = cfg.collect_print_freq
        self._deepcopy_obs = cfg.deepcopy_obs
        self._transform_obs = cfg.transform_obs
        self._use_augmentation = cfg.augmentation
        self._cfg = cfg
        self._timer = EasyTimer()
        self._end_flag = False
        self._env_config = env_config

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

    def reset_env(self, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Reset the environment.
            If _env is None, reset the old environment.
            If _env is not None, replace the old environment in the collector with the new passed \
                in environment and launch.
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
            Reset the policy.
            If _policy is None, reset the old policy.
            If _policy is not None, replace the old policy in the collector with the new passed in policy.
        Arguments:
            - policy (:obj:`Optional[namedtuple]`): the api namedtuple of collect_mode policy
        """
        assert hasattr(self, '_env'), "please set env first"
        if _policy is not None:
            self._policy = _policy
            self._default_n_episode = _policy.get_attribute('cfg').collect.get('n_episode', None)
            self._unroll_len = _policy.get_attribute('unroll_len')
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
            Reset the environment and policy.
            If _env is None, reset the old environment.
            If _env is not None, replace the old environment in the collector with the new passed \
                in environment and launch.
            If _policy is None, reset the old policy.
            If _policy is not None, replace the old policy in the collector with the new passed in policy.
        Arguments:
            - policy (:obj:`Optional[namedtuple]`): the api namedtuple of collect_mode policy
            - env (:obj:`Optional[BaseEnvManager]`): instance of the subclass of vectorized \
                env_manager(BaseEnvManager)
        """
        if _env is not None:
            self.reset_env(_env)
        if _policy is not None:
            self.reset_policy(_policy)

        self._obs_pool = CachePool('obs', self._env_num, deepcopy=self._deepcopy_obs)
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
            Reset the collector's state. Including reset the traj_buffer, obs_pool, policy_output_pool\
                and env_info. Reset these states according to env_id. You can refer to base_serial_collector\
                to get more messages.
        Arguments:
            - env_id (:obj:`int`): the id where we need to reset the collector's state
        """
        self._traj_buffer[env_id].clear()
        self._obs_pool.reset(env_id)
        self._policy_output_pool.reset(env_id)
        self._env_info[env_id] = {'time': 0., 'step': 0}

    def collect(self,
                n_episode: Optional[int] = None,
                train_iter: int = 0,
                policy_kwargs: Optional[dict] = None) -> List[Any]:
        if n_episode is None:
            if self._default_n_episode is None:
                raise RuntimeError("Please specify collect n_episode")
            else:
                n_episode = self._default_n_episode
        assert n_episode >= self._env_num, "Please make sure n_episode >= env_num{}/{}".format(n_episode, self._env_num)
        if policy_kwargs is None:
            policy_kwargs = {}
        collected_episode = 0
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
                # for env_id in ready_env_id:
                #     print('[collect] env_id = {}'.format(env_id))
                #     print('board = \n {}'.format(self._env._envs[env_id].board))
                obs_ = {env_id: obs[env_id] for env_id in ready_env_id}
                # Policy forward.
                self._obs_pool.update(obs_)
                simulation_envs = {}
                for env_id in ready_env_id:
                    simulation_envs[env_id] = ENV_REGISTRY.build(self._cfg.env.type, self._env_config)
                policy_output = self._policy.forward(simulation_envs, obs_)
                self._policy_output_pool.update(policy_output)
                # Interact with env.
                actions = {env_id: output['action'] for env_id, output in policy_output.items()}
                actions = to_ndarray(actions)
                timesteps = self._env.step(actions)

            # TODO(nyz) this duration may be inaccurate in async env
            interaction_duration = self._timer.value / len(timesteps)
            # TODO(nyz) vectorize this for loop
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
                    self._total_envstep_count += 1
                    # prepare data
                    if timestep.done:
                        transitions = to_tensor_transitions(self._traj_buffer[env_id])
                        if self._cfg.reward_shaping:
                            transitions = self.reward_shaping(transitions)
                        if self._cfg.get_train_sample:
                            train_sample = self._policy.get_train_sample(transitions)
                            return_data.extend(train_sample)
                        else:
                            return_data.append(transitions)
                        self._traj_buffer[env_id].clear()

                self._env_info[env_id]['time'] += self._timer.value + interaction_duration
                if timestep.done:
                    self._total_episode_count += 1
                    if timestep.obs['to_play'] == -1:  # one player mode
                        reward = timestep.info['final_eval_reward']
                    else:
                        if timestep.obs['to_play'] == 1:  # two player mode
                            reward = -timestep.info['final_eval_reward']
                        else:
                            reward = timestep.info['final_eval_reward']
                    reward = timestep.info['final_eval_reward']
                    info = {
                        'reward': reward,  #only means player1 reward
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
        # log
        self._output_log(train_iter)
        return return_data

    @property
    def envstep(self) -> int:
        """
        Overview:
            Print the total envstep count.
        Return:
            - envstep (:obj:`int`): the total envstep count
        """
        return self._total_envstep_count

    def close(self) -> None:
        """
        Overview:
            Close the collector. If end_flag is False, close the environment, flush the tb_logger\
                and close the tb_logger.
        """
        if self._end_flag:
            return
        self._end_flag = True
        self._env.close()
        self._tb_logger.flush()
        self._tb_logger.close()

    def __del__(self) -> None:
        """
        Overview:
            Execute the close command and close the collector. __del__ is automatically called to \
                destroy the collector instance when the collector finishes its work
        """
        self.close()

    def _output_log(self, train_iter: int) -> None:
        """
        Overview:
            Print the output log information. You can refer to Docs/Best Practice/How to understand\
             training generated folders/Serial mode/log/collector for more details.
        Arguments:
            - train_iter (:obj:`int`): the number of training iteration.
        """
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

    def reward_shaping(self, transitions):
        reward = transitions[-1]['reward']
        to_play = transitions[-1]['obs']['to_play']
        for t in transitions:
            if t['obs']['to_play'] == to_play:
                t['reward'] = int(reward)
            else:
                t['reward'] = int(-reward)
        return transitions
