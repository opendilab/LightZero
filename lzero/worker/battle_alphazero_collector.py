from collections import namedtuple
from typing import Optional, Any, List, Tuple

import numpy as np
from ding.envs import BaseEnvManager
from ding.torch_utils import to_ndarray
from ding.utils import build_logger, EasyTimer, SERIAL_COLLECTOR_REGISTRY
from ding.worker.collector.base_serial_collector import ISerialCollector, CachePool, TrajBuffer, INF, \
    to_tensor_transitions
from easydict import EasyDict
from zoo.board_games.go.envs.go_bot_policy_v0 import GoBotPolicyV0
from zoo.board_games.gomoku.envs.gomoku_bot_policy_v0 import GomokuBotPolicyV0
from zoo.board_games.tictactoe.envs.tictactoe_bot_policy_v0 import TictactoeBotPolicyV0


@SERIAL_COLLECTOR_REGISTRY.register('episode_alphazero_battle')
class BattleAlphaZeroCollector(ISerialCollector):
    """
    Overview:
        Episode collector(n_episode) with two policy battle
    Interfaces:
        __init__, reset, reset_env, reset_policy, collect, close
    Property:
        envstep
    """

    config = dict(deepcopy_obs=False, transform_obs=False, collect_print_freq=100, get_train_sample=False)

    def __init__(
            self,
            cfg: EasyDict,
            env: BaseEnvManager = None,
            policy: List[namedtuple] = None,
            tb_logger: 'SummaryWriter' = None,  # noqa
            exp_name: Optional[str] = 'default_experiment',
            instance_name: Optional[str] = 'collector'
    ) -> None:
        """
        Overview:
            Initialization method.
        Arguments:
            - cfg (:obj:`EasyDict`): Config dict
            - env (:obj:`BaseEnvManager`): the subclass of vectorized env_manager(BaseEnvManager)
            - policy (:obj:`List[namedtuple]`): the api namedtuple of collect_mode policy
            - tb_logger (:obj:`SummaryWriter`): tensorboard handle
        """
        self._exp_name = exp_name
        self._instance_name = instance_name
        self._collect_print_freq = cfg.collect_print_freq
        self._deepcopy_obs = cfg.deepcopy_obs
        self._transform_obs = cfg.transform_obs
        self._cfg = cfg
        self._timer = EasyTimer()
        self._end_flag = False

        if tb_logger is not None:
            self._logger, _ = build_logger(
                path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name, need_tb=False
            )
            self._tb_logger = tb_logger
        else:
            self._logger, self._tb_logger = build_logger(
                path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name
            )
        self._traj_len = float("inf")
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

    def reset_policy(self, _policy: Optional[List[namedtuple]] = None) -> None:
        """
        Overview:
            Reset the policy.
            If _policy is None, reset the old policy.
            If _policy is not None, replace the old policy in the collector with the new passed in policy.
        Arguments:
            - policy (:obj:`Optional[List[namedtuple]]`): the api namedtuple of collect_mode policy
        """
        assert hasattr(self, '_env'), "please set env first"
        if _policy is not None:
            assert len(_policy) == 2, "1v1 episode collector needs 2 policy, but found {}".format(len(_policy))
            self._policy = _policy
            self._default_n_episode = _policy[0].get_attribute('cfg').collect.get('n_episode', None)
            # self._unroll_len = _policy[0].get_attribute('unroll_len')
            # self._on_policy = _policy[0].get_attribute('cfg').on_policy
            self._traj_len = INF
            self._logger.debug(
                'Set default n_episode mode(n_episode({}), env_num({}), traj_len({}))'.format(
                    self._default_n_episode, self._env_num, self._traj_len
                )
            )
        for p in self._policy:
            if isinstance(p, dict):
                p['policy'].reset()
            else:
                p.reset()

    def reset(self, _policy: Optional[List[namedtuple]] = None, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Reset the environment and policy.
            If _env is None, reset the old environment.
            If _env is not None, replace the old environment in the collector with the new passed \
                in environment and launch.
            If _policy is None, reset the old policy.
            If _policy is not None, replace the old policy in the collector with the new passed in policy.
        Arguments:
            - policy (:obj:`Optional[List[namedtuple]]`): the api namedtuple of collect_mode policy
            - env (:obj:`Optional[BaseEnvManager]`): instance of the subclass of vectorized \
                env_manager(BaseEnvManager)
        """
        if _env is not None:
            self.reset_env(_env)
        if _policy is not None:
            self.reset_policy(_policy)

        self._obs_pool = CachePool('obs', self._env_num, deepcopy=self._deepcopy_obs)
        self._policy_output_pool = CachePool('policy_output', self._env_num)
        # _traj_buffer is {env_id: {policy_id: TrajBuffer}}, is used to store traj_len pieces of transitions
        self._traj_buffer = {
            env_id: {policy_id: TrajBuffer(maxlen=self._traj_len)
                     for policy_id in range(2)}
            for env_id in range(self._env_num)
        }
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
        for i in range(2):
            self._traj_buffer[env_id][i].clear()
        self._obs_pool.reset(env_id)
        self._policy_output_pool.reset(env_id)
        self._env_info[env_id] = {'time': 0., 'step': 0}

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

    def collect(self,
                n_episode: Optional[int] = None,
                train_iter: int = 0,
                policy_kwargs: Optional[dict] = None) -> Tuple[List[Any], List[Any]]:
        """
        Overview:
            Collect `n_episode` data with policy_kwargs, which is already trained `train_iter` iterations
        Arguments:
            - n_episode (:obj:`int`): the number of collecting data episode
            - train_iter (:obj:`int`): the number of training iteration
            - policy_kwargs (:obj:`dict`): the keyword args for policy forward
        Returns:
            - return_data (:obj:`Tuple[List, List]`): A tuple with training sample(data) and episode info, \
                the former is a list containing collected episodes if not get_train_sample, \
                otherwise, return train_samples split by unroll_len.
        """
        if n_episode is None:
            if self._default_n_episode is None:
                raise RuntimeError("Please specify collect n_episode")
            else:
                n_episode = self._default_n_episode
        assert n_episode >= self._env_num, "Please make sure n_episode >= env_num"
        if policy_kwargs is None:
            policy_kwargs = {}
        temperature = policy_kwargs['temperature']

        collected_episode = 0
        return_data = [[] for _ in range(2)]
        return_info = [[] for _ in range(2)]
        ready_env_id = set()
        remain_episode = n_episode

        while True:
            # for policy_id, policy in enumerate(self._policy):
            with self._timer:
                # Get current env obs.
                obs = self._env.ready_obs
                new_available_env_id = set(obs.keys()).difference(ready_env_id)
                ready_env_id = ready_env_id.union(set(list(new_available_env_id)[:remain_episode]))
                remain_episode -= min(len(new_available_env_id), remain_episode)

                obs_ = {env_id: obs[env_id] for env_id in ready_env_id}
                # Policy forward.
                self._obs_pool.update(obs_)
                simulation_envs = {}
                for env_id in ready_env_id:
                    # create the new simulation env instances from the current collect env using the same env_config.
                    simulation_envs[env_id] = self._env._env_fn[env_id]()

                # ==============================================================
                # policy forward
                # ==============================================================
                # policy_output = policy.forward(simulation_envs, obs_, temperature)

                obs_player_1 = {}
                obs_player_2 = {}
                simulation_envs_player_1 = {}
                simulation_envs_player_2 = {}
                ready_env_id_player_1 = []
                ready_env_id_player_2 = []
                for k, v in obs_.items():
                    if v['to_play'] == 1:
                        obs_player_1[k] = v
                        simulation_envs_player_1[k] = simulation_envs[k]
                        ready_env_id_player_1.append(k)
                    elif v['to_play'] == 2:
                        obs_player_2[k] = v
                        simulation_envs_player_2[k] = simulation_envs[k]
                        ready_env_id_player_2.append(k)

                if len(ready_env_id_player_1) > 0:
                    if isinstance(self._policy[0], dict):
                        policy_output_player_1 = self._policy[0]['policy'].forward(simulation_envs_player_1, obs_player_1,
                                                                         temperature)
                    else:
                        policy_output_player_1 = self._policy[0].forward(simulation_envs_player_1, obs_player_1,
                                                                         temperature)
                else:
                    policy_output_player_1 = {}
                if len(ready_env_id_player_2) > 0:
                    if isinstance(self._policy[1], dict):
                        policy_output_player_2 = self._policy[1]['policy'].forward(simulation_envs_player_2, obs_player_2,
                                                                         temperature)
                    else:
                        policy_output_player_2 = self._policy[1].forward(simulation_envs_player_2, obs_player_2,
                                                                         temperature)
                else:
                    policy_output_player_2 = {}

                policy_output = {}
                policy_output.update(policy_output_player_1)
                policy_output.update(policy_output_player_2)

                self._policy_output_pool.update(policy_output)
                # Interact with env.
                actions = {env_id: output['action'] for env_id, output in policy_output.items()}
                actions = to_ndarray(actions)

                # ==============================================================
                # Interact with env.
                # ==============================================================
                timesteps = self._env.step(actions)

            try:
                interaction_duration = self._timer.value / len(timesteps)
            except ZeroDivisionError:
                interaction_duration = 0.

            for env_id, timestep in timesteps.items():
                self._env_info[env_id]['step'] += 1
                self._total_envstep_count += 1
                if env_id in ready_env_id_player_1:
                    policy_id = 0
                elif env_id in ready_env_id_player_2:
                    policy_id = 1
                with self._timer:
                    if isinstance(self._policy[policy_id], dict):
                        if self._policy[policy_id]['policy_type'] in ['bot', 'historical']:
                            # The data produced by bot and historical policy is not used for training.
                            pass
                        elif self._policy[policy_id]['policy_type'] == 'main':
                            transition = self._policy[policy_id]['policy'].process_transition(
                                self._obs_pool[env_id], self._policy_output_pool[env_id],
                                timestep
                            )
                            transition['collect_iter'] = train_iter
                            self._traj_buffer[env_id][policy_id].append(transition)
                    else:
                        transition = self._policy[policy_id].process_transition(
                            self._obs_pool[env_id], self._policy_output_pool[env_id],
                            timestep
                        )
                        transition['collect_iter'] = train_iter
                        self._traj_buffer[env_id][policy_id].append(transition)

                    # prepare data
                    if timestep.done:
                        for policy_id in range(2):
                            if len(self._traj_buffer[env_id][policy_id]) > 0:
                                transitions = to_tensor_transitions(
                                    self._traj_buffer[env_id][policy_id], not self._deepcopy_obs
                                )
                                # reward_shaping
                                transitions = self.reward_shaping(transitions, timestep.info['eval_episode_return'])

                                return_data[policy_id].append(transitions)
                                self._traj_buffer[env_id][policy_id].clear()

                self._env_info[env_id]['time'] += self._timer.value + interaction_duration

                # If env is done, record episode info and reset
                if timestep.done:
                    self._total_episode_count += 1
                    # the eval_episode_return is calculated from Player 1's perspective
                    reward = timestep.info['eval_episode_return']
                    info = {
                        'reward': reward,
                        'time': self._env_info[env_id]['time'],
                        'step': self._env_info[env_id]['step'],
                    }
                    collected_episode += 1
                    self._episode_info.append(info)
                    for i, p in enumerate(self._policy):
                        # p.reset([env_id])
                        if isinstance(p, dict):
                            p['policy'].reset([env_id])
                        else:
                            p.reset([env_id])

                    self._reset_stat(env_id)
                    ready_env_id.remove(env_id)
                    for policy_id in range(2):
                        # return_info[policy_id].append(timestep.info[policy_id])
                        return_info[policy_id].append(timestep.info)

                    # break the agent loop
                    # break

            if collected_episode >= n_episode:
                break
        # log
        self._output_log(train_iter)
        return return_data, return_info

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
            # episode_return0 = [d['reward0'] for d in self._episode_info]
            # episode_return1 = [d['reward1'] for d in self._episode_info]
            episode_reward = [d['reward'] for d in self._episode_info]

            self._total_duration += duration
            info = {
                'episode_count': episode_count,
                'envstep_count': envstep_count,
                'avg_envstep_per_episode': envstep_count / episode_count,
                'avg_envstep_per_sec': envstep_count / duration,
                'avg_episode_per_sec': episode_count / duration,
                'collect_time': duration,

                # 'reward0_mean': np.mean(episode_return0),
                # 'reward0_std': np.std(episode_return0),
                # 'reward0_max': np.max(episode_return0),
                # 'reward0_min': np.min(episode_return0),
                # 'reward1_mean': np.mean(episode_return1),
                # 'reward1_std': np.std(episode_return1),
                # 'reward1_max': np.max(episode_return1),
                # 'reward1_min': np.min(episode_return1),

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
                self._tb_logger.add_scalar('{}_iter/'.format(self._instance_name) + k, v, train_iter)
                if k in ['total_envstep_count']:
                    continue
                self._tb_logger.add_scalar('{}_step/'.format(self._instance_name) + k, v, self._total_envstep_count)

    def reward_shaping(self, transitions, eval_episode_return):
        """
        Overview:
            Shape the reward according to the player.
        Return:
            - transitions: data transitions.
        """
        reward = transitions[-1]['reward']
        to_play = transitions[-1]['obs']['to_play']
        for t in transitions:
            if t['obs']['to_play'] == -1:
                # play_with_bot_mode
                # the eval_episode_return is calculated from Player 1's perspective
                t['reward'] = eval_episode_return
            else:
                # self_play_mode
                if t['obs']['to_play'] == to_play:
                    t['reward'] = int(reward)
                else:
                    t['reward'] = int(-reward)
        return transitions
