import time
from collections import deque, namedtuple
from typing import Optional, Any, List

import numpy as np
import torch
from ding.envs import BaseEnvManager
from ding.torch_utils import to_ndarray
from ding.utils import build_logger, EasyTimer, SERIAL_COLLECTOR_REGISTRY
from ding.worker.collector.base_serial_collector import ISerialCollector, CachePool, TrajBuffer, INF
from easydict import EasyDict
from torch.nn import L1Loss

from lzero.mcts.tree_search.game import GameHistory
from lzero.mcts.utils import prepare_observation_list


@SERIAL_COLLECTOR_REGISTRY.register('episode_efficientzero')
class EfficientZeroCollector(ISerialCollector):
    """
    Overview:
        EfficientZero collector(n_episode)
    Interfaces:
        __init__, reset, reset_env, reset_policy, collect, close
    Property:
        envstep
    """

    config = dict(game_block_pool_size=int(1e6), deepcopy_obs=False, transform_obs=False, collect_print_freq=100,
                  get_train_sample=False)

    def __init__(
            self,
            cfg: EasyDict,
            env: BaseEnvManager = None,
            policy: namedtuple = None,
            tb_logger: 'SummaryWriter' = None,  # noqa
            exp_name: Optional[str] = 'default_experiment',
            instance_name: Optional[str] = 'collector',
            replay_buffer: 'replay_buffer' = None,  # noqa
            game_config: 'game_config' = None,  # noqa
    ) -> None:
        """
        Overview:
            Initialization method.
        Arguments:
            - cfg (:obj:`EasyDict`): Config dict
            - env (:obj:`BaseEnvManager`): the subclass of vectorized env_manager(BaseEnvManager)
            - policy (:obj:`namedtuple`): the api namedtuple of collect_mode policy
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

        self.replay_buffer = replay_buffer
        self.game_config = game_config

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
        # A game_block_pool implementation based on the deque structure.
        self.game_block_pool = deque(maxlen=self._cfg.game_block_pool_size)

        self.pool_size = 1
        self.unroll_plus_td_steps = self.game_config.num_unroll_steps + self.game_config.td_steps
        self.last_model_index = -1

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

    # ==============================================================
    # MCTS related method
    # ==============================================================
    def get_priorities(self, i, pred_values_lst, search_values_lst):
        """
        Overview:
            obtain the priorities at index i.
        """
        if self.game_config.use_priority and not self.game_config.use_max_priority_for_new_data:
            pred_values = torch.from_numpy(np.array(pred_values_lst[i])).to(self.game_config.device).float().view(-1)
            search_values = torch.from_numpy(np.array(search_values_lst[i])).to(self.game_config.device
                                                                                ).float().view(-1)
            priorities = L1Loss(reduction='none'
                                )(pred_values,
                                  search_values).detach().cpu().numpy() + self.game_config.prioritized_replay_eps
        else:
            # priorities is None -> use the max priority for all newly collected data
            priorities = None

        return priorities

    def pad_and_save_last_trajectory(self, i, last_game_blocks, last_game_priorities, game_blocks, done):
        """
        Overview:
            put the last game history into the pool if the current game is finished
        Arguments:
            - last_game_blocks (:obj:`list`): list of the last game histories
            - last_game_priorities (:obj:`list`): list of the last game priorities
            - game_blocks (:obj:`list`): list of the current game histories
        Note:
            (last_game_blocks[i].obs_history[-4:][j] == game_blocks[i].obs_history[:4][j]).all() is True
        """
        # pad over last block trajectory
        beg_index = self.game_config.model.frame_stack_num
        end_index = beg_index + self.game_config.num_unroll_steps

        # the start <frame_stack_num> obs is init zero obs, so we take the [<frame_stack_num> : <frame_stack_num>+<num_unroll_steps>] obs as the pad obs
        # e.g. the start 4 obs is init zero obs, the num_unroll_steps is 5, so we take the [4:9] obs as the pad obs
        pad_obs_lst = game_blocks[i].obs_history[beg_index:end_index]
        pad_child_visits_lst = game_blocks[i].child_visit_history[:self.game_config.num_unroll_steps]
        # EfficientZero bug?
        # pad_child_visits_lst = game_blocks[i].child_visit_history[beg_index:end_index]

        beg_index = 0
        # self.unroll_plus_td_steps = self.game_config.num_unroll_steps + self.game_config.td_steps
        end_index = beg_index + self.unroll_plus_td_steps - 1

        pad_reward_lst = game_blocks[i].reward_history[beg_index:end_index]

        beg_index = 0
        end_index = beg_index + self.unroll_plus_td_steps

        pad_root_values_lst = game_blocks[i].root_value_history[beg_index:end_index]

        # pad over and save
        last_game_blocks[i].pad_over(pad_obs_lst, pad_reward_lst, pad_root_values_lst, pad_child_visits_lst)
        """
        Note:
            game_block element shape:
            obs: game_block_length + stack + num_unroll_steps, 20+4 +5
            rew: game_block_length + stack + num_unroll_steps + td_steps -1  20 +5+3-1
            action: game_block_length -> 20
            root_values:  game_block_length + num_unroll_steps + td_steps -> 20 +5+3
            child_visits： game_block_length + num_unroll_steps -> 20 +5
            to_play: game_block_length -> 20
            action_mask: game_block_length -> 20
        """

        last_game_blocks[i].game_block_to_array()

        # put the game history into the pool
        self.game_block_pool.append((last_game_blocks[i], last_game_priorities[i], done[i]))

        # reset last game_blocks
        last_game_blocks[i] = None
        last_game_priorities[i] = None

        return None

    def collect(self,
                n_episode: Optional[int] = None,
                train_iter: int = 0,
                policy_kwargs: Optional[dict] = None) -> List[Any]:
        """
        Overview:
            Collect `n_episode` data with policy_kwargs, which is already trained `train_iter` iterations
        Arguments:
            - n_episode (:obj:`int`): the number of collecting data episode
            - train_iter (:obj:`int`): the number of training iteration
            - policy_kwargs (:obj:`dict`): the keyword args for policy forward
        Returns:
            - return_data (:obj:`List`): A list containing collected episodes if not get_train_sample, otherwise, \
                return train_samples split by unroll_len.
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
        env_nums = self._env_num

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

        action_mask = [to_ndarray(init_obs[i]['action_mask']) for i in range(env_nums)]
        action_mask_dict = {i: to_ndarray(init_obs[i]['action_mask']) for i in range(env_nums)}

        if 'to_play' in init_obs[0]:
            two_player_game = True
        else:
            two_player_game = False

        if two_player_game:
            to_play = [to_ndarray(init_obs[i]['to_play']) for i in range(env_nums)]
            to_play_dict = {i: to_ndarray(init_obs[i]['to_play']) for i in range(env_nums)}

        dones = np.array([False for _ in range(env_nums)])
        game_blocks = [
            GameHistory(
                self._env.action_space,
                game_block_length=self.game_config.game_block_length,
                config=self.game_config
            ) for _ in range(env_nums)
        ]

        last_game_blocks = [None for _ in range(env_nums)]
        last_game_priorities = [None for _ in range(env_nums)]

        # stacked observation windows in reset stage for init game_blocks
        stack_obs_windows = [[] for _ in range(env_nums)]
        for i in range(env_nums):
            stack_obs_windows[i] = [
                to_ndarray(init_obs[i]['observation']) for _ in range(self.game_config.model.frame_stack_num)
            ]
            game_blocks[i].init(stack_obs_windows[i])

        # for priorities in self-play
        search_values_lst = [[] for _ in range(env_nums)]
        pred_values_lst = [[] for _ in range(env_nums)]

        # some logs
        eps_steps_lst, visit_entropies_lst = np.zeros(env_nums), np.zeros(env_nums)

        self_play_rewards = 0.
        self_play_ori_rewards = 0.
        self_play_moves = 0.
        self_play_episodes = 0.

        self_play_rewards_max = -np.inf
        self_play_moves_max = 0

        self_play_visit_entropy = []
        total_transitions = 0

        def _get_max_entropy(action_space):
            p = 1.0 / action_space
            ep = -action_space * p * np.log2(p)
            return ep

        max_visit_entropy = _get_max_entropy(self.game_config.model.action_space_size)
        # print('max_visit_entropy', max_visit_entropy)

        ready_env_id = set()
        remain_episode = n_episode
        # new_available_env_id = set(init_obs.keys()).difference(ready_env_id)
        # ready_env_id = ready_env_id.union(set(list(new_available_env_id)[:remain_episode]))
        # remain_episode -= min(len(new_available_env_id), remain_episode)

        while True:
            with self._timer:
                # stack_obs = [game_block.step_obs() for game_block in game_blocks]

                # Get current ready env obs.
                # only for subprocess, to get the ready_env_id
                obs = self._env.ready_obs
                # TODO(pu): subprocess
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
                    policy_output = self._policy.forward(stack_obs, action_mask, temperature, to_play)
                else:
                    policy_output = self._policy.forward(stack_obs, action_mask, temperature, None)

                actions_no_env_id = {k: v['action'] for k, v in policy_output.items()}
                distributions_dict_no_env_id = {k: v['distributions'] for k, v in policy_output.items()}
                value_dict_no_env_id = {k: v['value'] for k, v in policy_output.items()}
                pred_value_dict_no_env_id = {k: v['pred_value'] for k, v in policy_output.items()}
                visit_entropy_dict_no_env_id = {
                    k: v['visit_count_distribution_entropy']
                    for k, v in policy_output.items()
                }

                # TODO(pu): subprocess
                actions = {}
                distributions_dict = {}
                value_dict = {}
                pred_value_dict = {}
                visit_entropy_dict = {}
                for index, env_id in enumerate(ready_env_id):
                    actions[env_id] = actions_no_env_id.pop(index)
                    distributions_dict[env_id] = distributions_dict_no_env_id.pop(index)
                    value_dict[env_id] = value_dict_no_env_id.pop(index)
                    pred_value_dict[env_id] = pred_value_dict_no_env_id.pop(index)
                    visit_entropy_dict[env_id] = visit_entropy_dict_no_env_id.pop(index)

                # Interact with env.
                timesteps = self._env.step(actions)
                # for debug
                # if len(timesteps.keys())!=self._env_num:
                #     print(f'current ready env id is {list(timesteps.keys())}')

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
                    i = env_id
                    obs, reward, done, info = timestep.obs, timestep.reward, timestep.done, timestep.info

                    game_blocks[env_id].store_search_stats(distributions_dict[env_id], value_dict[env_id])
                    if two_player_game:
                        # for two_player board games
                        # append a transition tuple, including a_t, o_{t+1}, r_{t}, action_mask_{t}, to_play_{t}
                        # in ``game_blocks[env_id].init``, we have append o_{t} in ``self.obs_history``
                        game_blocks[env_id].append(
                            actions[env_id], to_ndarray(obs['observation']), reward, action_mask_dict[env_id],
                            to_play_dict[env_id]
                        )
                    else:
                        game_blocks[env_id].append(actions[env_id], to_ndarray(obs['observation']), reward)

                    # NOTE: the position of code snippet is very important.
                    # the obs['action_mask'] and obs['to_play'] is corresponding to next action
                    if two_player_game:
                        action_mask_dict[env_id] = to_ndarray(obs['action_mask'])
                        to_play_dict[env_id] = to_ndarray(obs['to_play'])

                    dones[env_id] = done
                    visit_entropies_lst[env_id] += visit_entropy_dict[env_id]

                    eps_steps_lst[env_id] += 1
                    total_transitions += 1

                    if self.game_config.use_priority and not self.game_config.use_max_priority_for_new_data:
                        pred_values_lst[env_id].append(pred_value_dict[env_id])
                        search_values_lst[env_id].append(value_dict[env_id])

                    # updte stack windows: delete the first obs and append the newest obs
                    del stack_obs_windows[env_id][0]
                    stack_obs_windows[env_id].append(to_ndarray(obs['observation']))

                    # ==============================================================
                    # we will save a game history if it is the end of the game or the next game history is finished.
                    # ==============================================================

                    # if game history is full, we will save the last game history
                    if game_blocks[env_id].is_full():
                        # pad over last block trajectory
                        if last_game_blocks[env_id] is not None:
                            # TODO(pu): return the one game history
                            self.pad_and_save_last_trajectory(
                                i, last_game_blocks, last_game_priorities, game_blocks, dones
                            )

                        # calculate priority
                        priorities = self.get_priorities(i, pred_values_lst, search_values_lst)
                        pred_values_lst[env_id] = []
                        search_values_lst[env_id] = []

                        # the current game_blocks become last_game_block
                        last_game_blocks[env_id] = game_blocks[env_id]
                        last_game_priorities[env_id] = priorities

                        # create new GameHistory
                        game_blocks[env_id] = GameHistory(
                            self._env.action_space,
                            game_block_length=self.game_config.game_block_length,
                            config=self.game_config
                        )
                        game_blocks[env_id].init(stack_obs_windows[env_id])
                        # print(game_blocks[env_id].reward_history)

                        # TODO(pu): return data

                    self._env_info[env_id]['step'] += 1
                    self._total_envstep_count += 1

                self._env_info[env_id]['time'] += self._timer.value + interaction_duration

                if timestep.done:
                    self._total_episode_count += 1
                    reward = timestep.info['final_eval_reward']
                    info = {
                        'reward': reward,
                        'time': self._env_info[env_id]['time'],
                        'step': self._env_info[env_id]['step'],
                        'visit_entropy': visit_entropies_lst[env_id] / eps_steps_lst[env_id],
                    }
                    collected_episode += 1
                    self._episode_info.append(info)

                    # ==============================================================
                    # if it is the end of the game, we will save the game history
                    # ==============================================================

                    # NOTE: put the penultimate game history in one episode into the trajectory_pool
                    # pad over 2th last game_block using the last game_block
                    if last_game_blocks[env_id] is not None:
                        self.pad_and_save_last_trajectory(
                            i, last_game_blocks, last_game_priorities, game_blocks, dones
                        )

                    # store current block trajectory
                    priorities = self.get_priorities(i, pred_values_lst, search_values_lst)

                    # NOTE: put the last game history in one episode into the trajectory_pool
                    game_blocks[env_id].game_block_to_array()

                    # assert len(game_blocks[env_id]) == len(priorities)
                    # NOTE: save the last game history in one episode into the trajectory_pool if it's not null
                    if len(game_blocks[env_id].reward_history) != 0:
                        self.game_block_pool.append((game_blocks[env_id], priorities, dones[env_id]))

                    # print(game_blocks[env_id].reward_history)
                    # TODO(pu)
                    # reset the finished env and init game_blocks
                    if n_episode > self._env_num:
                        init_obs = self._env.ready_obs

                        if len(init_obs.keys()) != self._env_num:
                            while env_id not in init_obs.keys():
                                init_obs = self._env.ready_obs
                                print(f'wait the {env_id} env to reset')

                        init_obs = init_obs[env_id]['observation']
                        #  init_obs [0]['observation']
                        init_obs = to_ndarray(init_obs)
                        action_mask_dict[env_id] = to_ndarray(init_obs[env_id]['action_mask'])
                        to_play_dict[env_id] = to_ndarray(init_obs[env_id]['to_play'])

                        game_blocks[env_id] = GameHistory(
                            self._env.action_space,
                            game_block_length=self.game_config.game_block_length,
                            config=self.game_config
                        )
                        stack_obs_windows[env_id] = [init_obs for _ in range(self.game_config.model.frame_stack_num)]
                        game_blocks[env_id].init(stack_obs_windows[env_id])
                        last_game_blocks[env_id] = None
                        last_game_priorities[env_id] = None

                    # log
                    self_play_moves_max = max(self_play_moves_max, eps_steps_lst[env_id])
                    self_play_visit_entropy.append(visit_entropies_lst[env_id] / eps_steps_lst[env_id])
                    self_play_moves += eps_steps_lst[env_id]
                    self_play_episodes += 1

                    pred_values_lst[env_id] = []
                    search_values_lst[env_id] = []
                    eps_steps_lst[env_id] = 0
                    visit_entropies_lst[env_id] = 0

                    # Env reset is done by env_manager automatically
                    self._policy.reset([env_id])
                    self._reset_stat(env_id)
                    # TODO(pu): subprocess
                    ready_env_id.remove(env_id)

            if collected_episode >= n_episode:
                # logs
                visit_entropies = np.array(self_play_visit_entropy).mean()
                # for debug
                # visit_entropies /= max_visit_entropy
                # print('visit_entropies:', visit_entropies)

                return_data = [self.game_block_pool[i][0] for i in range(len(self.game_block_pool))], [
                    {
                        'priorities': self.game_block_pool[i][1],
                        'end_tag': self.game_block_pool[i][2],
                        'unroll_plus_td_stepss': self.unroll_plus_td_steps
                    } for i in range(len(self.game_block_pool))
                ]
                """
                for i in range(len(self.game_block_pool)):
                    print(self.game_block_pool[i][0].obs_history.__len__())
                    print(self.game_block_pool[i][0].reward_history)
                    
                for i in range(len(return_data[0])):
                    print(return_data[0][i].reward_history)
    
                """
                # ==============================================================
                # save the game histories and clear the pool
                # ==============================================================
                # self.game_block_pool: list of (game_block, priority)
                # self.replay_buffer.push_games(
                #     [self.game_block_pool[i][0] for i in range(len(self.game_block_pool))], [
                #         {
                #             'priorities': self.game_block_pool[i][1],
                #             'end_tag': self.game_block_pool[i][2],
                #             'unroll_plus_td_stepss': self.unroll_plus_td_steps
                #         } for i in range(len(self.game_block_pool))
                #     ]
                # )

                # ==============================================================
                # save the buffer to .npy file
                # ==============================================================
                # np.save('/Users/puyuan/code/DI-engine/dizoo/board_games/atari/config/one_episode_replay_buffer_img',
                #         self.replay_buffer.buffer)
                # one_episode_replay_buffer_img = np.load('/Users/puyuan/code/DI-engine/dizoo/board_games/atari/config/one_episode_replay_buffer_img.npy',
                #         allow_pickle=True)

                # np.save('/Users/puyuan/code/DI-engine/dizoo/board_games/tictactoe/config/one_episode_replay_buffer_tictactoe_2-player-mode',
                #         self.replay_buffer.buffer)
                # one_episode_replay_buffer_tictactoe_2playermode = np.load('/Users/puyuan/code/DI-engine/dizoo/board_games/tictactoe/config/one_episode_replay_buffer_tictactoe_2-player-mode.npy', allow_pickle=True)

                break
        # log
        self._output_log(train_iter)
        return return_data

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
            visit_entropy = [d['visit_entropy'] for d in self._episode_info]
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
                'visit_entropy': np.mean(visit_entropy),
                # 'each_reward': episode_reward,
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
