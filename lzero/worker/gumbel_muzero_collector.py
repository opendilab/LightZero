import time
from collections import deque, namedtuple
from typing import Optional, Any, List

import numpy as np
import torch
from ding.envs import BaseEnvManager
from ding.torch_utils import to_ndarray
from ding.utils import build_logger, EasyTimer, SERIAL_COLLECTOR_REGISTRY
from torch.nn import L1Loss

from lzero.worker.muzero_collector import MuZeroCollector
from lzero.mcts.buffer.game_segment import GameSegment
from lzero.mcts.utils import prepare_observation


@SERIAL_COLLECTOR_REGISTRY.register('gumbel_muzero')
class GumbelMuZeroCollector(MuZeroCollector):
    """
    Overview:
        The Collector for MCTS+RL algorithms, including Gumbel Muzero.
    Interfaces:
        __init__, reset, reset_env, reset_policy, collect, close
    Property:
        envstep
    """

    # TO be compatible with MuZeroCollector
    config = dict()

    # ==============================================================
    # MCTS+RL related core code
    # ==============================================================
    def _compute_priorities(self, i, pred_values_lst, search_values_lst):
        """
        Overview:
            obtain the priorities at index i.
        Arguments:
            - i: index.
            - pred_values_lst: The list of value being predicted.
            - search_values_lst: The list of value obtained through search.
        """
        if self.policy_config.use_priority and not self.policy_config.use_max_priority_for_new_data:
            pred_values = torch.from_numpy(np.array(pred_values_lst[i])).to(self.policy_config.device).float().view(-1)
            search_values = torch.from_numpy(np.array(search_values_lst[i])).to(self.policy_config.device
                                                                                ).float().view(-1)
            priorities = L1Loss(reduction='none'
                                )(pred_values,
                                  search_values).detach().cpu().numpy() + self.policy_config.prioritized_replay_eps
        else:
            # priorities is None -> use the max priority for all newly collected data
            priorities = None

        return priorities
    

    def pad_and_save_last_trajectory(self, i, last_game_segments, last_game_priorities, game_segments, done) -> None:
        """
        Overview:
            put the last game block into the pool if the current game is finished
        Arguments:
            - last_game_segments (:obj:`list`): list of the last game segments
            - last_game_priorities (:obj:`list`): list of the last game priorities
            - game_segments (:obj:`list`): list of the current game segments
        Note:
            (last_game_segments[i].obs_segment[-4:][j] == game_segments[i].obs_segment[:4][j]).all() is True
        """
        # pad over last block trajectory
        beg_index = self.policy_config.model.frame_stack_num
        end_index = beg_index + self.policy_config.num_unroll_steps

        # the start <frame_stack_num> obs is init zero obs, so we take the [<frame_stack_num> : <frame_stack_num>+<num_unroll_steps>] obs as the pad obs
        # e.g. the start 4 obs is init zero obs, the num_unroll_steps is 5, so we take the [4:9] obs as the pad obs
        pad_obs_lst = game_segments[i].obs_segment[beg_index:end_index]
        pad_child_visits_lst = game_segments[i].child_visit_segment[:self.policy_config.num_unroll_steps]
        # EfficientZero original repo bug:
        # pad_child_visits_lst = game_segments[i].child_visit_segment[beg_index:end_index]

        beg_index = 0
        # self.unroll_plus_td_steps = self.policy_config.num_unroll_steps + self.policy_config.td_steps
        end_index = beg_index + self.unroll_plus_td_steps - 1

        pad_reward_lst = game_segments[i].reward_segment[beg_index:end_index]

        beg_index = 0
        end_index = beg_index + self.unroll_plus_td_steps

        pad_root_values_lst = game_segments[i].root_value_segment[beg_index:end_index]

        pad_improved_policy_prob = game_segments[i].improved_policy_probs[beg_index:end_index]

        # pad over and save
        last_game_segments[i].pad_over(pad_obs_lst, pad_reward_lst, pad_root_values_lst, pad_child_visits_lst, next_segment_improved_policy = pad_improved_policy_prob)
        """
        Note:
            game_segment element shape:
            obs: game_segment_length + stack + num_unroll_steps, 20+4 +5
            rew: game_segment_length + stack + num_unroll_steps + td_steps -1  20 +5+3-1
            action: game_segment_length -> 20
            root_values:  game_segment_length + num_unroll_steps + td_steps -> 20 +5+3
            child_visits： game_segment_length + num_unroll_steps -> 20 +5
            to_play: game_segment_length -> 20
            action_mask: game_segment_length -> 20
        """

        last_game_segments[i].game_segment_to_array()

        # put the game block into the pool
        self.game_segment_pool.append((last_game_segments[i], last_game_priorities[i], done[i]))

        # reset last game_segments
        last_game_segments[i] = None
        last_game_priorities[i] = None

        return None


    def collect(self,
                n_episode: Optional[int] = None,
                train_iter: int = 0,
                policy_kwargs: Optional[dict] = None) -> List[Any]:
        """
        Overview:
            Collect `n_episode` data with policy_kwargs, which is already trained `train_iter` iterations.
        Arguments:
            - n_episode (:obj:`int`): the number of collecting data episode.
            - train_iter (:obj:`int`): the number of training iteration.
            - policy_kwargs (:obj:`dict`): the keyword args for policy forward.
        Returns:
            - return_data (:obj:`List`): A list containing collected game_segments
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

        retry_waiting_time = 0.001
        while len(init_obs.keys()) != self._env_num:
            # In order to be compatible with subprocess env_manager, in which sometimes self._env_num is not equal to
            # len(self._env.ready_obs), especially in tictactoe env.
            self._logger.info('The current init_obs.keys() is {}'.format(init_obs.keys()))
            self._logger.info('Before sleeping, the _env_states is {}'.format(self._env._env_states))
            time.sleep(retry_waiting_time)
            self._logger.info('=' * 10 + 'Wait for all environments (subprocess) to finish resetting.' + '=' * 10)
            self._logger.info(
                'After sleeping {}s, the current _env_states is {}'.format(retry_waiting_time, self._env._env_states)
            )
            init_obs = self._env.ready_obs

        action_mask_dict = {i: to_ndarray(init_obs[i]['action_mask']) for i in range(env_nums)}
        to_play_dict = {i: to_ndarray(init_obs[i]['to_play']) for i in range(env_nums)}

        game_segments = [
            GameSegment(
                self._env.action_space,
                game_segment_length=self.policy_config.game_segment_length,
                config=self.policy_config
            ) for _ in range(env_nums)
        ]
        # stacked observation windows in reset stage for init game_segments
        observation_window_stack = [[] for _ in range(env_nums)]
        for env_id in range(env_nums):
            observation_window_stack[env_id] = deque(
                [to_ndarray(init_obs[env_id]['observation']) for _ in range(self.policy_config.model.frame_stack_num)],
                maxlen=self.policy_config.model.frame_stack_num
            )

            game_segments[env_id].reset(observation_window_stack[env_id])

        dones = np.array([False for _ in range(env_nums)])
        last_game_segments = [None for _ in range(env_nums)]
        last_game_priorities = [None for _ in range(env_nums)]
        # for priorities in self-play
        search_values_lst = [[] for _ in range(env_nums)]
        pred_values_lst = [[] for _ in range(env_nums)]
        improved_policy_lst = [[] for _ in range(env_nums)]

        # some logs
        eps_steps_lst, visit_entropies_lst = np.zeros(env_nums), np.zeros(env_nums)
        self_play_moves = 0.
        self_play_episodes = 0.
        self_play_moves_max = 0
        self_play_visit_entropy = []
        total_transitions = 0

        ready_env_id = set()
        remain_episode = n_episode

        while True:
            with self._timer:
                # Get current ready env obs.
                obs = self._env.ready_obs
                new_available_env_id = set(obs.keys()).difference(ready_env_id)
                ready_env_id = ready_env_id.union(set(list(new_available_env_id)[:remain_episode]))
                remain_episode -= min(len(new_available_env_id), remain_episode)

                stack_obs = {env_id: game_segments[env_id].get_obs() for env_id in ready_env_id}
                stack_obs = list(stack_obs.values())

                action_mask_dict = {env_id: action_mask_dict[env_id] for env_id in ready_env_id}
                to_play_dict = {env_id: to_play_dict[env_id] for env_id in ready_env_id}
                action_mask = [action_mask_dict[env_id] for env_id in ready_env_id]
                to_play = [to_play_dict[env_id] for env_id in ready_env_id]

                stack_obs = to_ndarray(stack_obs)

                stack_obs = prepare_observation(stack_obs, self.policy_config.model.model_type)

                stack_obs = torch.from_numpy(stack_obs).to(self.policy_config.device).float()

                # ==============================================================
                # policy forward
                # ==============================================================
                policy_output = self._policy.forward(stack_obs, action_mask, temperature, to_play)

                actions_no_env_id = {k: v['action'] for k, v in policy_output.items()}
                distributions_dict_no_env_id = {k: v['distributions'] for k, v in policy_output.items()}
                if self.policy_config.sampled_algo:
                    root_sampled_actions_dict_no_env_id = {
                        k: v['root_sampled_actions']
                        for k, v in policy_output.items()
                    }
                value_dict_no_env_id = {k: v['value'] for k, v in policy_output.items()}
                # ==============================================================
                # Gumbel Muzero related code
                # ==============================================================
                improved_policy_dict_no_env_id = {k: v['improved_policy_probs'] for k, v in policy_output.items()}
                pred_value_dict_no_env_id = {k: v['pred_value'] for k, v in policy_output.items()}
                visit_entropy_dict_no_env_id = {
                    k: v['visit_count_distribution_entropy']
                    for k, v in policy_output.items()
                }

                # TODO(pu): subprocess
                actions = {}
                distributions_dict = {}
                if self.policy_config.sampled_algo:
                    root_sampled_actions_dict = {}
                value_dict = {}
                improved_policy_dict = {}
                pred_value_dict = {}
                visit_entropy_dict = {}
                for index, env_id in enumerate(ready_env_id):
                    actions[env_id] = actions_no_env_id.pop(index)
                    distributions_dict[env_id] = distributions_dict_no_env_id.pop(index)
                    if self.policy_config.sampled_algo:
                        root_sampled_actions_dict[env_id] = root_sampled_actions_dict_no_env_id.pop(index)
                    value_dict[env_id] = value_dict_no_env_id.pop(index)
                    improved_policy_dict[env_id] = improved_policy_dict_no_env_id.pop(index)
                    pred_value_dict[env_id] = pred_value_dict_no_env_id.pop(index)
                    visit_entropy_dict[env_id] = visit_entropy_dict_no_env_id.pop(index)

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
                    obs, reward, done, info = timestep.obs, timestep.reward, timestep.done, timestep.info

                    if self.policy_config.sampled_algo:
                        game_segments[env_id].store_search_stats(
                            distributions_dict[env_id], value_dict[env_id], root_sampled_actions_dict[env_id], improved_policy = improved_policy_dict[env_id]
                        )
                    else:
                        game_segments[env_id].store_search_stats(distributions_dict[env_id], value_dict[env_id], improved_policy = improved_policy_dict[env_id])
                    # append a transition tuple, including a_t, o_{t+1}, r_{t}, action_mask_{t}, to_play_{t}
                    # in ``game_segments[env_id].init``, we have append o_{t} in ``self.obs_segment``
                    game_segments[env_id].append(
                        actions[env_id], to_ndarray(obs['observation']), reward, action_mask_dict[env_id],
                        to_play_dict[env_id]
                    )

                    # NOTE: the position of code snippet is very important.
                    # the obs['action_mask'] and obs['to_play'] is corresponding to next action
                    action_mask_dict[env_id] = to_ndarray(obs['action_mask'])
                    to_play_dict[env_id] = to_ndarray(obs['to_play'])

                    dones[env_id] = done
                    visit_entropies_lst[env_id] += visit_entropy_dict[env_id]

                    eps_steps_lst[env_id] += 1
                    total_transitions += 1

                    if self.policy_config.use_priority and not self.policy_config.use_max_priority_for_new_data:
                        pred_values_lst[env_id].append(pred_value_dict[env_id])
                        search_values_lst[env_id].append(value_dict[env_id])
                        improved_policy_lst[env_id].append(improved_policy_dict[env_id])

                    # append the newest obs
                    observation_window_stack[env_id].append(to_ndarray(obs['observation']))

                    # ==============================================================
                    # we will save a game block if it is the end of the game or the next game block is finished.
                    # ==============================================================

                    # if game block is full, we will save the last game block
                    if game_segments[env_id].is_full():
                        # pad over last block trajectory
                        if last_game_segments[env_id] is not None:
                            # TODO(pu): return the one game block
                            self.pad_and_save_last_trajectory(
                                env_id, last_game_segments, last_game_priorities, game_segments, dones
                            )

                        # calculate priority
                        priorities = self._compute_priorities(env_id, pred_values_lst, search_values_lst)
                        pred_values_lst[env_id] = []
                        search_values_lst[env_id] = []
                        improved_policy_lst[env_id] = []

                        # the current game_segments become last_game_segment
                        last_game_segments[env_id] = game_segments[env_id]
                        last_game_priorities[env_id] = priorities

                        # create new GameSegment
                        game_segments[env_id] = GameSegment(
                            self._env.action_space,
                            game_segment_length=self.policy_config.game_segment_length,
                            config=self.policy_config
                        )
                        game_segments[env_id].reset(observation_window_stack[env_id])

                    self._env_info[env_id]['step'] += 1
                    self._total_envstep_count += 1

                self._env_info[env_id]['time'] += self._timer.value + interaction_duration

                if timestep.done:
                    self._total_episode_count += 1
                    reward = timestep.info['eval_episode_return']
                    info = {
                        'reward': reward,
                        'time': self._env_info[env_id]['time'],
                        'step': self._env_info[env_id]['step'],
                        'visit_entropy': visit_entropies_lst[env_id] / eps_steps_lst[env_id],
                    }
                    collected_episode += 1
                    self._episode_info.append(info)

                    # ==============================================================
                    # if it is the end of the game, we will save the game block
                    # ==============================================================

                    # NOTE: put the penultimate game block in one episode into the trajectory_pool
                    # pad over 2th last game_segment using the last game_segment
                    if last_game_segments[env_id] is not None:
                        self.pad_and_save_last_trajectory(
                            env_id, last_game_segments, last_game_priorities, game_segments, dones
                        )

                    # store current block trajectory
                    priorities = self._compute_priorities(env_id, pred_values_lst, search_values_lst)

                    # NOTE: put the last game block in one episode into the trajectory_pool
                    game_segments[env_id].game_segment_to_array()

                    # assert len(game_segments[env_id]) == len(priorities)
                    # NOTE: save the last game block in one episode into the trajectory_pool if it's not null
                    if len(game_segments[env_id].reward_segment) != 0:
                        self.game_segment_pool.append((game_segments[env_id], priorities, dones[env_id]))

                    # print(game_segments[env_id].reward_segment)
                    # reset the finished env and init game_segments
                    if n_episode > self._env_num:
                        # Get current ready env obs.
                        init_obs = self._env.ready_obs
                        retry_waiting_time = 0.001
                        while len(init_obs.keys()) != self._env_num:
                            # In order to be compatible with subprocess env_manager, in which sometimes self._env_num is not equal to
                            # len(self._env.ready_obs), especially in tictactoe env.
                            self._logger.info('The current init_obs.keys() is {}'.format(init_obs.keys()))
                            self._logger.info('Before sleeping, the _env_states is {}'.format(self._env._env_states))
                            time.sleep(retry_waiting_time)
                            self._logger.info(
                                '=' * 10 + 'Wait for all environments (subprocess) to finish resetting.' + '=' * 10
                            )
                            self._logger.info(
                                'After sleeping {}s, the current _env_states is {}'.format(
                                    retry_waiting_time, self._env._env_states
                                )
                            )
                            init_obs = self._env.ready_obs

                        new_available_env_id = set(init_obs.keys()).difference(ready_env_id)
                        ready_env_id = ready_env_id.union(set(list(new_available_env_id)[:remain_episode]))
                        remain_episode -= min(len(new_available_env_id), remain_episode)

                        action_mask_dict[env_id] = to_ndarray(init_obs[env_id]['action_mask'])
                        to_play_dict[env_id] = to_ndarray(init_obs[env_id]['to_play'])

                        game_segments[env_id] = GameSegment(
                            self._env.action_space,
                            game_segment_length=self.policy_config.game_segment_length,
                            config=self.policy_config
                        )
                        observation_window_stack[env_id] = deque(
                            [init_obs[env_id]['observation'] for _ in range(self.policy_config.model.frame_stack_num)],
                            maxlen=self.policy_config.model.frame_stack_num
                        )
                        game_segments[env_id].reset(observation_window_stack[env_id])
                        last_game_segments[env_id] = None
                        last_game_priorities[env_id] = None

                    # log
                    self_play_moves_max = max(self_play_moves_max, eps_steps_lst[env_id])
                    self_play_visit_entropy.append(visit_entropies_lst[env_id] / eps_steps_lst[env_id])
                    self_play_moves += eps_steps_lst[env_id]
                    self_play_episodes += 1

                    pred_values_lst[env_id] = []
                    search_values_lst[env_id] = []
                    improved_policy_lst[env_id] = []
                    eps_steps_lst[env_id] = 0
                    visit_entropies_lst[env_id] = 0

                    # Env reset is done by env_manager automatically
                    self._policy.reset([env_id])
                    self._reset_stat(env_id)
                    # TODO(pu): subprocess mode, when n_episode > self._env_num, occasionally the ready_env_id=()
                    # and the stack_obs is np.array(None, dtype=object)
                    ready_env_id.remove(env_id)

            if collected_episode >= n_episode:
                # [data, meta_data]
                return_data = [self.game_segment_pool[i][0] for i in range(len(self.game_segment_pool))], [
                    {
                        'priorities': self.game_segment_pool[i][1],
                        'done': self.game_segment_pool[i][2],
                        'unroll_plus_td_steps': self.unroll_plus_td_steps
                    } for i in range(len(self.game_segment_pool))
                ]
                # for i in range(len(self.game_segment_pool)):
                #     print(self.game_segment_pool[i][0].obs_segment.__len__())
                #     print(self.game_segment_pool[i][0].reward_segment)
                # for i in range(len(return_data[0])):
                #     print(return_data[0][i].reward_segment)
                break
        # log
        self._output_log(train_iter)
        return return_data