import numpy as np
import torch
from ding.torch_utils import to_ndarray, to_tensor, to_device
from ding.utils import EasyTimer
from lzero.mcts.buffer.game_segment import GameSegment
from lzero.mcts.utils import prepare_observation


class MuZeroCollector:

    def __init__(self, cfg, policy, env):
        self._cfg = cfg.policy
        self._env = env
        self._env.seed(cfg.seed)
        self._policy = policy

        self._timer = EasyTimer()
        self._trajectory_pool = []
        self._default_n_episode = self._cfg.n_episode
        self._unroll_plus_td_steps = self._cfg.num_unroll_steps + self._cfg.td_steps
        self._last_collect_iter = 0

    def __call__(self, ctx):
        trained_iter = ctx.train_iter - self._last_collect_iter
        if ctx.train_iter != 0 and trained_iter < self._cfg.update_per_collect:
            return
        elif trained_iter == self._cfg.update_per_collect:
            self._last_collect_iter = ctx.train_iter
        n_episode = self._default_n_episode
        temperature = ctx.collect_kwargs['temperature']
        collected_episode = 0
        env_nums = self._env.env_num
        if self._env.closed:
            self._env.launch()
        else:
            self._env.reset()
        self._policy.reset()

        init_obs = self._env.ready_obs
        action_mask_dict = {i: to_ndarray(init_obs[i]['action_mask']) for i in range(env_nums)}

        dones = np.array([False for _ in range(env_nums)])
        game_segments = [
            GameSegment(self._env.action_space, game_segment_length=self._cfg.game_segment_length, config=self._cfg)
            for _ in range(env_nums)
        ]

        last_game_segments = [None for _ in range(env_nums)]
        last_game_priorities = [None for _ in range(env_nums)]

        # stacked observation windows in reset stage for init game_segments
        stack_obs_windows = [[] for _ in range(env_nums)]
        for i in range(env_nums):
            stack_obs_windows[i] = [
                to_ndarray(init_obs[i]['observation']) for _ in range(self._cfg.model.frame_stack_num)
            ]
            game_segments[i].reset(stack_obs_windows[i])

        # for priorities in self-play
        search_values_lst = [[] for _ in range(env_nums)]
        pred_values_lst = [[] for _ in range(env_nums)]

        # some logs
        eps_ori_reward_lst, eps_reward_lst, eps_steps_lst, visit_entropies_lst = np.zeros(env_nums), np.zeros(
            env_nums
        ), np.zeros(env_nums), np.zeros(env_nums)

        ready_env_id = set()
        remain_episode = n_episode

        return_data = []
        while True:
            with self._timer:
                obs = self._env.ready_obs
                new_available_env_id = set(obs.keys()).difference(ready_env_id)
                ready_env_id = ready_env_id.union(set(list(new_available_env_id)[:remain_episode]))
                remain_episode -= min(len(new_available_env_id), remain_episode)

                action_mask_dict = {env_id: action_mask_dict[env_id] for env_id in ready_env_id}
                action_mask = [action_mask_dict[env_id] for env_id in ready_env_id]

                stack_obs = [game_segments[env_id].get_obs() for env_id in ready_env_id]
                stack_obs = to_ndarray(stack_obs)
                stack_obs = prepare_observation(stack_obs, self._cfg.model.model_type)
                stack_obs = to_tensor(stack_obs)
                stack_obs = to_device(stack_obs, self._cfg.device)

                policy_output = self._policy.forward(stack_obs, action_mask, temperature)

                actions = {k: v['action'] for k, v in zip(ready_env_id, policy_output)}
                distributions_dict = {k: v['distributions'] for k, v in zip(ready_env_id, policy_output)}
                value_dict = {k: v['value'] for k, v in zip(ready_env_id, policy_output)}
                pred_value_dict = {k: v['pred_value'] for k, v in zip(ready_env_id, policy_output)}
                visit_entropy_dict = {
                    k: v['visit_count_distribution_entropy']
                    for k, v in zip(ready_env_id, policy_output)
                }

                timesteps = self._env.step(actions)
                ctx.env_step += len(ready_env_id)

            for env_id, timestep in timesteps.items():
                with self._timer:
                    i = env_id
                    obs, rew, done = timestep.obs, timestep.reward, timestep.done
                    game_segments[env_id].store_search_stats(distributions_dict[env_id], value_dict[env_id])
                    game_segments[env_id].append(
                        actions[env_id], to_ndarray(obs['observation']), rew, action_mask_dict[env_id]
                    )

                    action_mask_dict[env_id] = to_ndarray(obs['action_mask'])
                    eps_reward_lst[env_id] += rew
                    dones[env_id] = done
                    visit_entropies_lst[env_id] += visit_entropy_dict[env_id]

                    eps_steps_lst[env_id] += 1

                    if self._cfg.use_priority and not self._cfg.use_max_priority_for_new_data:
                        pred_values_lst[env_id].append(pred_value_dict[env_id])
                        search_values_lst[env_id].append(value_dict[env_id])

                    del stack_obs_windows[env_id][0]
                    stack_obs_windows[env_id].append(to_ndarray(obs['observation']))

                    #########
                    # we will save a game history if it is the end of the game or the next game history is finished.
                    #########

                    #########
                    # if game history is full, we will save the last game history
                    #########
                    if game_segments[env_id].is_full():
                        # pad over last block trajectory
                        if last_game_segments[env_id] is not None:
                            # TODO(pu): return the one game history
                            self.pad_and_save_last_trajectory(
                                i, last_game_segments, last_game_priorities, game_segments, dones
                            )

                        # calculate priority
                        priorities = self.get_priorities(i, pred_values_lst, search_values_lst)
                        pred_values_lst[env_id] = []
                        search_values_lst[env_id] = []

                        # the current game_segments become last_game_segment
                        last_game_segments[env_id] = game_segments[env_id]
                        last_game_priorities[env_id] = priorities

                        # create new GameSegment
                        game_segments[env_id] = GameSegment(
                            self._env.action_space, game_segment_length=self._cfg.game_segment_length, config=self._cfg
                        )
                        game_segments[env_id].reset(stack_obs_windows[env_id])

                if timestep.done:
                    collected_episode += 1

                    #########
                    # if it is the end of the game, we will save the game history
                    #########

                    # NOTE: put the penultimate game history in one episode into the _trajectory_pool
                    # pad over 2th last game_segment using the last game_segment
                    if last_game_segments[env_id] is not None:
                        self.pad_and_save_last_trajectory(
                            i, last_game_segments, last_game_priorities, game_segments, dones
                        )

                    # store current block trajectory
                    priorities = self.get_priorities(i, pred_values_lst, search_values_lst)

                    # NOTE: put the last game history in one episode into the _trajectory_pool
                    game_segments[env_id].game_segment_to_array()

                    # assert len(game_segments[env_id]) == len(priorities)
                    # NOTE: save the last game history in one episode into the _trajectory_pool if it's not null
                    if len(game_segments[env_id].reward_segment) != 0:
                        self._trajectory_pool.append((game_segments[env_id], priorities, dones[env_id]))

                    # reset the finished env and init game_segments
                    if n_episode > env_nums:
                        init_obs = self._env.ready_obs

                        if len(init_obs.keys()) != env_nums:
                            while env_id not in init_obs.keys():
                                init_obs = self._env.ready_obs
                                print(f'wait the {env_id} env to reset')

                        init_obs = init_obs[env_id]['observation']
                        init_obs = to_ndarray(init_obs)
                        action_mask_dict[env_id] = to_ndarray(init_obs[env_id]['action_mask'])

                        game_segments[env_id] = GameSegment(
                            self._env.action_space, game_segment_length=self._cfg.game_segment_length, config=self._cfg
                        )
                        stack_obs_windows[env_id] = [init_obs for _ in range(self._cfg.model.frame_stack_num)]
                        game_segments[env_id].init(stack_obs_windows[env_id])
                        last_game_segments[env_id] = None
                        last_game_priorities[env_id] = None

                    pred_values_lst[env_id] = []
                    search_values_lst[env_id] = []
                    eps_steps_lst[env_id] = 0
                    eps_reward_lst[env_id] = 0
                    eps_ori_reward_lst[env_id] = 0
                    visit_entropies_lst[env_id] = 0

                    self._policy.reset([env_id])
                    ready_env_id.remove(env_id)

            if collected_episode >= n_episode:
                L = len(self._trajectory_pool)
                return_data = [self._trajectory_pool[i][0] for i in range(L)], [
                    {
                        'priorities': self._trajectory_pool[i][1],
                        'done': self._trajectory_pool[i][2],
                        'unroll_plus_td_steps': self._unroll_plus_td_steps
                    } for i in range(L)
                ]

                del self._trajectory_pool[:]
                break
        ctx.trajectories = return_data

    def pad_and_save_last_trajectory(self, i, last_game_segments, last_game_priorities, game_segments, done):
        """
        Overview:
            put the last game history into the pool if the current game is finished
        Arguments:
            - last_game_segments (:obj:`list`): list of the last game histories
            - last_game_priorities (:obj:`list`): list of the last game priorities
            - game_segments (:obj:`list`): list of the current game histories
        Note:
            (last_game_segments[i].obs_history[-4:][j] == game_segments[i].obs_history[:4][j]).all() is True
        """
        # pad over last block trajectory
        beg_index = self._cfg.model.frame_stack_num
        end_index = beg_index + self._cfg.num_unroll_steps

        # the start <frame_stack_num> obs is init zero obs, so we take the [<frame_stack_num> : <frame_stack_num>+<num_unroll_steps>] obs as the pad obs
        # e.g. the start 4 obs is init zero obs, the num_unroll_steps is 5, so we take the [4:9] obs as the pad obs
        pad_obs_lst = game_segments[i].obs_segment[beg_index:end_index]
        pad_child_visits_lst = game_segments[i].child_visit_segment[:self._cfg.num_unroll_steps]
        # pad_child_visits_lst = game_segments[i].child_visit_history[beg_index:end_index]

        beg_index = 0
        # self._unroll_plus_td_steps = self._cfg.num_unroll_steps + self._cfg.td_steps
        end_index = beg_index + self._unroll_plus_td_steps - 1

        pad_reward_lst = game_segments[i].reward_segment[beg_index:end_index]

        beg_index = 0
        end_index = beg_index + self._unroll_plus_td_steps

        pad_root_values_lst = game_segments[i].root_value_segment[beg_index:end_index]

        # pad over and save
        last_game_segments[i].pad_over(pad_obs_lst, pad_reward_lst, pad_root_values_lst, pad_child_visits_lst)
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

        # put the game history into the pool
        self._trajectory_pool.append((last_game_segments[i], last_game_priorities[i], done[i]))

        # reset last game_segments
        last_game_segments[i] = None
        last_game_priorities[i] = None

    def get_priorities(self, i, pred_values_lst, search_values_lst):
        if self._cfg.use_priority and not self._cfg.use_max_priority_for_new_data:
            pred_values = torch.from_numpy(np.array(pred_values_lst[i])).to(self._cfg.device).float().view(-1)
            search_values = torch.from_numpy(np.array(search_values_lst[i])).to(self._cfg.device).float().view(-1)
            priorities = torch.abs(pred_values - search_values).cpu().numpy()
            priorities += self._cfg.prioritized_replay_eps
        else:
            # priorities is None -> use the max priority for all newly collected data
            priorities = None

        return priorities