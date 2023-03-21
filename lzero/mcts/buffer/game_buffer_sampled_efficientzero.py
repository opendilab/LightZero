import copy
import time
from typing import Any, List, Optional, Union

import numpy as np
import torch
from ding.data.buffer import Buffer
from ding.torch_utils.data_helper import to_ndarray
from ding.utils import BUFFER_REGISTRY
from easydict import EasyDict

from lzero.mcts.tree_search.mcts_ctree_sampled import SampledEfficientZeroMCTSCtree as MCTSCtree
from lzero.mcts.tree_search.mcts_ptree_sampled import SampledEfficientZeroMCTSPtree as MCTSPtree
from lzero.mcts.utils import BufferedData
from lzero.mcts.utils import prepare_observation_list
from lzero.policy import concat_output, concat_output_value, inverse_scalar_transform


@BUFFER_REGISTRY.register('game_buffer_sampled_efficientzero')
class SampledEfficientZeroGameBuffer(Buffer):
    """
    Overview:
        The specific game buffer for Sampled EfficientZero policy.
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    # the default_config for SampledEfficientZeroGameBuffer.
    config = dict(
        model=dict(
            action_space_size=6,
            continuous_action_space=False,
            num_of_sampled_actions=6,
            representation_network_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
        ),
        # learn_mode config
        learn=dict(
            # (int) How many samples in a training batch
            batch_size=256,
        ),
        # ==============================================================
        # begin of additional game_config
        # ==============================================================
        ## common
        mcts_ctree=True,
        device='cuda',
        env_type='not_board_games',
        # the size/capacity of replay_buffer, in the terms of transitions.
        replay_buffer_size=int(1e6),

        ## learn
        num_simulations=50,
        td_steps=5,
        num_unroll_steps=5,
        lstm_horizon_len=5,

        ## reanalyze
        reanalyze_ratio=0.3,
        reanalyze_outdated=True,
        # whether to use root value in reanalyzing part
        use_root_value=False,
        mini_infer_size=256,

        ## priority
        use_priority=True,
        use_max_priority_for_new_data=True,
        # how much prioritization is used: 0 means no prioritization while 1 means full prioritization
        priority_prob_alpha=0.6,
        # how much correction is used: 0 means no correction while 1 means full correction
        priority_prob_beta=0.4,
        prioritized_replay_eps=1e-6,

        ## UCB
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
        pb_c_base=19652,
        pb_c_init=1.25,
        discount_factor=0.997,
        value_delta_max=0.01,
        # ==============================================================
        # end of additional game_config
        # ==============================================================
    )

    def __init__(self, cfg: dict):
        super().__init__(cfg.replay_buffer_size)
        """
        Overview:
            Use the default configuration mechanism. If a user passes in a cfg with a key that matches an existing key 
            in the default configuration, the user-provided value will override the default configuration. Otherwise, 
            the default configuration will be used.
        """
        default_config = self.default_config()
        default_config.update(cfg)
        self._cfg = default_config
        assert self._cfg.env_type in ['not_board_games', 'board_games']

        self.batch_size = self._cfg.learn.batch_size
        self.replay_buffer_size = self._cfg.replay_buffer_size
        self._alpha = self._cfg.priority_prob_alpha

        self.keep_ratio = 1
        self.model_index = 0
        self.model_update_interval = 10

        self.game_block_buffer = []
        self.game_pos_priorities = []
        self.game_block_game_pos_look_up = []

        self._eps_collected = 0
        self.base_idx = 0
        self.clear_time = 0

    def push(self, data: Any, meta: Optional[dict] = None):
        """
        Overview:
            Push data and it's meta information in buffer.
            Save a game history block
        Arguments:
            - data (:obj:`Any`): The data which will be pushed into buffer.
                                 i.e. a game history block
            - meta (:obj:`dict`): Meta information, e.g. priority, count, staleness.
                - done: bool
                    True -> the game is finished. (always True)
                - unroll_plus_td_steps: int
                    if the game is not finished, we only save the transitions that can be computed
                - priorities: list
                    the priorities corresponding to the transitions in the game history
        Returns:
            - buffered_data (:obj:`BufferedData`): The pushed data.
        """
        if meta['done']:
            self._eps_collected += 1
        valid_len = len(data)

        if meta['priorities'] is None:
            max_prio = self.game_pos_priorities.max() if self.game_block_buffer else 1
            # if no 'priorities' provided, set the valid part of the new-added game history the max_prio
            self.game_pos_priorities = np.concatenate(
                (self.game_pos_priorities,
                 [max_prio for _ in range(valid_len)] + [0. for _ in range(valid_len, len(data))])
            )
        else:
            print('==' * 20)
            print('here' * 20)
            print('==' * 20)
            assert len(data) == len(meta['priorities']), " priorities should be of same length as the game steps"
            priorities = meta['priorities'].copy().reshape(-1)
            priorities[valid_len:len(data)] = 0.
            self.game_pos_priorities = np.concatenate((self.game_pos_priorities, priorities))

        self.game_block_buffer.append(data)
        self.game_block_game_pos_look_up += [(self.base_idx + len(self.game_block_buffer) - 1, step_pos) for step_pos in
                                             range(len(data))]

    def push_game_blocks(self, data_and_meta: Any):
        """
        Overview:
            Push game data and it's meta information in buffer.
            Save a game history block
        Keys:
            - data (:obj:`Any`): The data which will be pushed into buffer.
                                 i.e. a game history block
            - meta (:obj:`dict`): Meta information
        """
        data, meta = data_and_meta
        for (data_game, meta_game) in zip(data, meta):
            self.push(data_game, meta_game)

    def sample(
            self,
            size: Optional[int] = None,
            indices: Optional[List[str]] = None,
            replace: bool = False,
            sample_range: Optional[slice] = None,
            ignore_insufficient: bool = False,
            groupby: str = None,
            rolling_window: int = None
    ) -> Union[List[BufferedData], List[List[BufferedData]]]:
        """
        Overview:
            To be compatible with Buffer class
        """
        pass

    def get_transition(self, idx):
        """
        Overview:
            Sample one transition according to the idx
        Arguments:
            - idx: transition index
        Returns:
            - transition (:obj:`tuple`): One transition of pushed data.
        """
        game_block_idx, pos_in_game_block = self.game_block_game_pos_look_up[idx]
        game_block_idx -= self.base_idx
        transition = self.game_block_buffer[game_block_idx][pos_in_game_block]
        return transition

    def get_game_block_from_idx(self, idx: int) -> BufferedData:
        """
        Overview:
            Get one game according to the idx
        Arguments:
            - idx: game index
        Returns:
            - game: (:obj:`GameHistory`): One game history of pushed data.
        """
        return self.get_game(idx)

    def get_game(self, idx):
        """
        Overview:
            sample one game history according to the idx
        Arguments:
            - idx: transition index
            - return the game history including this transition
            - game_block_idx is the index of this game history in the self.game_block_buffer list
            - pos_in_game_block is the relative position of this transition in this game history
        Returns:
            - game: (:obj:`GameHistory`): One game history of pushed data.
        """

        game_block_idx, pos_in_game_block = self.game_block_game_pos_look_up[idx]
        game_block_idx -= self.base_idx
        game = self.game_block_buffer[game_block_idx]
        return game

    def update(self, index, data: Optional[Any] = None, meta: Optional[dict] = None) -> bool:
        """
        Overview:
            Update data and meta by index
        Arguments:
            - index (:obj:`str`): Index of one transition to be updated.
            - data (:obj:`any`): Pure data.  one transition.
            - meta (:obj:`dict`): Meta information.
        Returns:
            - success (:obj:`bool`): Success or not, if data with the index not exist in buffer, return false.
        """

        success = False
        if index < self.get_num_of_transitions():
            prio = meta['priorities']
            self.game_pos_priorities[index] = prio
            game_block_idx, pos_in_game_block = self.game_block_game_pos_look_up[index]
            game_block_idx -= self.base_idx
            # update one transition
            self.game_block_buffer[game_block_idx][pos_in_game_block] = data
            success = True

        return success

    def batch_update(self, indices: List[str], metas: Optional[List[Optional[dict]]] = None) -> None:
        """
        Overview:
            Batch update meta by indices, maybe useful in some data architectures.
        Arguments:
            - indices (:obj:`List[str]`): Index of data.
            - metas (:obj:`Optional[List[Optional[dict]]]`): Meta information.
        Returns:
            - success (:obj:`bool`): Success or not, if data with the index not exist in buffer, return false.
        """
        # only update the priorities for data still in replay buffer
        for i in range(len(indices)):
            if metas['make_time'][i] > self.clear_time:
                idx, prio = indices[i], metas['batch_priorities'][i]
                self.game_pos_priorities[idx] = prio

    def update_priority(self, train_data, batch_priorities) -> None:
        """
        Overview:
            Update the priority of training data.
        Arguments:
            - train_data (:obj:`Optional[List[Optional[np.ndarray]]]`): training data to be updated priority.
            - batch_priorities (:obj:`batch_priorities`): priorities to update to.
        """
        self.batch_update(indices=train_data[0][4],
                          metas={'make_time': train_data[0][6], 'batch_priorities': batch_priorities})

    def remove_oldest_data_to_fit(self):
        """
        Overview:
            remove some oldest data if the replay buffer is full.
        """
        nums_of_game_histoty = self.get_num_of_game_blocks()
        total_transition = self.get_num_of_transitions()
        if total_transition > self.replay_buffer_size:
            index = 0
            for i in range(nums_of_game_histoty):
                total_transition -= len(self.game_block_buffer[i])
                if total_transition <= self.replay_buffer_size * self.keep_ratio:
                    index = i
                    break

            if total_transition >= self._cfg.learn.batch_size:
                self._remove(index + 1)

    def _remove(self, num_excess_games):
        """
        Overview:
            delete game histories in index [0: num_excess_games]
        """
        excess_games_steps = sum([len(game) for game in self.game_block_buffer[:num_excess_games]])
        del self.game_block_buffer[:num_excess_games]
        self.game_pos_priorities = self.game_pos_priorities[excess_games_steps:]
        del self.game_block_game_pos_look_up[:excess_games_steps]
        self.base_idx += num_excess_games

        self.clear_time = time.time()

    def delete(self, index: str):
        """
        Overview:
            Delete one data sample by index
        Arguments:
            - index (:obj:`str`): Index
        """
        pass

    def clear(self) -> None:
        del self.game_block_buffer[:]

    def get_batch_size(self):
        return self.batch_size

    def get_priorities(self):
        return self.game_pos_priorities

    def get_num_of_episodes(self):
        # number of collected episodes
        return self._eps_collected

    def get_num_of_game_blocks(self) -> int:
        # number of games, i.e. num of game history blocks
        return len(self.game_block_buffer)

    def count(self):
        # number of games, i.e. num of game history blocks
        return len(self.game_block_buffer)

    def get_num_of_transitions(self):
        # total number of transitions
        return len(self.game_pos_priorities)

    def __copy__(self) -> "GameBuffer":
        buffer = type(self)(cfg=self._cfg)
        buffer.storage = self.game_block_buffer
        return buffer

    def prepare_batch_context(self, batch_size, beta):
        """
        Overview:
            Prepare a batch context that contains:
            game_lst: a list of game histories
            pos_in_game_block_list: transition index in game (relative index)
            batch_index_list: transition index in replay buffer
            weights: the weight concerning the priority
            make_time: the time the batch is made (for correctly updating replay buffer
                when data is deleted)
        Arguments:
            - batch_size: int batch size
            - beta: float the parameter in PER for calculating the priority
        Returns:
            - context (:obj:`Tuple`): Context information of a batch, including game_list, game_pos,
              batch_index_list, weights and make_time.
        """
        assert beta > 0

        # total number of transitions
        total = self.get_num_of_transitions()

        if self._cfg.use_priority is False:
            self.game_pos_priorities = np.ones_like(self.game_pos_priorities)

        # +1e-6 for numerical stability
        probs = self.game_pos_priorities ** self._alpha + 1e-6

        probs /= probs.sum()
        # TODO(pu): sample data in PER way
        # sample according to transition index
        # TODO(pu): replace=True
        # batch_index_list = np.random.choice(total, batch_size, p=probs, replace=True)
        batch_index_list = np.random.choice(total, batch_size, p=probs, replace=False)

        # TODO(pu): reanalyze the outdated data according to their generated time
        if self._cfg.reanalyze_outdated is True:
            batch_index_list.sort()

        weights = (total * probs[batch_index_list]) ** (-beta)
        weights /= weights.max()

        game_lst = []
        pos_in_game_block_list = []

        for idx in batch_index_list:
            try:
                game_block_idx, pos_in_game_block = self.game_block_game_pos_look_up[idx]
            except Exception as error:
                print(error)
            game_block_idx -= self.base_idx
            game = self.game_block_buffer[game_block_idx]

            game_lst.append(game)
            pos_in_game_block_list.append(pos_in_game_block)

        make_time = [time.time() for _ in range(len(batch_index_list))]

        context = (game_lst, pos_in_game_block_list, batch_index_list, weights, make_time)
        return context

    def make_batch(self, batch_context, reanalyze_ratio):
        """
        Overview:
            prepare the context of a batch
            reward_value_context:        the context of reanalyzed value targets
            policy_re_context:           the context of reanalyzed policy targets
            policy_non_re_context:       the context of non-reanalyzed policy targets
            current_batch:                the inputs of batch
        Arguments:
            batch_context: Any batch context from replay buffer
             reanalyze_ratio: float ratio of reanalyzed policy (value is 100% reanalyzed)
        Returns:
            - context (:obj:`Tuple`): reward_value_context, policy_re_context, policy_non_re_context, current_batch
        """
        # obtain the batch context from replay buffer
        game_lst, pos_in_game_block_list, batch_index_list, weights, make_time_lst = batch_context
        batch_size = len(batch_index_list)
        obs_lst, action_lst, mask_lst = [], [], []
        root_sampled_actions_lst = []
        # prepare the inputs of a batch
        for i in range(batch_size):
            game = game_lst[i]
            pos_in_game_block = pos_in_game_block_list[i]
            ######################
            # sampled related core code
            ######################
            actions_tmp = game.action_history[pos_in_game_block:pos_in_game_block + self._cfg.num_unroll_steps].tolist()

            # NOTE: self._cfg.num_unroll_steps + 1, TODO(pu): why +1
            root_sampled_actions_tmp = game.root_sampled_actions[
                                       pos_in_game_block:pos_in_game_block + self._cfg.num_unroll_steps + 1]
            # root_sampled_actions_tmp = game.root_sampled_actions[pos_in_game_block:pos_in_game_block + self._cfg.num_unroll_steps]

            # add mask for invalid actions (out of trajectory)
            # mask_tmp = [1. for i in range(len(_actions))]
            # mask_tmp += [0. for _ in range(self._cfg.num_unroll_steps - len(mask_tmp))]
            mask_tmp = [1. for i in range(len(root_sampled_actions_tmp))]
            mask_tmp += [0. for _ in range(self._cfg.num_unroll_steps + 1 - len(mask_tmp))]

            # pad random action
            if self._cfg.model.continuous_action_space:
                actions_tmp += [
                    np.random.randn(self._cfg.model.action_space_size)
                    for _ in range(self._cfg.num_unroll_steps - len(actions_tmp))
                ]
                root_sampled_actions_tmp += [
                    np.random.rand(self._cfg.model.num_of_sampled_actions, self._cfg.model.action_space_size)
                    for _ in range(self._cfg.num_unroll_steps + 1 - len(root_sampled_actions_tmp))
                ]
            else:
                actions_tmp += [
                    np.random.randint(0, self._cfg.model.action_space_size, 1).item()
                    for _ in range(self._cfg.num_unroll_steps - len(actions_tmp))
                ]
                if len(root_sampled_actions_tmp[0].shape) == 1:
                    root_sampled_actions_tmp += [
                        np.arange(self._cfg.model.action_space_size)
                        #  TODO(pu): why +1
                        for _ in range(self._cfg.num_unroll_steps + 1 - len(root_sampled_actions_tmp))
                    ]
                else:
                    root_sampled_actions_tmp += [
                        np.random.randint(0, self._cfg.model.action_space_size,
                                          self._cfg.model.num_of_sampled_actions).reshape(
                            self._cfg.model.num_of_sampled_actions, 1
                        )  # TODO(pu): why +1
                        for _ in range(self._cfg.num_unroll_steps + 1 - len(root_sampled_actions_tmp))
                    ]

            # obtain the input observations
            # stack+num_unroll_steps  4+5
            # pad if length of obs in game_block is less than stack+num_unroll_steps
            obs_lst.append(
                game_lst[i].obs(pos_in_game_block_list[i], num_unroll_steps=self._cfg.num_unroll_steps, padding=True))
            action_lst.append(actions_tmp)
            root_sampled_actions_lst.append(root_sampled_actions_tmp)

            mask_lst.append(mask_tmp)

        # formalize the input observations
        obs_lst = prepare_observation_list(obs_lst)
        ######################
        # sampled related core code
        ######################
        # formalize the inputs of a batch
        current_batch = [obs_lst, action_lst, root_sampled_actions_lst, mask_lst, batch_index_list, weights,
                         make_time_lst]

        try:
            root_sampled_actions_lst = np.concatenate([root_sampled_actions_lst])
        except Exception as error:
            print(error)

        for i in range(len(current_batch)):
            current_batch[i] = np.asarray(current_batch[i])

        total_transitions = self.get_num_of_transitions()

        # obtain the context of value targets
        reward_value_context = self.prepare_reward_value_context(
            batch_index_list, game_lst, pos_in_game_block_list, total_transitions
        )
        """
        only reanalyze recent reanalyze_ratio (e.g. 50%) data
        """
        reanalyze_num = int(batch_size * reanalyze_ratio)
        # if self._cfg.reanalyze_outdated is True:
        # batch_index_list is sorted according to its generated env_steps

        # 0:reanalyze_num -> reanalyzed policy, reanalyze_num:end -> non reanalyzed policy
        # reanalyzed policy
        if reanalyze_num > 0:
            # obtain the context of reanalyzed policy targets
            policy_re_context = self.prepare_policy_reanalyzed_context(
                batch_index_list[:reanalyze_num], game_lst[:reanalyze_num], pos_in_game_block_list[:reanalyze_num]
            )
        else:
            policy_re_context = None

        # non reanalyzed policy
        if reanalyze_num < batch_size:
            # obtain the context of non-reanalyzed policy targets
            policy_non_re_context = self.prepare_policy_non_reanalyzed_context(
                batch_index_list[reanalyze_num:], game_lst[reanalyze_num:], pos_in_game_block_list[reanalyze_num:]
            )
        else:
            policy_non_re_context = None

        context = reward_value_context, policy_re_context, policy_non_re_context, current_batch
        return context

    def prepare_reward_value_context(self, indices, games, state_index_lst, total_transitions):
        """
        Overview:
            prepare the context of rewards and values for calculating TD value target in reanalyzing part.
        Arguments:
            - indices (:obj:`list`): transition index in replay buffer
            - games (:obj:`list`): list of game histories
            - state_index_lst (:obj:`list`): list of transition index in game_block
            - total_transitions (:obj:`int`): number of collected transitions
        Returns:
            - reward_value_context (:obj:`list`): value_obs_lst, value_mask, state_index_lst, rewards_lst, traj_lens, 
              td_steps_lst, action_mask_history, to_play_history
        """
        zero_obs = games[0].zero_obs()
        config = self._cfg
        value_obs_lst = []
        # the value is valid or not (out of trajectory)
        value_mask = []
        rewards_lst = []
        traj_lens = []
        # for two_player board games
        action_mask_history, to_play_history = [], []

        td_steps_lst = []
        for game, state_index, idx in zip(games, state_index_lst, indices):
            traj_len = len(game)
            traj_lens.append(traj_len)

            # for atari
            # # off-policy correction: shorter horizon of td steps
            # delta_td = (total_transitions - idx) // config.auto_td_steps
            # td_steps = config.td_steps - delta_td
            # td_steps = np.clip(td_steps, 1, 5).astype(np.int)
            # TODO(pu):
            td_steps = np.clip(config.td_steps, 1, max(1, traj_len - state_index)).astype(np.int32)

            # NOTE: prepare the corresponding observations for bootstrapped values o_{t+k}
            # o[t+ td_steps, t + td_steps + stack frames + num_unroll_steps]
            # t=2+3 -> o[2+3, 2+3+4+5] -> o[5, 14]
            game_obs = game.obs(state_index + td_steps, config.num_unroll_steps)

            rewards_lst.append(game.reward_history)

            # for two_player board games
            action_mask_history.append(game.action_mask_history)
            to_play_history.append(game.to_play_history)

            for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
                # get the <num_unroll_steps+1>  bootstrapped target obs
                td_steps_lst.append(td_steps)
                # index of bootstrapped obs o_{t+td_steps}
                bootstrap_index = current_index + td_steps

                if bootstrap_index < traj_len:
                    value_mask.append(1)
                    # beg_index = bootstrap_index - (state_index + td_steps)
                    # max of beg_index is num_unroll_steps
                    beg_index = current_index - state_index
                    end_index = beg_index + config.model.frame_stack_num
                    # the stacked obs in time t
                    obs = game_obs[beg_index:end_index]
                else:
                    value_mask.append(0)
                    obs = zero_obs

                value_obs_lst.append(obs)

        reward_value_context = [
            value_obs_lst, value_mask, state_index_lst, rewards_lst, traj_lens, td_steps_lst, action_mask_history,
            to_play_history
        ]
        return reward_value_context

    def prepare_policy_non_reanalyzed_context(self, indices, games, state_index_lst):
        """
        Overview:
            prepare the context of policies for calculating policy target in non-reanalyzing part, just return the policy in self-play
        Arguments:
            - indices (:obj:`list`): transition index in replay buffer
            - games (:obj:`list`): list of game histories
            - state_index_lst (:obj:`list`): list transition index in game
        Returns:
            - policy_non_re_context (:obj:`list`): state_index_lst, child_visits, traj_lens, action_mask_history, to_play_history
        """
        child_visits = []
        traj_lens = []
        # for two_player board games
        action_mask_history, to_play_history = [], []

        for game, state_index, idx in zip(games, state_index_lst, indices):
            traj_len = len(game)
            traj_lens.append(traj_len)
            # for two_player board games
            action_mask_history.append(game.action_mask_history)
            to_play_history.append(game.to_play_history)

            child_visits.append(game.child_visit_history)

        policy_non_re_context = [state_index_lst, child_visits, traj_lens, action_mask_history, to_play_history]
        return policy_non_re_context

    def prepare_policy_reanalyzed_context(self, indices, games, state_index_lst):
        """
        Overview:
            prepare the context of policies for calculating policy target in reanalyzing part.
        Arguments:
            - indices (:obj:'list'):transition index in replay buffer
            - games (:obj:'list'):list of game histories
            - state_index_lst (:obj:'list'): transition index in game
        Returns:
            - policy_re_context (:obj:`list`): policy_obs_lst, policy_mask, state_index_lst, indices,
              child_visits, traj_lens, action_mask_history, to_play_history
        """
        zero_obs = games[0].zero_obs()
        config = self._cfg

        with torch.no_grad():
            # for policy
            policy_obs_lst = []
            policy_mask = []  # 0 -> out of traj, 1 -> new policy
            rewards, child_visits, traj_lens = [], [], []
            # for two_player board games
            action_mask_history, to_play_history = [], []
            for game, state_index in zip(games, state_index_lst):
                traj_len = len(game)
                traj_lens.append(traj_len)
                rewards.append(game.reward_history)
                # for two_player board games
                action_mask_history.append(game.action_mask_history)
                to_play_history.append(game.to_play_history)

                child_visits.append(game.child_visit_history)
                # prepare the corresponding observations
                game_obs = game.obs(state_index, config.num_unroll_steps)
                for current_index in range(state_index, state_index + config.num_unroll_steps + 1):

                    if current_index < traj_len:
                        policy_mask.append(1)
                        beg_index = current_index - state_index
                        end_index = beg_index + config.model.frame_stack_num
                        obs = game_obs[beg_index:end_index]
                    else:
                        policy_mask.append(0)
                        obs = zero_obs
                    policy_obs_lst.append(obs)

        policy_re_context = [
            policy_obs_lst, policy_mask, state_index_lst, indices, child_visits, traj_lens, action_mask_history,
            to_play_history
        ]
        return policy_re_context

    def compute_target_reward_value(self, reward_value_context, model):
        """
        Overview:
            prepare reward and value targets from the context of rewards and values.
        Arguments:
            - reward_value_context (:obj:'list'): the reward value context
            - model (:obj:'torch.tensor'):model of the target model
        Returns:
            - batch_value_prefixs (:obj:'np.ndarray): batch of value prefix
            - batch_values (:obj:'np.ndarray): batch of value estimation
        """
        value_obs_lst, value_mask, state_index_lst, rewards_lst, traj_lens, td_steps_lst, action_mask_history, \
        to_play_history = reward_value_context
        device = self._cfg.device
        batch_size = len(value_obs_lst)
        game_block_batch_size = len(state_index_lst)

        if to_play_history[0][0] in [1, 2]:
            # for two_player board games
            # to_play
            to_play = []
            for bs in range(game_block_batch_size):
                to_play_tmp = list(
                    to_play_history[bs][state_index_lst[bs]:state_index_lst[bs] + self._cfg.num_unroll_steps + 1]
                )
                if len(to_play_tmp) < self._cfg.num_unroll_steps + 1:
                    # effective play index is {1,2}
                    to_play_tmp += [1 for _ in range(self._cfg.num_unroll_steps + 1 - len(to_play_tmp))]
                to_play.append(to_play_tmp)
            # to_play = to_ndarray(to_play)
            tmp = []
            for i in to_play:
                tmp += list(i)
            to_play = tmp
            # action_mask
            action_mask = []
            for bs in range(game_block_batch_size):
                action_mask_tmp = list(
                    action_mask_history[bs][state_index_lst[bs]:state_index_lst[bs] + self._cfg.num_unroll_steps + 1]
                )
                if len(action_mask_tmp) < self._cfg.num_unroll_steps + 1:
                    action_mask_tmp += [
                        list(np.ones(self._cfg.model.action_space_size, dtype=np.int8))
                        for _ in range(self._cfg.num_unroll_steps + 1 - len(action_mask_tmp))
                    ]
                action_mask.append(action_mask_tmp)
            action_mask = to_ndarray(action_mask)
            tmp = []
            for i in action_mask:
                tmp += i
            action_mask = tmp

        batch_values, batch_value_prefixs = [], []
        with torch.no_grad():
            value_obs_lst = prepare_observation_list(value_obs_lst)
            # split a full batch into slices of mini_infer_size: to save the GPU memory for more GPU actors
            m_batch = self._cfg.mini_infer_size
            slices = np.ceil(batch_size / m_batch).astype(np.int_)
            network_output = []
            for i in range(slices):
                beg_index = m_batch * i
                end_index = m_batch * (i + 1)
                m_obs = torch.from_numpy(value_obs_lst[beg_index:end_index]).to(device).float()

                # calculate the target value
                m_output = model.initial_inference(m_obs)

                # TODO(pu)
                if not model.training:
                    # if not in training, obtain the scalars of the value/reward
                    m_output.hidden_state = m_output.hidden_state.detach().cpu().numpy()
                    m_output.value = inverse_scalar_transform(m_output.value,
                                                              self._cfg.model.support_scale).detach().cpu().numpy()
                    m_output.policy_logits = m_output.policy_logits.detach().cpu().numpy()
                    m_output.reward_hidden_state = (
                        m_output.reward_hidden_state[0].detach().cpu().numpy(),
                        m_output.reward_hidden_state[1].detach().cpu().numpy()
                    )

                network_output.append(m_output)

            # concat the output slices after model inference
            if self._cfg.use_root_value:
                # use the root values from MCTS
                # the root values have limited improvement but require much more GPU actors;
                _, value_prefix_pool, policy_logits_pool, hidden_state_roots, reward_hidden_state_roots = concat_output(
                    network_output
                )
                value_prefix_pool = value_prefix_pool.squeeze().tolist()
                policy_logits_pool = policy_logits_pool.tolist()

                if self._cfg.mcts_ctree:
                    """
                    cpp mcts_tree
                    """
                    if to_play_history[0][0] in [None, -1]:
                        # we use to_play=-1 means play_with_bot_mode game
                        to_play = [-1 for i in range(batch_size)]
                        # if action_mask_history[0][0] is None:
                        # continuous action space env: all -1
                        legal_actions = [
                            [-1 for i in range(self._cfg.model.action_space_size)] for _ in range(batch_size)
                        ]
                    else:
                        legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(batch_size)]

                    roots = MCTSCtree.Roots(
                        batch_size, legal_actions, self._cfg.model.action_space_size,
                        self._cfg.model.num_of_sampled_actions, self._cfg.model.continuous_action_space
                    )
                    noises = [
                        np.random.dirichlet([self._cfg.root_dirichlet_alpha] * self._cfg.model.num_of_sampled_actions
                                            ).astype(np.float32).tolist() for _ in range(batch_size)
                    ]
                    roots.prepare(
                        self._cfg.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool, to_play
                    )
                    # do MCTS for a new policy with the recent target model
                    MCTSCtree(self._cfg).search(roots, model, hidden_state_roots, reward_hidden_state_roots, to_play)
                else:
                    """
                    python mcts_tree
                    """
                    if to_play_history[0][0] in [None, -1]:
                        # for one_player atari games
                        action_mask = [
                            list(np.ones(self._cfg.model.action_space_size, dtype=np.int8)) for _ in range(batch_size)
                        ]
                    legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(batch_size)]
                    roots = MCTSPtree.Roots(
                        batch_size, legal_actions, num_of_sampled_actions=self._cfg.model.num_of_sampled_actions
                    )
                    noises = [
                        np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(sum(action_mask[j]))
                                            ).astype(np.float32).tolist() for j in range(batch_size)
                    ]

                    if to_play_history[0][0] in [None, -1]:
                        roots.prepare(
                            self._cfg.root_exploration_fraction,
                            noises,
                            value_prefix_pool,
                            policy_logits_pool,
                            to_play=-1
                        )
                        # do MCTS for a new policy with the recent target model
                        MCTS_ptree(self._cfg).search(
                            roots, model, hidden_state_roots, reward_hidden_state_roots, to_play=-1
                        )
                    else:
                        roots.prepare(
                            self._cfg.root_exploration_fraction,
                            noises,
                            value_prefix_pool,
                            policy_logits_pool,
                            to_play=to_play
                        )
                        # do MCTS for a new policy with the recent target model
                        MCTS_ptree(self._cfg).search(
                            roots, model, hidden_state_roots, reward_hidden_state_roots, to_play=to_play
                        )

                roots_values = roots.get_values()
                value_lst = np.array(roots_values)
            else:
                # use the predicted values
                value_lst = concat_output_value(network_output)

            # get last state value
            if to_play_history[0][0] in [1, 2]:
                # TODO(pu): board_games
                value_lst = value_lst.reshape(-1) * np.array(
                    [
                        self._cfg.discount_factor ** td_steps_lst[i] if int(td_steps_lst[i]) %
                                                                        2 == 0 else -self._cfg.discount_factor **
                                                                                     td_steps_lst[i] for
                        i in range(batch_size)
                    ]
                )

            else:
                value_lst = value_lst.reshape(-1) * (
                        np.array([self._cfg.discount_factor for _ in range(batch_size)]) ** td_steps_lst
                )
            value_lst = value_lst * np.array(value_mask)
            value_lst = value_lst.tolist()

            horizon_id, value_index = 0, 0
            for traj_len_non_re, reward_lst, state_index, to_play_list in zip(traj_lens, rewards_lst, state_index_lst,
                                                                              to_play_history):
                # traj_len = len(game)
                target_values = []
                target_value_prefixs = []

                value_prefix = 0.0
                base_index = state_index
                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                    bootstrap_index = current_index + td_steps_lst[value_index]
                    # for i, reward in enumerate(game.rewards[current_index:bootstrap_index]):
                    for i, reward in enumerate(reward_lst[current_index:bootstrap_index]):
                        if to_play_history[0][0] in [1, 2]:
                            # TODO(pu): board_games
                            if to_play_list[current_index] == to_play_list[i]:
                                value_lst[value_index] += reward * self._cfg.discount_factor ** i
                            else:
                                value_lst[value_index] += -reward * self._cfg.discount_factor ** i
                        else:
                            value_lst[value_index] += reward * self._cfg.discount_factor ** i
                            """
                            TODO(pu): why value don't use discount_factor factor
                            """

                    # reset every lstm_horizon_len
                    if horizon_id % self._cfg.lstm_horizon_len == 0:
                        value_prefix = 0.0
                        base_index = current_index
                    horizon_id += 1

                    if current_index < traj_len_non_re:
                        target_values.append(value_lst[value_index])
                        # Since the horizon is small and the discount_factor is close to 1.
                        # Compute the reward sum to approximate the value prefix for simplification
                        value_prefix += reward_lst[
                            current_index]  # * config.discount_factor ** (current_index - base_index)
                        target_value_prefixs.append(value_prefix)
                    else:
                        target_values.append(0)
                        target_value_prefixs.append(value_prefix)
                    # NOTE:
                    value_index += 1

                batch_value_prefixs.append(target_value_prefixs)
                batch_values.append(target_values)

        batch_value_prefixs = np.asarray(batch_value_prefixs)
        batch_values = np.asarray(batch_values)
        return batch_value_prefixs, batch_values

    def compute_target_policy_reanalyzed(self, policy_re_context, model):
        """
        Overview:
            prepare policy targets from the reanalyzed context of policies
        Arguments:
            - policy_re_context (:obj:`List`): List of policy context to reanalyzed
        Returns:
            - batch_target_policies_re
        """
        batch_target_policies_re = []
        if policy_re_context is None:
            return batch_target_policies_re

        # for two_player board games
        policy_obs_lst, policy_mask, state_index_lst, indices, child_visits, traj_lens, action_mask_history, \
        to_play_history = policy_re_context
        batch_size = len(policy_obs_lst)
        game_block_batch_size = len(state_index_lst)

        device = self._cfg.device

        if self._cfg.env_type == 'board_games':
            # for two_player board games
            # to_play
            to_play = []
            for bs in range(game_block_batch_size):
                to_play_tmp = list(
                    to_play_history[bs][state_index_lst[bs]:state_index_lst[bs] + self._cfg.num_unroll_steps + 1]
                )
                if len(to_play_tmp) < self._cfg.num_unroll_steps + 1:
                    # effective play index is {1,2}
                    to_play_tmp += [1 for _ in range(self._cfg.num_unroll_steps + 1 - len(to_play_tmp))]
                to_play.append(to_play_tmp)
            # to_play = to_ndarray(to_play)
            tmp = []
            for i in to_play:
                tmp += list(i)
            to_play = tmp
            # action_mask
            action_mask = []
            for bs in range(game_block_batch_size):
                action_mask_tmp = list(
                    action_mask_history[bs][state_index_lst[bs]:state_index_lst[bs] + self._cfg.num_unroll_steps + 1]
                )
                if len(action_mask_tmp) < self._cfg.num_unroll_steps + 1:
                    action_mask_tmp += [
                        list(np.ones(self._cfg.model.action_space_size, dtype=np.int8))
                        for _ in range(self._cfg.num_unroll_steps + 1 - len(action_mask_tmp))
                    ]
                action_mask.append(action_mask_tmp)
            # action_mask = to_ndarray(action_mask)
            tmp = []
            for i in action_mask:
                tmp += i
            action_mask = tmp

            # the minimal size is <self._cfg. num_unroll_steps+1>
            legal_actions = [
                [i for i, x in enumerate(action_mask[j]) if x == 1]
                for j in range(max(self._cfg.num_unroll_steps + 1, batch_size))
            ]

        with torch.no_grad():
            policy_obs_lst = prepare_observation_list(policy_obs_lst)
            # split a full batch into slices of mini_infer_size: to save the GPU memory for more GPU actors
            m_batch = self._cfg.mini_infer_size
            slices = np.ceil(batch_size / m_batch).astype(np.int_)
            network_output = []
            for i in range(slices):
                beg_index = m_batch * i
                end_index = m_batch * (i + 1)
                m_obs = torch.from_numpy(policy_obs_lst[beg_index:end_index]).to(device).float()

                # NOTE
                m_output = model.initial_inference(m_obs)
                # TODO(pu)
                if not model.training:
                    # if not in training, obtain the scalars of the value/reward
                    m_output.hidden_state = m_output.hidden_state.detach().cpu().numpy()
                    m_output.value = inverse_scalar_transform(m_output.value,
                                                              self._cfg.model.support_scale).detach().cpu().numpy()
                    m_output.policy_logits = m_output.policy_logits.detach().cpu().numpy()
                    m_output.reward_hidden_state = (
                        m_output.reward_hidden_state[0].detach().cpu().numpy(),
                        m_output.reward_hidden_state[1].detach().cpu().numpy()
                    )
                network_output.append(m_output)

            _, value_prefix_pool, policy_logits_pool, hidden_state_roots, reward_hidden_state_roots = concat_output(
                network_output
            )
            value_prefix_pool = value_prefix_pool.squeeze().tolist()
            policy_logits_pool = policy_logits_pool.tolist()
            if self._cfg.mcts_ctree:
                ######################
                # sampled related core code
                ######################
                """
                cpp mcts_tree
                """
                if to_play_history[0][0] in [None, -1]:
                    # we use to_play=-1 means play_with_bot_mode game
                    to_play = [-1 for i in range(batch_size)]
                    # if action_mask_history[0][0] is None:
                    # continuous action space env: all -1
                    legal_actions = [[-1 for i in range(self._cfg.model.action_space_size)] for _ in range(batch_size)]
                else:
                    legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(batch_size)]

                roots = MCTSCtree.Roots(
                    batch_size, legal_actions, self._cfg.model.action_space_size,
                    self._cfg.model.num_of_sampled_actions, self._cfg.model.continuous_action_space
                )
                noises = [
                    np.random.dirichlet([self._cfg.root_dirichlet_alpha] * self._cfg.model.num_of_sampled_actions
                                        ).astype(np.float32).tolist() for _ in range(batch_size)
                ]
                roots.prepare(
                    self._cfg.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool, to_play
                )
                # do MCTS for a new policy with the recent target model
                MCTSCtree(self._cfg).search(roots, model, hidden_state_roots, reward_hidden_state_roots, to_play)
            else:
                """
                python mcts_tree
                """
                if to_play_history[0][0] in [None, -1]:
                    # we use to_play=-1 means play_with_bot_mode game in mcts_ptree
                    to_play = [None for i in range(batch_size)]
                    # continuous action space env: all -1
                    legal_actions = [
                        [-1 for i in range(self._cfg.model.num_of_sampled_actions)] for _ in range(batch_size)
                    ]
                    # action_mask = [
                    #     list(np.ones(self._cfg.model.num_of_sampled_actions, dtype=np.int8)) for _ in range(batch_size)
                    # ]

                else:
                    # if board_games, we have action_mask
                    legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(batch_size)]

                roots = MCTSPtree.Roots(
                    batch_size,
                    legal_actions,
                    action_space_size=self._cfg.model.action_space_size,
                    num_of_sampled_actions=self._cfg.model.num_of_sampled_actions,
                    continuous_action_space=self._cfg.model.continuous_action_space
                )
                # noises = [
                #     np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(sum(action_mask[j]))
                #                         ).astype(np.float32).tolist() for j in range(batch_size)
                # ]
                noises = [
                    np.random.dirichlet([self._cfg.root_dirichlet_alpha] * self._cfg.model.num_of_sampled_actions
                                        ).astype(np.float32).tolist() for _ in range(batch_size)
                ]
                roots.prepare(
                    self._cfg.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool, to_play
                )
                # do MCTS for a new policy with the recent target model
                MCTS_ptree(self._cfg).search(roots, model, hidden_state_roots, reward_hidden_state_roots, to_play)

                roots_legal_actions_list = roots.legal_actions_list

            roots_distributions = roots.get_distributions()

            # ==============================================================
            # fix reanalyze in sez
            # ==============================================================
            roots_sampled_actions = roots.get_sampled_actions()
            try:
                root_sampled_actions = np.array([action.value for action in roots_sampled_actions])
            except Exception as error:
                root_sampled_actions = np.array([action for action in roots_sampled_actions])

            policy_index = 0
            for state_index, game_idx in zip(state_index_lst, indices):
                target_policies = []

                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                    distributions = roots_distributions[policy_index]
                    ######################
                    # sampled related core code
                    ######################
                    if policy_mask[policy_index] == 0:
                        # the null target policy
                        # for sampled the invalid target policy
                        target_policies.append([0 for _ in range(self._cfg.model.num_of_sampled_actions)])
                    else:
                        if distributions is None:
                            # if at some obs, the legal_action is None, add the fake target_policy
                            target_policies.append(
                                list(
                                    np.ones(self._cfg.model.num_of_sampled_actions) /
                                    self._cfg.model.num_of_sampled_actions
                                )
                            )
                        else:
                            if self._cfg.mcts_ctree:
                                ## cpp mcts_tree
                                if to_play_history[0][0] in [None, -1]:
                                    # for one_player atari games
                                    # TODO(pu): very important
                                    sum_visits = sum(distributions)
                                    policy = [visit_count / sum_visits for visit_count in distributions]
                                    target_policies.append(policy)
                                else:
                                    # for two_player board games
                                    policy_tmp = [0 for _ in range(self._cfg.model.action_space_size)]
                                    # to make sure target_policies have the same dimension
                                    # target_policy = torch.from_numpy(target_policy) be correct
                                    sum_visits = sum(distributions)
                                    policy = [visit_count / sum_visits for visit_count in distributions]
                                    for index, legal_action in enumerate(roots_legal_actions_list[policy_index]):
                                        policy_tmp[legal_action] = policy[index]
                                    target_policies.append(policy_tmp)
                            else:
                                """
                                python mcts_tree
                                """
                                if to_play_history[0][0] in [None, -1]:
                                    # TODO(pu): very important
                                    sum_visits = sum(distributions)
                                    policy = [visit_count / sum_visits for visit_count in distributions]
                                    target_policies.append(policy)
                                    # target_policies.append(distributions)
                                else:
                                    # for two_player board games
                                    policy_tmp = [0 for _ in range(self._cfg.model.action_space_size)]
                                    # to make sure target_policies have the same dimension
                                    # target_policy = torch.from_numpy(target_policy) be correct
                                    sum_visits = sum(distributions)
                                    policy = [visit_count / sum_visits for visit_count in distributions]
                                    for index, legal_action in enumerate(roots_legal_actions_list[policy_index]):
                                        policy_tmp[legal_action] = policy[index]
                                    target_policies.append(policy_tmp)

                    policy_index += 1

                batch_target_policies_re.append(target_policies)

        batch_target_policies_re = np.array(batch_target_policies_re)

        return batch_target_policies_re, root_sampled_actions

    def compute_target_policy_non_reanalyzed(self, policy_non_re_context):
        """
        Overview:
            prepare policy targets from the non-reanalyzed context of policies
        Arguments:
            - policy_non_re_context (:obj:`List`): List containing:
                - state_index_lst
                - child_visits
                - traj_lens
                - action_mask_history
                - to_play_history
        Returns:
            - batch_target_policies_non_re
        """
        batch_target_policies_non_re = []
        if policy_non_re_context is None:
            return batch_target_policies_non_re

        state_index_lst, child_visits, traj_lens, action_mask_history, to_play_history = policy_non_re_context

        game_block_batch_size = len(state_index_lst)

        if self._cfg.env_type == 'board_games':
            # for two_player board games
            # action_mask
            action_mask = []
            for bs in range(game_block_batch_size):
                action_mask_tmp = list(
                    action_mask_history[bs][state_index_lst[bs]:state_index_lst[bs] + self._cfg.num_unroll_steps + 1]
                )
                if len(action_mask_tmp) < self._cfg.num_unroll_steps + 1:
                    action_mask_tmp += [
                        list(np.ones(self._cfg.model.action_space_size, dtype=np.int8))
                        for _ in range(self._cfg.num_unroll_steps + 1 - len(action_mask_tmp))
                    ]
                action_mask.append(action_mask_tmp)
            action_mask = to_ndarray(action_mask)
            tmp = []
            for i in action_mask:
                tmp += i
            action_mask = tmp

            # the minimal size is <self._cfg. num_unroll_steps+1>
            legal_actions = [
                [i for i, x in enumerate(action_mask[j]) if x == 1]
                for j in range(game_block_batch_size * (self._cfg.num_unroll_steps + 1))
            ]

        with torch.no_grad():
            policy_index = 0
            # for policy
            policy_mask = []  # 0 -> out of traj, 1 -> old policy
            # for game, state_index in zip(games, state_index_lst):
            for traj_len, child_visit, state_index in zip(traj_lens, child_visits, state_index_lst):
                # traj_len = len(game)
                target_policies = []

                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                    if current_index < traj_len:
                        # target_policies.append(child_visit[current_index])
                        policy_mask.append(1)
                        # child_visit is already a distribution
                        distributions = child_visit[current_index]
                        if self._cfg.mcts_ctree:
                            """
                            cpp mcts_tree
                            """
                            if self._cfg.env_type == 'not_board_games':
                                # for one_player atari games
                                target_policies.append(distributions)
                            else:
                                # for two_player board games
                                policy_tmp = [0 for _ in range(self._cfg.model.action_space_size)]
                                # to make sure target_policies have the same dimension <self._cfg.model.action_space_size>
                                # sum_visits = sum(distributions)
                                # distributions = [visit_count / sum_visits for visit_count in distributions]
                                for index, legal_action in enumerate(legal_actions[policy_index]):
                                    # try:
                                    # only the action in ``legal_action`` the policy logits is nonzero
                                    policy_tmp[legal_action] = distributions[index]
                                    # except Exception as error:
                                    #     print(error)
                                target_policies.append(policy_tmp)
                        else:
                            """
                            python mcts_tree
                            """
                            if self._cfg.env_type == 'not_board_games':
                                # for one_player atari games
                                target_policies.append(distributions)
                            else:
                                # for two_player board games
                                policy_tmp = [0 for _ in range(self._cfg.model.action_space_size)]
                                # to make sure target_policies have the same dimension <self._cfg.model.action_space_size>
                                # sum_visits = sum(distributions)
                                # distributions = [visit_count / sum_visits for visit_count in distributions]
                                for index, legal_action in enumerate(legal_actions[policy_index]):
                                    # try:
                                    # only the action in ``legal_action`` the policy logits is nonzero
                                    policy_tmp[legal_action] = distributions[index]
                                    # except Exception as error:
                                    #     print(error)
                                target_policies.append(policy_tmp)

                    else:
                        # ==============================================================
                        # sampled related core code
                        # ==============================================================
                        # the invalid target policy
                        target_policies.append([0 for _ in range(self._cfg.model.num_of_sampled_actions)])
                        policy_mask.append(0)

                    policy_index += 1

                batch_target_policies_non_re.append(target_policies)
        batch_target_policies_non_re = np.asarray(batch_target_policies_non_re)
        return batch_target_policies_non_re

    def sample_train_data(self, batch_size, policy):
        """
        Overview:
            sample data from ``GameBuffer`` and prepare the current and target batch for training
        Arguments:
            - batch_size (:obj:`int`): batch size
            - policy (:obj:`torch.tensor`): model of policy
        Returns:
            - train_data (:obj:`List`): List of train data
        """

        policy._target_model.to(self._cfg.device)
        policy._target_model.eval()

        batch_context = self.prepare_batch_context(batch_size, self._cfg.priority_prob_beta)
        input_context = self.make_batch(batch_context, self._cfg.reanalyze_ratio)
        reward_value_context, policy_re_context, policy_non_re_context, current_batch = input_context

        # target reward, value
        batch_value_prefixs, batch_values = self.compute_target_reward_value(reward_value_context, policy._target_model)

        batch_target_policies_non_re = self.compute_target_policy_non_reanalyzed(policy_non_re_context)

        if self._cfg.reanalyze_ratio > 0:
            # target policy
            batch_target_policies_re, root_sampled_actions = self.compute_target_policy_reanalyzed(policy_re_context,
                                                                                                   policy._target_model)
            # ==============================================================
            # fix reanalyze in sez:
            # use the latest root_sampled_actions after the reanalyze process,
            # because the batch_target_policies_re is corresponding to the latest root_sampled_actions
            # ==============================================================
            assert (self._cfg.reanalyze_ratio > 0 and self._cfg.reanalyze_outdated is True), \
                "in sampled effiicientzero, if self._cfg.reanalyze_ratio>0, you must set self._cfg.reanalyze_outdated=True"
            # current_batch = [obs_lst, action_lst, root_sampled_actions_lst, mask_lst, batch_index_list, weights, make_time_lst]
            if self._cfg.model.continuous_action_space:
                current_batch[2][:int(batch_size * self._cfg.reanalyze_ratio)] = root_sampled_actions.reshape(
                    int(batch_size * self._cfg.reanalyze_ratio), self._cfg.num_unroll_steps + 1,
                    self._cfg.model.num_of_sampled_actions, self._cfg.model.action_space_size)
            else:
                current_batch[2][:int(batch_size * self._cfg.reanalyze_ratio)] = root_sampled_actions.reshape(
                    int(batch_size * self._cfg.reanalyze_ratio), self._cfg.num_unroll_steps + 1,
                    self._cfg.model.num_of_sampled_actions, 1)

        if 0 < self._cfg.reanalyze_ratio < 1:
            try:
                batch_policies = np.concatenate([batch_target_policies_re, batch_target_policies_non_re])
            except Exception as error:
                print(error)
        elif self._cfg.reanalyze_ratio == 1:
            batch_policies = batch_target_policies_re
        elif self._cfg.reanalyze_ratio == 0:
            batch_policies = batch_target_policies_non_re
        targets_batch = [batch_value_prefixs, batch_values, batch_policies]
        # a batch contains the inputs and the targets
        train_data = [current_batch, targets_batch]
        return train_data

    # the following is to be compatible with Buffer class.
    def save_data(self):
        pass

    def load_data(self):
        pass

    def get(self):
        pass
