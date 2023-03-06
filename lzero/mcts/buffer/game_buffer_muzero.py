"""
Acknowledgement: The following code is adapted from https://github.com/YeWR/EfficientZero/core/replay_buffer.py
"""
import copy
import itertools
import logging
import random
import time
from typing import Any, List, Optional, Union
from lzero.mcts.utils import BufferedData

import numpy as np
import torch
from ding.data.buffer import Buffer
from ding.torch_utils.data_helper import to_ndarray
from ding.utils import BUFFER_REGISTRY
from easydict import EasyDict

# python mcts_tree
import lzero.mcts.ptree.ptree_mz as ptree
from lzero.mcts.tree_search.mcts_ptree import MuZeroMCTSPtree as MCTS_ptree
# cpp mcts_tree
from lzero.mcts.ctree.ctree_muzero import mz_tree as ctree
from lzero.mcts.tree_search.mcts_ctree import MuZeroMCTSCtree as MCTS_ctree
from lzero.mcts.utils import prepare_observation_list, concat_output, concat_output_value
from lzero.mcts.scaling_transform import inverse_scalar_transform




@BUFFER_REGISTRY.register('game_buffer_muzero')
class MuZeroGameBuffer(Buffer):
    """
    Overview:
        The specific game buffer for MuZero policy.
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    # the default_config for MuZeroGameBuffer.
    config = dict(
        model=dict(
            # the stacked obs shape -> the transformed obs shape:
            # [S, W, H, C] -> [S x C, W, H]
            # e.g. [4, 96, 96, 3] -> [4*3, 96, 96]
            # observation_shape=(12, 96, 96),  # if frame_stack_num=4, gray_scale=False
            # observation_shape=(3, 96, 96),  # if frame_stack_num=1, gray_scale=False
            observation_shape=(4, 96, 96),  # if frame_stack_num=4, gray_scale=True
            action_space_size=6,
            representation_model_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
            # whether to use the self_supervised_learning_loss.
            self_supervised_learning_loss=True,
            # whether to use discrete support to represent categorical distribution for value, reward/value_prefix.
            categorical_distribution=True,
            activation=torch.nn.ReLU(inplace=True),
            batch_norm_momentum=0.1,
            last_linear_layer_init_zero=True,
            state_norm=False,
            # the key difference setting between image-input and vector input.
            image_channel=1,
            frame_stack_num=4,
            downsample=True,
            # ==============================================================
            # the default config is large size model, same as the EfficientZero original paper.
            # ==============================================================
            num_res_blocks=1,
            num_channels=64,
            lstm_hidden_size=512,
            # the following model para. is usually fixed
            reward_head_channels=16,
            value_head_channels=16,
            policy_head_channels=16,
            fc_reward_layers=[32],
            fc_value_layers=[32],
            fc_policy_layers=[32],
            support_scale=300,
            reward_support_size=601,
            value_support_size=601,
            proj_hid=1024,
            proj_out=1024,
            pred_hid=512,
            pred_out=1024,
            # the above model para. is usually fixed
        ),
        # learn_mode config
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            # For different env, we have different episode_length,
            # we usually set update_per_collect = collector_env_num * episode_length / batch_size * reuse_factor
            update_per_collect=100,
            # (int) How many samples in a training batch
            batch_size=256,
            lr_manually=True,
            optim_type='SGD',
            learning_rate=0.2,  # init lr for manually decay schedule
            # optim_type='Adam',
            # learning_rate=0.001,  # lr for Adam optimizer
            # (int) Frequency of target network update.
            target_update_freq=100,
            # (bool) Whether ignore done(usually for max step termination env)
            ignore_done=False,
            weight_decay=1e-4,
            momentum=0.9,
            grad_clip_type='clip_norm',
            grad_clip_value=10,
        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_episode" episodes per collect.
            n_episode=8,
            unroll_len=1,
        ),
        # ==============================================================
        # begin of additional game_config
        # ==============================================================
        ## common
        mcts_ctree=True,
        device='cuda',
        collector_env_num=8,
        evaluator_env_num=3,
        env_type='not_board_games',
        battle_mode='play_with_bot_mode',
        game_wrapper=True,
        monitor_statistics=True,
        game_block_length=200,
        # the size/capacity of replay_buffer, in the terms of transitions.
        replay_buffer_size=int(1e6),

        ## observation
        # the key difference setting between image-input and vector input.
        cvt_string=False,
        gray_scale=False,
        use_augmentation=True,
        # style of augmentation
        augmentation=['shift', 'intensity'],

        ## learn
        num_simulations=50,
        td_steps=5,
        num_unroll_steps=5,
        max_grad_norm=10,
        # the weight of different loss
        reward_loss_weight=1,
        value_loss_weight=0.25,
        policy_loss_weight=1,
        ssl_loss_weight=0,

        # ``threshold_training_steps_for_final_lr`` is only used for adjusting lr manually.
        # threshold_training_steps_for_final_lr=int(
        #     threshold_env_steps_for_final_lr / collector_env_num / average_episode_length_when_converge * update_per_collect),
        threshold_training_steps_for_final_lr=int(1e5),
        # lr: 0.2 -> 0.02 -> 0.002

        # ``threshold_training_steps_for_final_temperature`` is only used for adjusting temperature manually.
        # threshold_training_steps_for_final_temperature=int(
        #     threshold_env_steps_for_final_temperature / collector_env_num / average_episode_length_when_converge * update_per_collect),
        threshold_training_steps_for_final_temperature=int(1e5),
        # temperature: 1 -> 0.5 -> 0.25
        auto_temperature=True,
        # ``fixed_temperature_value`` is effective only when auto_temperature=False
        fixed_temperature_value=0.25,

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
        discount=0.997,
        value_delta_max=0.01,
        # ==============================================================
        # end of additional game_config
        # ==============================================================
    )

    def __init__(self, cfg: dict):
        super().__init__(cfg.other.replay_buffer.replay_buffer_size)
        # NOTE: utilize the default config
        default_config = self.default_config()
        default_config.update(cfg)
        self._cfg = default_config
        assert self._cfg.env_type in ['not_board_games', 'board_games']

        self.replay_buffer_size = self._cfg.replay_buffer_size
        self.batch_size = self._cfg.learn.batch_size
        self.keep_ratio = 1

        self.model_index = 0
        self.model_update_interval = 10

        self.game_block_buffer = []
        self.game_pos_priorities = []
        self.game_block_game_pos_look_up = []

        self._eps_collected = 0
        self.base_idx = 0
        self._alpha = self._cfg.priority_prob_alpha
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
                - end_tag: bool
                    True -> the game is finished. (always True)
                - unroll_plus_td_stepss: int
                    if the game is not finished, we only save the transitions that can be computed
                - priorities: list
                    the priorities corresponding to the transitions in the game history
        Returns:
            - buffered_data (:obj:`BufferedData`): The pushed data.
        """

        if meta['end_tag']:
            self._eps_collected += 1
            valid_len = len(data)
        else:
            valid_len = len(data) - meta['unroll_plus_td_stepss']

        if meta['priorities'] is None:
            max_prio = self.game_pos_priorities.max() if self.game_block_buffer else 1
            # if no 'priorities' provided, set the valid part of the new-added game history the max_prio
            self.game_pos_priorities = np.concatenate(
                (self.game_pos_priorities, [max_prio for _ in range(valid_len)] + [0. for _ in range(valid_len, len(data))])
            )
        else:
            assert len(data) == len(meta['priorities']), " priorities should be of same length as the game steps"
            priorities = meta['priorities'].copy().reshape(-1)
            priorities[valid_len:len(data)] = 0.
            self.game_pos_priorities = np.concatenate((self.game_pos_priorities, priorities))

        self.game_block_buffer.append(data)
        self.game_block_game_pos_look_up += [(self.base_idx + len(self.game_block_buffer) - 1, step_pos) for step_pos in range(len(data))]

    def push_games(self, data: Any, meta):
        """
        Overview:
            save a list of game histories
        """
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
            Sample data with length ``size``.
        Arguments:
            - size (:obj:`Optional[int]`): The number of the data that will be sampled.
            - indices (:obj:`Optional[List[str]]`): Sample with multiple indices.
            - replace (:obj:`bool`): If use replace is true, you may receive duplicated data from the buffer.
            - sample_range (:obj:`slice`): Sample range slice.
            - ignore_insufficient (:obj:`bool`): If ignore_insufficient is true, sampling more than buffer size
                with no repetition will not cause an exception.
            - groupby (:obj:`str`): Groupby key in meta.
            - rolling_window (:obj:`int`): Return batches of window size.
        Returns:
            - sample_data (:obj:`Union[List[BufferedData], List[List[BufferedData]]]`):
                A list of data with length ``size``, may be nested if groupby or rolling_window is set.
        """
        storage = self.game_block_buffer
        if sample_range:
            storage = list(itertools.islice(self.storage, sample_range.start, sample_range.stop, sample_range.step))

        # Size and indices
        assert size or indices, "One of size and indices must not be empty."
        if (size and indices) and (size != len(indices)):
            raise AssertionError("Size and indices length must be equal.")
        if not size:
            size = len(indices)
        # Indices and groupby
        assert not (indices and groupby), "Cannot use groupby and indicex at the same time."
        # Groupby and rolling_window
        assert not (groupby and rolling_window), "Cannot use groupby and rolling_window at the same time."
        assert not (indices and rolling_window), "Cannot use indices and rolling_window at the same time."

        value_error = None
        sampled_data = []
        if indices:
            sampled_data = [self.game_block_buffer[game_block_idx] for game_block_idx in indices]

        elif groupby:
            sampled_data = self._sample_by_group(size=size, groupby=groupby, replace=replace, storage=storage)
        elif rolling_window:
            sampled_data = self._sample_by_rolling_window(
                size=size, replace=replace, rolling_window=rolling_window, storage=storage
            )
        else:
            if replace:
                sampled_data = random.choices(storage, k=size)
            else:
                try:
                    sampled_data = random.sample(storage, k=size)
                except ValueError as e:
                    value_error = e

        if value_error or len(sampled_data) != size:
            if ignore_insufficient:
                logging.warning(
                    "Sample operation is ignored due to data insufficient, current buffer is {} while sample is {}".
                    format(self.count(), size)
                )
            else:
                raise ValueError("There are less than {} records/groups in buffer({})".format(size, self.count()))

        return sampled_data

    def get_transition(self, idx):
        """
        Overview:
            sample one transition according to the idx
        """
        game_block_idx, pos_in_game_block = self.game_block_game_pos_look_up[idx]
        game_block_idx -= self.base_idx
        transition = self.game_block_buffer[game_block_idx][pos_in_game_block]
        return transition

    def get(self, idx: int) -> BufferedData:
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
        """
        # only update the priorities for data still in replay buffer
        for i in range(len(indices)):
            if metas['make_time'][i] > self.clear_time:
                idx, prio = indices[i], metas['batch_priorities'][i]
                self.game_pos_priorities[idx] = prio

    def update_priority(self, train_data, batch_priorities) -> None:
        # update priority in replay_buffer
        # inputs_batch, targets_batch, replay_buffer = train_data
        # obs_batch_ori, action_batch, mask_batch, indices, weights_lst, make_time = inputs_batch
        self.batch_update(indices=train_data[0][3],
                          metas={'make_time': train_data[0][5], 'batch_priorities': batch_priorities})

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
            game_block_list: a list of game histories
            pos_in_game_block_list: transition index in game (relative index)
            batch_index_list: the index of start transition of sampled minibatch in replay buffer
            weights_list: the weight concerning the priority
            make_time: the time the batch is made (for correctly updating replay buffer
                when data is deleted)
        Arguments:
            - batch_size: int batch size
            - beta: float the parameter in PER for calculating the priority
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

        # TODO
        if self._cfg.reanalyze_outdated is True:
            batch_index_list.sort()

        weights_list = (total * probs[batch_index_list]) ** (-beta)
        weights_list /= weights_list.max()

        game_block_list = []
        pos_in_game_block_list = []

        for idx in batch_index_list:
            game_block_idx, pos_in_game_block = self.game_block_game_pos_look_up[idx]
            game_block_idx -= self.base_idx
            game_block = self.game_block_buffer[game_block_idx]

            game_block_list.append(game_block)
            pos_in_game_block_list.append(pos_in_game_block)

        make_time = [time.time() for _ in range(len(batch_index_list))]

        context = (game_block_list, pos_in_game_block_list, batch_index_list, weights_list, make_time)
        return context

    # @profile
    def make_batch(self, batch_context, ratio):
        """
        Overview:
            prepare the context of a batch
            reward_value_context:        the context of reanalyzed value targets
            policy_re_context:           the context of reanalyzed policy targets
            policy_non_re_context:       the context of non-reanalyzed policy targets
            inputs_batch:                the inputs of batch
        Arguments:
            batch_context: Any batch context from replay buffer
            ratio: float ratio of reanalyzed policy (value is 100% reanalyzed)
        """
        # obtain the batch context from replay buffer
        game_block_list, pos_in_game_block_list, batch_index_list, weights_list, make_time_list = batch_context
        batch_size = len(batch_index_list)
        obs_list, action_list, mask_list = [], [], []
        # prepare the inputs of a batch
        for i in range(batch_size):
            game = game_block_list[i]
            pos_in_game_block = pos_in_game_block_list[i]

            _actions = game.action_history[pos_in_game_block:pos_in_game_block +
                                           self._cfg.num_unroll_steps].tolist()
            # add mask for invalid actions (out of trajectory)
            _mask = [1. for i in range(len(_actions))]
            _mask += [0. for _ in range(self._cfg.num_unroll_steps - len(_mask))]

            # pad random action
            _actions += [
                np.random.randint(0, game.action_space_size) for _ in range(self._cfg.num_unroll_steps - len(_actions))
            ]

            # obtain the input observations
            # stack+num_unroll_steps  4+5
            # pad if length of obs in game_block is less than stack+num_unroll_steps
            obs_list.append(
                game_block_list[i].obs(
                    pos_in_game_block_list[i], extra_len=self._cfg.num_unroll_steps, padding=True
                )
            )
            action_list.append(_actions)
            mask_list.append(_mask)

        # formalize the input observations
        obs_list = prepare_observation_list(obs_list)

        # formalize the inputs of a batch
        inputs_batch = [obs_list, action_list, mask_list, batch_index_list, weights_list, make_time_list]
        for i in range(len(inputs_batch)):
            inputs_batch[i] = np.asarray(inputs_batch[i])

        total_transitions = self.get_num_of_transitions()

        # obtain the context of value targets
        reward_value_context = self.prepare_reward_value_context(
            batch_index_list, game_block_list, pos_in_game_block_list, total_transitions
        )
        """
        only reanalyze recent ratio (e.g. 50%) data
        """
        reanalyze_num = int(batch_size * ratio)
        # if self._cfg.reanalyze_outdated is True:
        # batch_index_list is sorted according to its generated enn_steps

        # 0:reanalyze_num -> reanalyzed policy, reanalyze_num: end -> non reanalyzed policy
        # reanalyzed policy
        if reanalyze_num > 0:
            # obtain the context of reanalyzed policy targets
            policy_re_context = self.prepare_policy_reanalyzed_context(
                batch_index_list[:reanalyze_num], game_block_list[:reanalyze_num],
                pos_in_game_block_list[:reanalyze_num]
            )
        else:
            policy_re_context = None

        # non reanalyzed policy
        if reanalyze_num < batch_size:
            # obtain the context of non-reanalyzed policy targets
            policy_non_re_context = self.prepare_policy_non_reanalyzed_context(
                batch_index_list[reanalyze_num:], game_block_list[reanalyze_num:],
                pos_in_game_block_list[reanalyze_num:]
            )
        else:
            policy_non_re_context = None

        context = reward_value_context, policy_re_context, policy_non_re_context, inputs_batch
        return context

    def prepare_reward_value_context(
        self, batch_index_list, game_block_list, pos_in_game_block_list, total_transitions
    ):
        """
        Overview:
            prepare the context of rewards and values for calculating TD value target in reanalyzing part.
        Arguments:
            - batch_index_list (:obj:`list`): the index of start transition of sampled minibatch in replay buffer
            - game_block_list (:obj:`list`): list of game histories
            - pos_in_game_block_list (:obj:`list`): list of transition index in game_block
            - total_transitions (:obj:`int`): number of collected transitions
        """
        zero_obs = game_block_list[0].zero_obs()
        value_obs_list = []
        # the value is valid or not (out of trajectory)
        value_mask = []
        rewards_list = []
        traj_lens = []
        # for two_player board games
        action_mask_history, to_play_history = [], []

        td_steps_list = []
        for game_block, state_index, idx in zip(game_block_list, pos_in_game_block_list, batch_index_list):
            traj_len = len(game_block)
            traj_lens.append(traj_len)

            # for atari
            # # off-policy correction: shorter horizon of td steps
            # delta_td = (total_transitions - idx) // config.auto_td_steps
            # td_steps = config.td_steps - delta_td
            # td_steps = np.clip(td_steps, 1, 5).astype(np.int)
            # TODO(pu):
            td_steps = np.clip(self._cfg.td_steps, 1, max(1, traj_len - state_index)).astype(np.int32)

            # prepare the corresponding observations for bootstrapped values o_{t+k}
            # o[t+ td_steps, t + td_steps + stack frames + num_unroll_steps]
            # t=2+3 -> o[2+3, 2+3+4+5] -> o[5, 14]
            game_obs = game_block.obs(state_index + td_steps, self._cfg.num_unroll_steps)

            rewards_list.append(game_block.reward_history)

            # for two_player board games
            action_mask_history.append(game_block.action_mask_history)
            to_play_history.append(game_block.to_play_history)

            for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                # get the <num_unroll_steps+1>  bootstrapped target obs
                td_steps_list.append(td_steps)
                # index of bootstrapped obs o_{t+td_steps}
                bootstrap_index = current_index + td_steps

                if bootstrap_index < traj_len:
                    value_mask.append(1)
                    # beg_index = bootstrap_index - (state_index + td_steps)
                    # max of beg_index is num_unroll_steps
                    beg_index = current_index - state_index
                    end_index = beg_index + self._cfg.model.frame_stack_num
                    # the stacked obs in time t
                    obs = game_obs[beg_index:end_index]
                else:
                    value_mask.append(0)
                    obs = zero_obs

                value_obs_list.append(obs)

        reward_value_context = [
            value_obs_list, value_mask, pos_in_game_block_list, rewards_list, traj_lens, td_steps_list,
            action_mask_history, to_play_history
        ]
        return reward_value_context

    def prepare_policy_non_reanalyzed_context(self, batch_index_list, game_block_list, pos_in_game_block_list):
        """
        Overview:
            prepare the context of policies for calculating policy target in non-reanalyzing part, just return the policy in self-play
        Arguments:
            - batch_index_list (:obj:`list`): the index of start transition of sampled minibatch in replay buffer
            - game_block_list (:obj:`list`): list of game histories
            - pos_in_game_block_list (:obj:`list`): list transition index in game
        """
        child_visits = []
        traj_lens = []
        # for two_player board games
        action_mask_history, to_play_history = [], []

        for game_block, state_index, idx in zip(game_block_list, pos_in_game_block_list, batch_index_list):
            traj_len = len(game_block)
            traj_lens.append(traj_len)
            # for two_player board games
            action_mask_history.append(game_block.action_mask_history)
            to_play_history.append(game_block.to_play_history)

            child_visits.append(game_block.child_visit_history)

        policy_non_re_context = [
            pos_in_game_block_list, child_visits, traj_lens, action_mask_history, to_play_history
        ]
        return policy_non_re_context

    def prepare_policy_reanalyzed_context(self, batch_index_list, game_block_list, pos_in_game_block_list):
        """
        Overview:
            prepare the context of policies for calculating policy target in reanalyzing part.
        Arguments:
            - batch_index_list (:obj:'list'): start transition index in the replay buffer
            - game_block_list (:obj:'list'): list of game histories
            - pos_in_game_block_list (:obj:'list'): position of transition index in one game history
        """
        zero_obs = game_block_list[0].zero_obs()

        with torch.no_grad():
            # for policy
            policy_obs_list = []
            policy_mask = []  # 0 -> out of traj, 1 -> new policy
            rewards, child_visits, traj_lens = [], [], []
            # for two_player board games
            action_mask_history, to_play_history = [], []
            for game_block, state_index in zip(game_block_list, pos_in_game_block_list):
                traj_len = len(game_block)
                traj_lens.append(traj_len)
                rewards.append(game_block.reward_history)
                # for two_player board games
                action_mask_history.append(game_block.action_mask_history)
                to_play_history.append(game_block.to_play_history)

                child_visits.append(game_block.child_visit_history)
                # prepare the corresponding observations
                game_obs = game_block.obs(state_index, self._cfg.num_unroll_steps)
                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):

                    if current_index < traj_len:
                        policy_mask.append(1)
                        beg_index = current_index - state_index
                        end_index = beg_index + self._cfg.model.frame_stack_num
                        obs = game_obs[beg_index:end_index]
                    else:
                        policy_mask.append(0)
                        obs = zero_obs
                    policy_obs_list.append(obs)

        policy_re_context = [
            policy_obs_list, policy_mask, pos_in_game_block_list, batch_index_list, child_visits, traj_lens,
            action_mask_history, to_play_history
        ]
        return policy_re_context

    # @profile
    def compute_target_reward_value(self, reward_value_context, model):
        """
        Overview:
            prepare reward and value targets from the context of rewards and values.
        """
        value_obs_list, value_mask, pos_in_game_block_list, rewards_list, traj_lens, td_steps_list, action_mask_history, \
        to_play_history = reward_value_context
        device = self._cfg.device
        batch_size = len(value_obs_list)
        game_block_batch_size = len(pos_in_game_block_list)

        if to_play_history[0][0] is not None:
            # for two_player board games
            # to_play
            to_play = []
            for bs in range(game_block_batch_size):
                to_play_tmp = list(
                    to_play_history[bs][pos_in_game_block_list[bs]:pos_in_game_block_list[bs] +
                                        self._cfg.num_unroll_steps + 1]
                )
                if len(to_play_tmp) < self._cfg.num_unroll_steps + 1:
                    # effective play index is {1,2}
                    to_play_tmp += [0 for _ in range(self._cfg.num_unroll_steps + 1 - len(to_play_tmp))]
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
                    action_mask_history[bs][pos_in_game_block_list[bs]:pos_in_game_block_list[bs] +
                                            self._cfg.num_unroll_steps + 1]
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

        batch_values, batch_rewards = [], []
        with torch.no_grad():
            value_obs_list = prepare_observation_list(value_obs_list)
            # split a full batch into slices of mini_infer_size: to save the GPU memory for more GPU actors
            m_batch = self._cfg.mini_infer_size
            slices = np.ceil(batch_size / m_batch).astype(np.int_)
            network_output = []
            for i in range(slices):
                beg_index = m_batch * i
                end_index = m_batch * (i + 1)

                m_obs = torch.from_numpy(value_obs_list[beg_index:end_index]).to(device).float()

                # calculate the target value
                m_output = model.initial_inference(m_obs)

                # TODO(pu)
                if not model.training:
                    # if not in training, obtain the scalars of the value/reward
                    m_output.hidden_state = m_output.hidden_state.detach().cpu().numpy()
                    m_output.value = inverse_scalar_transform(m_output.value,
                                                              self._cfg.model.support_scale).detach().cpu().numpy()
                    m_output.policy_logits = m_output.policy_logits.detach().cpu().numpy()
                network_output.append(m_output)

            # concat the output slices after model inference
            if self._cfg.use_root_value:
                # use the root values from MCTS
                # the root values have limited improvement but require much more GPU actors;
                _, reward_pool, policy_logits_pool, hidden_state_roots = concat_output(network_output)
                reward_pool = reward_pool.squeeze().tolist()
                policy_logits_pool = policy_logits_pool.tolist()

                if self._cfg.mcts_ctree:
                    """
                    cpp mcts_tree
                    """
                    if to_play_history[0][0] is None:
                        # for one_player atari games
                        action_mask = [
                            list(np.ones(self._cfg.model.action_space_size, dtype=np.int8)) for _ in range(batch_size)
                        ]
                        to_play = [0 for i in range(batch_size)]

                    legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(batch_size)]
                    roots = ctree.Roots(batch_size, legal_actions)

                    noises = [
                        np.random.dirichlet([self._cfg.root_dirichlet_alpha] * self._cfg.model.action_space_size
                                            ).astype(np.float32).tolist() for _ in range(batch_size)
                    ]
                    roots.prepare(self._cfg.root_exploration_fraction, noises, reward_pool, policy_logits_pool, to_play)
                    # do MCTS for a new policy with the recent target model
                    MCTS_ctree(self._cfg).search(roots, model, hidden_state_roots, to_play)
                else:
                    """
                    python mcts_tree
                    """
                    if to_play_history[0][0] is None:
                        # for one_player atari games
                        action_mask = [
                            list(np.ones(self._cfg.model.action_space_size, dtype=np.int8)) for _ in range(batch_size)
                        ]
                    legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(batch_size)]
                    roots = ptree.Roots(batch_size, legal_actions, self._cfg.num_simulations)
                    noises = [
                        np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(sum(action_mask[j]))
                                            ).astype(np.float32).tolist() for j in range(batch_size)
                    ]

                    if to_play_history[0][0] is None:
                        roots.prepare(
                            self._cfg.root_exploration_fraction, noises, reward_pool, policy_logits_pool, to_play=None
                        )
                        # do MCTS for a new policy with the recent target model
                        MCTS_ptree(self._cfg).search(roots, model, hidden_state_roots, to_play=None)
                    else:
                        roots.prepare(
                            self._cfg.root_exploration_fraction,
                            noises,
                            reward_pool,
                            policy_logits_pool,
                            to_play=to_play
                        )
                        # do MCTS for a new policy with the recent target model
                        MCTS_ptree(self._cfg).search(roots, model, hidden_state_roots, to_play=to_play)

                roots_values = roots.get_values()
                value_list = np.array(roots_values)
            else:
                # use the predicted values
                value_list = concat_output_value(network_output)

            # get last state value
            value_list = value_list.reshape(-1) * (
                np.array([self._cfg.discount for _ in range(batch_size)]) ** td_steps_list
            )
            value_list = value_list * np.array(value_mask)
            value_list = value_list.tolist()

            horizon_id, value_index = 0, 0
            for traj_len_non_re, reward_list, state_index in zip(traj_lens, rewards_list, pos_in_game_block_list):
                # traj_len = len(game)
                target_values = []
                target_rewards = []

                reward = 0.0
                base_index = state_index
                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                    bootstrap_index = current_index + td_steps_list[value_index]
                    # for i, reward in enumerate(game.rewards[current_index:bootstrap_index]):
                    for i, reward in enumerate(reward_list[current_index:bootstrap_index]):
                        value_list[value_index] += reward * self._cfg.discount ** i

                    horizon_id += 1

                    if current_index < traj_len_non_re:
                        target_values.append(value_list[value_index])
                        # Since the horizon is small and the discount is close to 1.
                        # Compute the reward sum to approximate the value prefix for simplification
                        # reward += reward_list[current_index]  # * self._cfg.discount ** (current_index - base_index)
                        # reward += reward_list[current_index]  # * self._cfg.discount ** (current_index - base_index)
                        # target_rewards.append(reward)
                        target_rewards.append(reward_list[current_index])
                    else:
                        target_values.append(0)
                        # target_rewards.append(reward)
                        target_rewards.append(0.0)
                    value_index += 1

                batch_rewards.append(target_rewards)
                batch_values.append(target_values)

        batch_rewards = np.asarray(batch_rewards)
        batch_values = np.asarray(batch_values)
        return batch_rewards, batch_values

    # @profile
    def compute_target_policy_reanalyzed(self, policy_re_context, model):
        """
        compute policy targets from the reanalyzed context of policies
        """
        batch_target_policies_re = []
        if policy_re_context is None:
            return batch_target_policies_re

        # for two_player board games
        policy_obs_list, policy_mask, pos_in_game_block_list, batch_index_list, child_visits, traj_lens, action_mask_history, \
        to_play_history = policy_re_context
        batch_size = len(policy_obs_list)
        game_block_batch_size = len(pos_in_game_block_list)

        device = self._cfg.device

        if self._cfg.env_type == 'board_games':
            # for two_player board games: prepare the to_play and action_mask mini-batch

            # to_play
            to_play = []
            for bs in range(game_block_batch_size):
                to_play_tmp = list(
                    to_play_history[bs][pos_in_game_block_list[bs]:pos_in_game_block_list[bs] +
                                        self._cfg.num_unroll_steps + 1]
                )
                if len(to_play_tmp) < self._cfg.num_unroll_steps + 1:
                    # effective play index is {1,2}
                    to_play_tmp += [0 for _ in range(self._cfg.num_unroll_steps + 1 - len(to_play_tmp))]
                to_play.append(to_play_tmp)

            tmp = []
            for i in to_play:
                tmp += list(i)
            to_play = tmp

            # action_mask
            action_mask = []
            for bs in range(game_block_batch_size):
                action_mask_tmp = list(
                    action_mask_history[bs][pos_in_game_block_list[bs]:pos_in_game_block_list[bs] +
                                            self._cfg.num_unroll_steps + 1]
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
                for j in range(max(self._cfg.num_unroll_steps + 1, batch_size))
            ]

        with torch.no_grad():
            policy_obs_list = prepare_observation_list(policy_obs_list)
            # split a full batch into slices of mini_infer_size: to save the GPU memory for more GPU actors
            m_batch = self._cfg.mini_infer_size
            slices = np.ceil(batch_size / m_batch).astype(np.int_)
            network_output = []
            for i in range(slices):
                beg_index = m_batch * i
                end_index = m_batch * (i + 1)

                m_obs = torch.from_numpy(policy_obs_list[beg_index:end_index]).to(device).float()

                m_output = model.initial_inference(m_obs)
                # TODO(pu)
                if not model.training:
                    # if not in training, obtain the scalars of the value/reward
                    m_output.hidden_state = m_output.hidden_state.detach().cpu().numpy()
                    m_output.value = inverse_scalar_transform(m_output.value,
                                                              self._cfg.model.support_scale).detach().cpu().numpy()
                    m_output.policy_logits = m_output.policy_logits.detach().cpu().numpy()

                network_output.append(m_output)

            _, reward_pool, policy_logits_pool, hidden_state_roots = concat_output(network_output)
            reward_pool = reward_pool.squeeze().tolist()
            policy_logits_pool = policy_logits_pool.tolist()
            if self._cfg.mcts_ctree:
                """
                cpp mcts_tree
                """
                if to_play_history[0][0] is None:
                    # for one_player atari games
                    action_mask = [
                        list(np.ones(self._cfg.model.action_space_size, dtype=np.int8)) for _ in range(batch_size)
                    ]
                    to_play = [0 for i in range(batch_size)]

                legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(batch_size)]
                roots = ctree.Roots(batch_size, legal_actions)

                noises = [
                    np.random.dirichlet([self._cfg.root_dirichlet_alpha] * self._cfg.model.action_space_size
                                        ).astype(np.float32).tolist() for _ in range(batch_size)
                ]
                roots.prepare(self._cfg.root_exploration_fraction, noises, reward_pool, policy_logits_pool, to_play)
                # do MCTS for a new policy with the recent target model
                MCTS_ctree(self._cfg).search(roots, model, hidden_state_roots, to_play)

                # TODO(pu)
                # roots_legal_actions_list = roots.legal_actions_list
                roots_legal_actions_list = legal_actions
            else:
                """
                python mcts_tree
                """
                if to_play_history[0][0] is None:
                    # for one_player atari games
                    action_mask = [
                        list(np.ones(self._cfg.model.action_space_size, dtype=np.int8)) for _ in range(batch_size)
                    ]
                    legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(batch_size)]

                roots = ptree.Roots(batch_size, legal_actions, self._cfg.num_simulations)
                noises = [
                    np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(sum(action_mask[j]))
                                        ).astype(np.float32).tolist() for j in range(batch_size)
                ]
                if to_play_history[0][0] is None:
                    roots.prepare(
                        self._cfg.root_exploration_fraction, noises, reward_pool, policy_logits_pool, to_play=None
                    )
                    # do MCTS for a new policy with the recent target model
                    MCTS_ptree(self._cfg).search(roots, model, hidden_state_roots, to_play=None)
                else:
                    roots.prepare(
                        self._cfg.root_exploration_fraction, noises, reward_pool, policy_logits_pool, to_play=to_play
                    )
                    # do MCTS for a new policy with the recent target model
                    MCTS_ptree(self._cfg).search(roots, model, hidden_state_roots, to_play=to_play)
                roots_legal_actions_list = roots.legal_actions_list

            roots_distributions = roots.get_distributions()

            policy_index = 0
            for state_index, game_index in zip(pos_in_game_block_list, batch_index_list):
                target_policies = []

                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                    distributions = roots_distributions[policy_index]

                    if policy_mask[policy_index] == 0:
                        target_policies.append([0 for _ in range(self._cfg.model.action_space_size)])
                    else:
                        if distributions is None:
                            # if at some obs, the legal_action is None, add the fake target_policy
                            target_policies.append(
                                list(np.ones(self._cfg.model.action_space_size) / self._cfg.model.action_space_size)
                            )
                        else:
                            if self._cfg.mcts_ctree:
                                """
                                cpp mcts_tree
                                """
                                if to_play_history[0][0] is None:
                                    # for one_player atari games
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
                            else:
                                """
                                python mcts_tree
                                """
                                if to_play_history[0][0] is None:
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

        return batch_target_policies_re

    # @profile
    def compute_target_policy_non_reanalyzed(self, policy_non_re_context):
        """
        Overview:
            prepare policy targets from the non-reanalyzed context of policies
        Arguments:
            - policy_non_re_context (:obj:`List`): List containing:
                - pos_in_game_block_list
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

        pos_in_game_block_list, child_visits, traj_lens, action_mask_history, to_play_history = policy_non_re_context

        game_block_batch_size = len(pos_in_game_block_list)

        if self._cfg.env_type == 'board_games':
            # for two_player board games
            # action_mask
            action_mask = []
            for bs in range(game_block_batch_size):
                action_mask_tmp = list(
                    action_mask_history[bs][pos_in_game_block_list[bs]:pos_in_game_block_list[bs] +
                                            self._cfg.num_unroll_steps + 1]
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
            # for game, state_index in zip(games, pos_in_game_block_list):
            for traj_len, child_visit, state_index in zip(traj_lens, child_visits, pos_in_game_block_list):
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
                        # the invalid target policy
                        policy_mask.append(0)
                        target_policies.append([0 for _ in range(self._cfg.model.action_space_size)])

                    policy_index += 1

                batch_target_policies_non_re.append(target_policies)
        batch_target_policies_non_re = np.asarray(batch_target_policies_non_re)
        return batch_target_policies_non_re

    def sample_train_data(self, batch_size, policy):
        """
        Overview:
            sample data from ``GameBuffer`` and prepare the current and target batch for training
        """
        policy._target_model.to(self._cfg.device)
        policy._target_model.eval()

        batch_context = self.prepare_batch_context(batch_size, self._cfg.priority_prob_beta)
        input_context = self.make_batch(batch_context, self._cfg.reanalyze_ratio)
        reward_value_context, policy_re_context, policy_non_re_context, inputs_batch = input_context

        # target reward, value
        batch_rewards, batch_values = self.compute_target_reward_value(reward_value_context, policy._target_model)
        # target policy
        batch_target_policies_re = self.compute_target_policy_reanalyzed(policy_re_context, policy._target_model)
        batch_target_policies_non_re = self.compute_target_policy_non_reanalyzed(policy_non_re_context)

        if 0 < self._cfg.reanalyze_ratio < 1:
            batch_policies = np.concatenate([batch_target_policies_re, batch_target_policies_non_re])
        elif self._cfg.reanalyze_ratio == 1:
            batch_policies = batch_target_policies_re
        elif self._cfg.reanalyze_ratio == 0:
            batch_policies = batch_target_policies_non_re

        targets_batch = [batch_rewards, batch_values, batch_policies]
        # a batch contains the inputs and the targets
        train_data = [inputs_batch, targets_batch]
        return train_data

    def save_data(self, file_name: str):
        """
        Overview:
            Save buffer data into a file.
        Arguments:
            - file_name (:obj:`str`): file name of buffer data
        """
        pass

    def load_data(self, file_name: str):
        """
        Overview:
            Load buffer data from a file.
        Arguments:
            - file_name (:obj:`str`): file name of buffer data
        """
        pass
