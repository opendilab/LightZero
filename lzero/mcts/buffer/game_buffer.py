import copy
import time
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional, Union, TYPE_CHECKING

import numpy as np
from ding.torch_utils.data_helper import to_list
from ding.utils import BUFFER_REGISTRY
from easydict import EasyDict

if TYPE_CHECKING:
    from lzero.policy import MuZeroPolicy, EfficientZeroPolicy, SampledEfficientZeroPolicy, GumbelMuZeroPolicy


@BUFFER_REGISTRY.register('game_buffer')
class GameBuffer(ABC, object):
    """
    Overview:
        The base game buffer class for MuZeroPolicy, EfficientZeroPolicy, SampledEfficientZeroPolicy, GumeblMuZeroPolicy.
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    # Default configuration for GameBuffer.
    config = dict(
        # (int) The size/capacity of the replay buffer in terms of transitions.
        replay_buffer_size=int(1e6),
        # (float) The ratio of experiences required for the reanalyzing part in a minibatch.
        reanalyze_ratio=0.3,
        # (bool) Whether to consider outdated experiences for reanalyzing. If True, we first sort the data in the minibatch by the time it was produced
        # and only reanalyze the oldest ``reanalyze_ratio`` fraction.
        reanalyze_outdated=True,
        # (bool) Whether to use the root value in the reanalyzing part. Please refer to EfficientZero paper for details.
        use_root_value=False,
        # (int) The number of samples required for mini inference.
        mini_infer_size=256,
    )

    def __init__(self, cfg: dict):
        super().__init__()
        """
        Overview:
            Use the default configuration mechanism. If a user passes in a cfg with a key that matches an existing key
            in the default configuration, the user-provided value will override the default configuration. Otherwise,
            the default configuration will be used.
        """
        default_config = self.default_config()
        default_config.update(cfg)
        self._cfg = default_config
        self._cfg = cfg
        assert self._cfg.env_type in ['not_board_games', 'board_games']
        self.replay_buffer_size = self._cfg.replay_buffer_size
        self.batch_size = self._cfg.batch_size
        self._alpha = self._cfg.priority_prob_alpha
        self._beta = self._cfg.priority_prob_beta

        self.game_segment_buffer = []
        self.game_pos_priorities = []
        self.game_segment_game_pos_look_up = []

        self.keep_ratio = 1
        self.num_of_collected_episodes = 0
        self.base_idx = 0
        self.clear_time = 0

    @abstractmethod
    def sample(
            self, batch_size: int, policy: Union["MuZeroPolicy", "EfficientZeroPolicy", "SampledEfficientZeroPolicy", "GumeblMuZeroPolicy"]
    ) -> List[Any]:
        """
        Overview:
            sample data from ``GameBuffer`` and prepare the current and target batch for training.
        Arguments:
            - batch_size (:obj:`int`): batch size.
            - policy (:obj:`Union["MuZeroPolicy", "EfficientZeroPolicy", "SampledEfficientZeroPolicy", "GumeblMuZeroPolicy"]`): policy.
        Returns:
            - train_data (:obj:`List`): List of train data, including current_batch and target_batch.
        """

    @abstractmethod
    def _make_batch(self, orig_data: Any, reanalyze_ratio: float) -> Tuple[Any]:
        """
        Overview:
            prepare the context of a batch
            reward_value_context:        the context of reanalyzed value targets
            policy_re_context:           the context of reanalyzed policy targets
            policy_non_re_context:       the context of non-reanalyzed policy targets
            current_batch:                the inputs of batch
        Arguments:
            orig_data: Any batch context from replay buffer
            reanalyze_ratio: float ratio of reanalyzed policy (value is 100% reanalyzed)
        Returns:
            - context (:obj:`Tuple`): reward_value_context, policy_re_context, policy_non_re_context, current_batch
        """
        pass

    def _sample_orig_data(self, batch_size: int) -> Tuple:
        """
        Overview:
             sample orig_data that contains:
                game_segment_list: a list of game segments
                pos_in_game_segment_list: transition index in game (relative index)
                batch_index_list: the index of start transition of sampled minibatch in replay buffer
                weights_list: the weight concerning the priority
                make_time: the time the batch is made (for correctly updating replay buffer when data is deleted)
        Arguments:
            - batch_size (:obj:`int`): batch size
            - beta: float the parameter in PER for calculating the priority
        """
        assert self._beta > 0
        num_of_transitions = self.get_num_of_transitions()
        if self._cfg.use_priority is False:
            self.game_pos_priorities = np.ones_like(self.game_pos_priorities)

        # +1e-6 for numerical stability
        probs = self.game_pos_priorities ** self._alpha + 1e-6
        probs /= probs.sum()

        # sample according to transition index
        # TODO(pu): replace=True
        batch_index_list = np.random.choice(num_of_transitions, batch_size, p=probs, replace=False)

        if self._cfg.reanalyze_outdated is True:
            # NOTE: used in reanalyze part
            batch_index_list.sort()

        weights_list = (num_of_transitions * probs[batch_index_list]) ** (-self._beta)
        weights_list /= weights_list.max()

        game_segment_list = []
        pos_in_game_segment_list = []

        for idx in batch_index_list:
            game_segment_idx, pos_in_game_segment = self.game_segment_game_pos_look_up[idx]
            game_segment_idx -= self.base_idx
            game_segment = self.game_segment_buffer[game_segment_idx]

            game_segment_list.append(game_segment)
            pos_in_game_segment_list.append(pos_in_game_segment)

        make_time = [time.time() for _ in range(len(batch_index_list))]

        orig_data = (game_segment_list, pos_in_game_segment_list, batch_index_list, weights_list, make_time)
        return orig_data

    def _preprocess_to_play_and_action_mask(
        self, game_segment_batch_size, to_play_segment, action_mask_segment, pos_in_game_segment_list
    ):
        """
        Overview:
            prepare the to_play and action_mask for the target obs in ``value_obs_list``
                - to_play: {list: game_segment_batch_size * (num_unroll_steps+1)}
                - action_mask: {list: game_segment_batch_size * (num_unroll_steps+1)}
        """
        to_play = []
        for bs in range(game_segment_batch_size):
            to_play_tmp = list(
                to_play_segment[bs][pos_in_game_segment_list[bs]:pos_in_game_segment_list[bs] +
                                    self._cfg.num_unroll_steps + 1]
            )
            if len(to_play_tmp) < self._cfg.num_unroll_steps + 1:
                # NOTE: the effective to play index is {1,2}, for null padding data, we set to_play=-1
                to_play_tmp += [-1 for _ in range(self._cfg.num_unroll_steps + 1 - len(to_play_tmp))]
            to_play.append(to_play_tmp)
        to_play = sum(to_play, [])

        if self._cfg.model.continuous_action_space is True:
            # when the action space of the environment is continuous, action_mask[:] is None.
            return to_play, None

        action_mask = []
        for bs in range(game_segment_batch_size):
            action_mask_tmp = list(
                action_mask_segment[bs][pos_in_game_segment_list[bs]:pos_in_game_segment_list[bs] +
                                        self._cfg.num_unroll_steps + 1]
            )
            if len(action_mask_tmp) < self._cfg.num_unroll_steps + 1:
                action_mask_tmp += [
                    list(np.ones(self._cfg.model.action_space_size, dtype=np.int8))
                    for _ in range(self._cfg.num_unroll_steps + 1 - len(action_mask_tmp))
                ]
            action_mask.append(action_mask_tmp)
        action_mask = to_list(action_mask)
        action_mask = sum(action_mask, [])

        return to_play, action_mask

    @abstractmethod
    def _prepare_reward_value_context(
            self, batch_index_list: List[str], game_segment_list: List[Any], pos_in_game_segment_list: List[Any],
            total_transitions: int
    ) -> List[Any]:
        """
        Overview:
            prepare the context of rewards and values for calculating TD value target in reanalyzing part.
        Arguments:
            - batch_index_list (:obj:`list`): the index of start transition of sampled minibatch in replay buffer
            - game_segment_list (:obj:`list`): list of game segments
            - pos_in_game_segment_list (:obj:`list`): list of transition index in game_segment
            - total_transitions (:obj:`int`): number of collected transitions
        Returns:
            - reward_value_context (:obj:`list`): value_obs_lst, value_mask, state_index_lst, rewards_lst, game_segment_lens,
              td_steps_lst, action_mask_segment, to_play_segment
        """
        pass

    @abstractmethod
    def _prepare_policy_non_reanalyzed_context(
            self, batch_index_list: List[int], game_segment_list: List[Any], pos_in_game_segment_list: List[int]
    ) -> List[Any]:
        """
        Overview:
            prepare the context of policies for calculating policy target in non-reanalyzing part, just return the policy in self-play
        Arguments:
            - batch_index_list (:obj:`list`): the index of start transition of sampled minibatch in replay buffer
            - game_segment_list (:obj:`list`): list of game segments
            - pos_in_game_segment_list (:obj:`list`): list transition index in game
        Returns:
            - policy_non_re_context (:obj:`list`): state_index_lst, child_visits, game_segment_lens, action_mask_segment, to_play_segment
        """
        pass

    @abstractmethod
    def _prepare_policy_reanalyzed_context(
            self, batch_index_list: List[str], game_segment_list: List[Any], pos_in_game_segment_list: List[str]
    ) -> List[Any]:
        """
        Overview:
            prepare the context of policies for calculating policy target in reanalyzing part.
        Arguments:
            - batch_index_list (:obj:'list'): start transition index in the replay buffer
            - game_segment_list (:obj:'list'): list of game segments
            - pos_in_game_segment_list (:obj:'list'): position of transition index in one game history
        Returns:
            - policy_re_context (:obj:`list`): policy_obs_lst, policy_mask, state_index_lst, indices,
              child_visits, game_segment_lens, action_mask_segment, to_play_segment
        """
        pass

    @abstractmethod
    def _compute_target_reward_value(self, reward_value_context: List[Any], model: Any) -> List[np.ndarray]:
        """
        Overview:
            prepare reward and value targets from the context of rewards and values.
        Arguments:
            - reward_value_context (:obj:'list'): the reward value context
            - model (:obj:'torch.tensor'):model of the target model
        Returns:
            - batch_value_prefixs (:obj:'np.ndarray): batch of value prefix
            - batch_target_values (:obj:'np.ndarray): batch of value estimation
        """
        pass

    @abstractmethod
    def _compute_target_policy_reanalyzed(self, policy_re_context: List[Any], model: Any) -> np.ndarray:
        """
        Overview:
            prepare policy targets from the reanalyzed context of policies
        Arguments:
            - policy_re_context (:obj:`List`): List of policy context to reanalyzed
        Returns:
            - batch_target_policies_re
        """
        pass

    @abstractmethod
    def _compute_target_policy_non_reanalyzed(
            self, policy_non_re_context: List[Any], policy_shape: Optional[int]
    ) -> np.ndarray:
        """
        Overview:
            prepare policy targets from the non-reanalyzed context of policies
        Arguments:
            - policy_non_re_context (:obj:`List`): List containing:
                - pos_in_game_segment_list
                - child_visits
                - game_segment_lens
                - action_mask_segment
                - to_play_segment
        Returns:
            - batch_target_policies_non_re
        """
        pass

    @abstractmethod
    def update_priority(
            self, train_data: Optional[List[Optional[np.ndarray]]], batch_priorities: Optional[Any]
    ) -> None:
        """
        Overview:
            Update the priority of training data.
        Arguments:
            - train_data (:obj:`Optional[List[Optional[np.ndarray]]]`): training data to be updated priority.
            - batch_priorities (:obj:`batch_priorities`): priorities to update to.
        """
        pass

    def push_game_segments(self, data_and_meta: Any) -> None:
        """
        Overview:
            Push game_segments data and it's meta information into buffer.
            Save a game segment
        Arguments:
            - data_and_meta
                - data (:obj:`Any`): The data (game segments) which will be pushed into buffer.
                - meta (:obj:`dict`): Meta information, e.g. priority, count, staleness.
        """
        data, meta = data_and_meta
        for (data_game, meta_game) in zip(data, meta):
            self._push_game_segment(data_game, meta_game)

    def _push_game_segment(self, data: Any, meta: Optional[dict] = None) -> None:
        """
        Overview:
            Push data and it's meta information in buffer.
            Save a game segment.
        Arguments:
            - data (:obj:`Any`): The data (a game segment) which will be pushed into buffer.
            - meta (:obj:`dict`): Meta information, e.g. priority, count, staleness.
                - done (:obj:`bool`): whether the game is finished.
                - unroll_plus_td_steps (:obj:`int`): if the game is not finished, we only save the transitions that can be computed
                - priorities (:obj:`list`): the priorities corresponding to the transitions in the game history
        Returns:
            - buffered_data (:obj:`BufferedData`): The pushed data.
        """
        if meta['done']:
            self.num_of_collected_episodes += 1
            valid_len = len(data)
        else:
            valid_len = len(data) - meta['unroll_plus_td_steps']

        if meta['priorities'] is None:
            max_prio = self.game_pos_priorities.max() if self.game_segment_buffer else 1
            # if no 'priorities' provided, set the valid part of the new-added game history the max_prio
            self.game_pos_priorities = np.concatenate(
                (
                    self.game_pos_priorities, [max_prio
                                               for _ in range(valid_len)] + [0. for _ in range(valid_len, len(data))]
                )
            )
        else:
            assert len(data) == len(meta['priorities']), " priorities should be of same length as the game steps"
            priorities = meta['priorities'].copy().reshape(-1)
            priorities[valid_len:len(data)] = 0.
            self.game_pos_priorities = np.concatenate((self.game_pos_priorities, priorities))

        self.game_segment_buffer.append(data)
        self.game_segment_game_pos_look_up += [
            (self.base_idx + len(self.game_segment_buffer) - 1, step_pos) for step_pos in range(len(data))
        ]

    def remove_oldest_data_to_fit(self) -> None:
        """
        Overview:
            remove some oldest data if the replay buffer is full.
        """
        assert self.replay_buffer_size > self._cfg.batch_size, "replay buffer size should be larger than batch size"
        nums_of_game_segments = self.get_num_of_game_segments()
        total_transition = self.get_num_of_transitions()
        if total_transition > self.replay_buffer_size:
            index = 0
            for i in range(nums_of_game_segments):
                total_transition -= len(self.game_segment_buffer[i])
                if total_transition <= self.replay_buffer_size * self.keep_ratio:
                    # find the max game_segment index to keep in the buffer
                    index = i
                    break
            if total_transition >= self._cfg.batch_size:
                self._remove(index + 1)

    def _remove(self, excess_game_segment_index: List[int]) -> None:
        """
        Overview:
            delete game segments in index [0: excess_game_segment_index]
        Arguments:
            - excess_game_segment_index (:obj:`List[str]`): Index of data.
        """
        excess_game_positions = sum(
            [len(game_segment) for game_segment in self.game_segment_buffer[:excess_game_segment_index]]
        )
        del self.game_segment_buffer[:excess_game_segment_index]
        self.game_pos_priorities = self.game_pos_priorities[excess_game_positions:]
        del self.game_segment_game_pos_look_up[:excess_game_positions]
        self.base_idx += excess_game_segment_index
        self.clear_time = time.time()

    def get_num_of_episodes(self) -> int:
        # number of collected episodes
        return self.num_of_collected_episodes

    def get_num_of_game_segments(self) -> int:
        # num of game segments
        return len(self.game_segment_buffer)

    def get_num_of_transitions(self) -> int:
        # total number of transitions
        return len(self.game_segment_game_pos_look_up)

    def __repr__(self):
        return f'current buffer statistics is: num_of_all_collected_episodes: {self.num_of_collected_episodes}, num of game segments: {len(self.game_segment_buffer)}, number of transitions: {len(self.game_segment_game_pos_look_up)}'
