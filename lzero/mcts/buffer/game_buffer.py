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
        The base game buffer class for MuZeroPolicy, EfficientZeroPolicy, SampledEfficientZeroPolicy, GumbelMuZeroPolicy.
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
        reanalyze_ratio=0,
        # (bool) Whether to consider outdated experiences for reanalyzing. If True, we first sort the data in the minibatch by the time it was produced
        # and only reanalyze the oldest ``reanalyze_ratio`` fraction.
        reanalyze_outdated=True,
        # (bool) Whether to use the root value in the reanalyzing part. Please refer to EfficientZero paper for details.
        use_root_value=False,
        # (int) The number of samples required for mini inference.
        mini_infer_size=10240,
        # (str) The type of sampled data. The default is 'transition'. Options: 'transition', 'episode'.
        sample_type='transition',
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
        assert self._cfg.action_type in ['fixed_action_space', 'varied_action_space']

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
            self, batch_size: int, policy: Union["MuZeroPolicy", "EfficientZeroPolicy", "SampledEfficientZeroPolicy", "GumbelMuZeroPolicy"]
    ) -> List[Any]:
        """
        Overview:
            sample data from ``GameBuffer`` and prepare the current and target batch for training.
        Arguments:
            - batch_size (:obj:`int`): batch size.
            - policy (:obj:`Union["MuZeroPolicy", "EfficientZeroPolicy", "SampledEfficientZeroPolicy", "GumbelMuZeroPolicy"]`): policy.
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

            # print(f'len(game_segment)=:len(game_segment.action_segment): {len(game_segment)}')
            # print(f'len(game_segment.obs_segment): {game_segment.obs_segment.shape[0]}')

            # In the reanalysis phase, `pos_in_game_segment` should be a multiple of `num_unroll_steps`.
            # Indices exceeding `game_segment_length` are padded with the next segment and are not updated
            # in the current implementation. Therefore, we need to sample `pos_in_game_segment` within
            # [0, game_segment_length - num_unroll_steps] to avoid padded data.

            if self._cfg.action_type == 'varied_action_space':
                # TODO: Consider increasing `self._cfg.game_segment_length` to ensure sampling efficiency.
                if pos_in_game_segment >= self._cfg.game_segment_length - self._cfg.num_unroll_steps:
                    pos_in_game_segment = np.random.choice(self._cfg.game_segment_length - self._cfg.num_unroll_steps, 1).item()
            else:
                # NOTE: Sample the init position from the whole segment, but not from the padded part
                if pos_in_game_segment >= self._cfg.game_segment_length:
                    pos_in_game_segment = np.random.choice(self._cfg.game_segment_length, 1).item()

            pos_in_game_segment_list.append(pos_in_game_segment)
            

        make_time = [time.time() for _ in range(len(batch_index_list))]

        orig_data = (game_segment_list, pos_in_game_segment_list, batch_index_list, weights_list, make_time)
        return orig_data

    def _sample_orig_reanalyze_batch(self, batch_size: int) -> Tuple:
        """
        Overview:
            This function samples a batch of game segments for reanalysis from the replay buffer.
            It uses priority sampling based on the `reanalyze_time` of each game segment, with segments
            that have been reanalyzed more frequently receiving lower priority.

            The function returns a tuple containing information about the sampled game segments,
            including their positions within each segment and the time the batch was created.
        Arguments:
            - batch_size (:obj:`int`):
                The number of samples to draw in this batch.

        Returns:
            - Tuple:
                A tuple containing the following elements:
                - game_segment_list: A list of the sampled game segments.
                - pos_in_game_segment_list: A list of indices representing the position of each transition
                  within its corresponding game segment.
                - batch_index_list: The indices of the sampled game segments in the replay buffer.
                - make_time: A list of timestamps (set to `0` in this implementation) indicating when
                  the batch was created.

        Key Details:
            1. **Priority Sampling**:
               Game segments are sampled based on a probability distribution calculated using
               the `reanalyze_time` of each segment. Segments that have been reanalyzed more frequently
               are less likely to be selected.
            2. **Segment Slicing**:
               Each selected game segment is sampled at regular intervals determined by the
               `num_unroll_steps` parameter. Up to `samples_per_segment` transitions are sampled
               from each selected segment.
            3. **Handling Extra Samples**:
               If the `batch_size` is not perfectly divisible by the number of samples per segment,
               additional segments are sampled to make up the difference.
            4. **Reanalyze Time Update**:
               The `reanalyze_time` attribute of each sampled game segment is incremented to reflect
               that it has been selected for reanalysis again.
        Raises:
            - ValueError:
                If the `game_segment_length` is too small to accommodate the `num_unroll_steps`.
        """
        train_sample_num = len(self.game_segment_buffer)
        assert self._cfg.reanalyze_partition <= 0.75, "The reanalyze partition should be less than 0.75."
        valid_sample_num = int(train_sample_num * self._cfg.reanalyze_partition)

        # Calculate the number of samples per segment
        samples_per_segment = self._cfg.game_segment_length // self._cfg.num_unroll_steps

        # Make sure that the batch size can be divided by the number of samples per segment
        if samples_per_segment == 0:
            raise ValueError("The game segment length is too small for num_unroll_steps.")

        # Calculate the number of samples per segment
        batch_size_per_segment = batch_size // samples_per_segment

        # If the batch size cannot be divided, process the remainder part
        extra_samples = batch_size % samples_per_segment

        # We use the reanalyze_time in the game_segment_buffer to generate weights
        reanalyze_times = np.array([segment.reanalyze_time for segment in self.game_segment_buffer[:valid_sample_num]])

        # Calculate weights: the larger the reanalyze_time, the smaller the weight (use exp(-reanalyze_time))
        base_decay_rate = 100
        decay_rate = base_decay_rate / valid_sample_num
        weights = np.exp(-decay_rate * reanalyze_times)

        # Normalize the weights to a probability distribution
        probabilities = weights / np.sum(weights)

        # Sample game segments according to the probabilities
        selected_game_segments = np.random.choice(valid_sample_num, batch_size_per_segment, replace=False,
                                                  p=probabilities)

        # If there are extra samples to be allocated, randomly select some game segments and sample again
        if extra_samples > 0:
            extra_game_segments = np.random.choice(valid_sample_num, extra_samples, replace=False, p=probabilities)
            selected_game_segments = np.concatenate((selected_game_segments, extra_game_segments))

        game_segment_list = []
        pos_in_game_segment_list = []
        batch_index_list = []

        for game_segment_idx in selected_game_segments:
            game_segment_idx -= self.base_idx
            game_segment = self.game_segment_buffer[game_segment_idx]

            # Update reanalyze_time only once
            game_segment.reanalyze_time += 1

            # The sampling position should be 0, 0 + num_unroll_steps, ... (integer multiples of num_unroll_steps)
            for i in range(samples_per_segment):
                game_segment_list.append(game_segment)
                pos_in_game_segment = i * self._cfg.num_unroll_steps
                if pos_in_game_segment >= len(game_segment):
                    pos_in_game_segment = np.random.choice(len(game_segment), 1).item()
                pos_in_game_segment_list.append(pos_in_game_segment)
                batch_index_list.append(game_segment_idx)

        # Set the make_time for each sample (set to 0 for now, but can be the actual time if needed).
        make_time = [0. for _ in range(len(batch_index_list))]

        orig_data = (game_segment_list, pos_in_game_segment_list, batch_index_list, [], make_time)
        return orig_data

    def _sample_orig_reanalyze_data(self, batch_size: int) -> Tuple:
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
        segment_length = (self.get_num_of_transitions()//2000)
        assert self._beta > 0
        num_of_transitions = self.get_num_of_transitions()
        sample_points = num_of_transitions // segment_length

        batch_index_list = np.random.choice(2000, batch_size, replace=False)

        if self._cfg.reanalyze_outdated is True:
            # NOTE: used in reanalyze part
            batch_index_list.sort()

        # TODO(xcy): use weighted sample
        game_segment_list = []
        pos_in_game_segment_list = []

        for idx in batch_index_list:
            game_segment_idx, pos_in_game_segment = self.game_segment_game_pos_look_up[idx*segment_length]
            game_segment_idx -= self.base_idx
            game_segment = self.game_segment_buffer[game_segment_idx]

            game_segment_list.append(game_segment)
            pos_in_game_segment_list.append(pos_in_game_segment)

        make_time = [time.time() for _ in range(len(batch_index_list))]

        orig_data = (game_segment_list, pos_in_game_segment_list, batch_index_list, [], make_time)
        return orig_data

    def _sample_orig_data_episode(self, batch_size: int) -> Tuple:
        """
        Overview:
            Sample original data for a training batch, which includes:
                - game_segment_list: A list of game segments.
                - pos_in_game_segment_list: Indices of transitions within the game segments.
                - batch_index_list: Indices of the start transitions of the sampled mini-batch in the replay buffer.
                - weights_list: Weights for each sampled transition, used for prioritization.
                - make_time: Timestamps indicating when the batch was created (useful for managing replay buffer updates).
        Arguments:
            - batch_size (:obj:`int`): The number of samples to draw for the batch.
            - beta (:obj:`float`): Parameter for Prioritized Experience Replay (PER) that adjusts the importance of samples.
        """
        assert self._beta > 0, "Beta must be greater than zero."

        num_of_transitions = self.get_num_of_transitions()

        if not self._cfg.use_priority:
            self.game_pos_priorities = np.ones_like(self.game_pos_priorities)

        # Add a small constant for numerical stability
        probs = self.game_pos_priorities ** self._alpha + 1e-6
        probs /= probs.sum()

        # Sample game segment indices
        num_of_game_segments = self.get_num_of_game_segments()
        batch_episode_index_list = np.random.choice(num_of_game_segments, batch_size, replace=False)

        if self._cfg.reanalyze_outdated:
            # Sort for consistency when reanalyzing
            batch_episode_index_list.sort()

        batch_index_list = batch_episode_index_list * self._cfg.game_segment_length

        # Calculate weights for the sampled transitions
        weights_list = (num_of_transitions * probs[batch_index_list]) ** (-self._beta)
        weights_list /= weights_list.max()

        game_segment_list = []
        pos_in_game_segment_list = []

        # Collect game segments and their initial positions
        for episode_index in batch_episode_index_list:
            game_segment = self.game_segment_buffer[episode_index]
            game_segment_list.append(game_segment)
            pos_in_game_segment_list.append(0)  # Starting position in game segments

        # Record the time when the batch is created
        make_time = [time.time() for _ in range(len(batch_episode_index_list))]

        orig_data = (game_segment_list, pos_in_game_segment_list, batch_index_list, weights_list, make_time)
        return orig_data

    def _preprocess_to_play_and_action_mask(
            self, game_segment_batch_size, to_play_segment, action_mask_segment, pos_in_game_segment_list, unroll_steps = None
    ):
        """
        Overview:
            prepare the to_play and action_mask for the target obs in ``value_obs_list``
                - to_play: {list: game_segment_batch_size * (num_unroll_steps+1)}
                - action_mask: {list: game_segment_batch_size * (num_unroll_steps+1)}
        """
        unroll_steps = unroll_steps if unroll_steps is not None else self._cfg.num_unroll_steps

        to_play = []
        for bs in range(game_segment_batch_size):
            to_play_tmp = list(
                to_play_segment[bs][pos_in_game_segment_list[bs]:pos_in_game_segment_list[bs] +
                                                                 unroll_steps + 1]
            )
            if len(to_play_tmp) < unroll_steps + 1:
                # NOTE: the effective to play index is {1,2}, for null padding data, we set to_play=-1
                to_play_tmp += [-1 for _ in range(unroll_steps + 1 - len(to_play_tmp))]
            to_play.append(to_play_tmp)
        to_play = sum(to_play, [])

        if self._cfg.model.continuous_action_space is True:
            # when the action space of the environment is continuous, action_mask[:] is None.
            return to_play, None

        action_mask = []
        for bs in range(game_segment_batch_size):
            action_mask_tmp = list(
                action_mask_segment[bs][pos_in_game_segment_list[bs]:pos_in_game_segment_list[bs] +
                                                                     unroll_steps + 1]
            )
            if len(action_mask_tmp) < unroll_steps + 1:
                action_mask_tmp += [
                    list(np.ones(self._cfg.model.action_space_size, dtype=np.int8))
                    for _ in range(unroll_steps + 1 - len(action_mask_tmp))
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
        try:
            data_length = len(data.action_segment) if len(data.action_segment) < self._cfg.game_segment_length else self._cfg.game_segment_length
        except Exception as e:
            # to be compatible with unittest
            print(e)
            data_length = len(data)

        if meta['done']:
            self.num_of_collected_episodes += 1
            valid_len = data_length
        else:
            valid_len = data_length - meta['unroll_plus_td_steps']
            # print(f'valid_len is {valid_len}')

        if meta['priorities'] is None:
            if self.game_segment_buffer:
                max_prio = self.game_pos_priorities.max() if len(self.game_pos_priorities) > 0 else 1
            else:
                max_prio = 1
            
            # if no 'priorities' provided, set the valid part of the new-added game history the max_prio
            self.game_pos_priorities = np.concatenate((self.game_pos_priorities, [max_prio for _ in range(valid_len)] + [0. for _ in range(valid_len, data_length)]))
        else:
            assert data_length == len(meta['priorities']), " priorities should be of same length as the game steps"
            priorities = meta['priorities'].copy().reshape(-1)
            priorities[valid_len:data_length] = 0.
            self.game_pos_priorities = np.concatenate((self.game_pos_priorities, priorities))

        self.game_segment_buffer.append(data)
        self.game_segment_game_pos_look_up += [
            (self.base_idx + len(self.game_segment_buffer) - 1, step_pos) for step_pos in range(data_length)
        ]
        # print(f'potioritys is {self.game_pos_priorities}')
        # print(f'num of transitions is {len(self.game_segment_game_pos_look_up)}')

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
                length_data = len(self.game_segment_buffer[i].action_segment) if len(self.game_segment_buffer[i].action_segment)<self._cfg.game_segment_length else self._cfg.game_segment_length
                total_transition -= length_data
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
