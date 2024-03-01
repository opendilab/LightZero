import copy
from typing import List, Tuple

import numpy as np
from easydict import EasyDict

from ding.utils.compression_helper import jpeg_data_decompressor


class GameSegment:
    """
    Overview:
        A game segment from a full episode trajectory.

        The length of one episode in (Atari) games is often quite large. This class represents a single game segment
        within a larger trajectory, split into several blocks.

    Interfaces:
        - __init__
        - __len__
        - reset
        - pad_over
        - is_full
        - legal_actions
        - append
        - get_observation
        - zero_obs
        - step_obs
        - get_targets
        - game_segment_to_array
        - store_search_stats
    """

    def __init__(self, action_space: int, game_segment_length: int = 200, config: EasyDict = None) -> None:
        """
        Overview:
            Init the ``GameSegment`` according to the provided arguments.
        Arguments:
             action_space (:obj:`int`): action space
            - game_segment_length (:obj:`int`): the transition number of one ``GameSegment`` block
        """
        self.action_space = action_space
        self.game_segment_length = game_segment_length
        self.num_unroll_steps = config.num_unroll_steps
        self.td_steps = config.td_steps
        self.frame_stack_num = config.model.frame_stack_num
        self.discount_factor = config.discount_factor
        self.action_space_size = config.model.action_space_size
        self.gray_scale = config.gray_scale
        self.transform2string = config.transform2string
        self.sampled_algo = config.sampled_algo
        self.gumbel_algo = config.gumbel_algo
        self.use_ture_chance_label_in_chance_encoder = config.use_ture_chance_label_in_chance_encoder

        if isinstance(config.model.observation_shape, int) or len(config.model.observation_shape) == 1:
            # for vector obs input, e.g. classical control and box2d environments
            self.zero_obs_shape = config.model.observation_shape
        elif len(config.model.observation_shape) == 3:
            # image obs input, e.g. atari environments
            self.zero_obs_shape = (
                config.model.observation_shape[-2], config.model.observation_shape[-1], config.model.image_channel
            )

        self.obs_segment = []
        self.action_segment = []
        self.reward_segment = []

        self.child_visit_segment = []
        self.root_value_segment = []

        self.action_mask_segment = []
        self.to_play_segment = []

        self.target_values = []
        self.target_rewards = []
        self.target_policies = []

        self.improved_policy_probs = []

        if self.sampled_algo:
            self.root_sampled_actions = []
        if self.use_ture_chance_label_in_chance_encoder:
            self.chance_segment = []

    def get_unroll_obs(self, timestep: int, num_unroll_steps: int = 0, padding: bool = False) -> np.ndarray:
        """
        Overview:
            Get an observation of the correct format: o[t, t + stack frames + num_unroll_steps].
        Arguments:
            - timestep (int): The time step.
            - num_unroll_steps (int): The extra length of the observation frames.
            - padding (bool): If True, pad frames if (t + stack frames) is outside of the trajectory.
        """
        stacked_obs = self.obs_segment[timestep:timestep + self.frame_stack_num + num_unroll_steps]
        if padding:
            pad_len = self.frame_stack_num + num_unroll_steps - len(stacked_obs)
            if pad_len > 0:
                pad_frames = np.array([stacked_obs[-1] for _ in range(pad_len)])
                stacked_obs = np.concatenate((stacked_obs, pad_frames))
        if self.transform2string:
            stacked_obs = [jpeg_data_decompressor(obs, self.gray_scale) for obs in stacked_obs]
        return stacked_obs

    def zero_obs(self) -> List:
        """
        Overview:
            Return an observation frame filled with zeros.
        Returns:
            ndarray: An array filled with zeros.
        """
        return [np.zeros(self.zero_obs_shape, dtype=np.float32) for _ in range(self.frame_stack_num)]

    def get_obs(self) -> List:
        """
        Overview:
            Return an observation in the correct format for model inference.
        Returns:
              stacked_obs (List): An observation in the correct format for model inference.
          """
        timestep_obs = len(self.obs_segment) - self.frame_stack_num
        timestep_reward = len(self.reward_segment)
        assert timestep_obs == timestep_reward, "timestep_obs: {}, timestep_reward: {}".format(
            timestep_obs, timestep_reward
        )
        timestep = timestep_reward
        stacked_obs = self.obs_segment[timestep:timestep + self.frame_stack_num]
        if self.transform2string:
            stacked_obs = [jpeg_data_decompressor(obs, self.gray_scale) for obs in stacked_obs]
        return stacked_obs

    def append(
            self,
            action: np.ndarray,
            obs: np.ndarray,
            reward: np.ndarray,
            action_mask: np.ndarray = None,
            to_play: int = -1,
            chance: int = 0,
    ) -> None:
        """
        Overview:
            Append a transition tuple, including a_t, o_{t+1}, r_{t}, action_mask_{t}, to_play_{t}.
        """
        self.action_segment.append(action)
        self.obs_segment.append(obs)
        self.reward_segment.append(reward)

        self.action_mask_segment.append(action_mask)
        self.to_play_segment.append(to_play)
        if self.use_ture_chance_label_in_chance_encoder:
            self.chance_segment.append(chance)

    def pad_over(
            self, next_segment_observations: List, next_segment_rewards: List, next_segment_root_values: List,
            next_segment_child_visits: List, next_segment_improved_policy: List = None, next_chances: List = None,
    ) -> None:
        """
        Overview:
            To make sure the correction of value targets, we need to add (o_t, r_t, etc) from the next game_segment
            , which is necessary for the bootstrapped values at the end states of previous game_segment.
            e.g: len = 100; target value v_100 = r_100 + gamma^1 r_101 + ... + gamma^4 r_104 + gamma^5 v_105,
            but r_101, r_102, ... are from the next game_segment.
        Arguments:
            - next_segment_observations (:obj:`list`): o_t from the next game_segment
            - next_segment_rewards (:obj:`list`): r_t from the next game_segment
            - next_segment_root_values (:obj:`list`): root values of MCTS from the next game_segment
            - next_segment_child_visits (:obj:`list`): root visit count distributions of MCTS from the next game_segment
            - next_segment_improved_policy (:obj:`list`): root children select policy of MCTS from the next game_segment (Only used in Gumbel MuZero)
        """
        assert len(next_segment_observations) <= self.num_unroll_steps
        assert len(next_segment_child_visits) <= self.num_unroll_steps
        assert len(next_segment_root_values) <= self.num_unroll_steps + self.td_steps
        assert len(next_segment_rewards) <= self.num_unroll_steps + self.td_steps - 1
        # ==============================================================
        # The core difference between GumbelMuZero and MuZero
        # ==============================================================
        if self.gumbel_algo:
            assert len(next_segment_improved_policy) <= self.num_unroll_steps + self.td_steps

        # NOTE: next block observation should start from (stacked_observation - 1) in next trajectory
        for observation in next_segment_observations:
            self.obs_segment.append(copy.deepcopy(observation))

        for reward in next_segment_rewards:
            self.reward_segment.append(reward)

        for value in next_segment_root_values:
            self.root_value_segment.append(value)

        for child_visits in next_segment_child_visits:
            self.child_visit_segment.append(child_visits)
        
        if self.gumbel_algo:
            for improved_policy in next_segment_improved_policy:
                self.improved_policy_probs.append(improved_policy)
        if self.use_ture_chance_label_in_chance_encoder:
            for chances in next_chances:
                self.chance_segment.append(chances)

    def get_targets(self, timestep: int) -> Tuple:
        """
        Overview:
            return the value/reward/policy targets at step timestep
        """
        return self.target_values[timestep], self.target_rewards[timestep], self.target_policies[timestep]

    def store_search_stats(
            self, visit_counts: List, root_value: List, root_sampled_actions: List = None, improved_policy: List = None, idx: int = None
    ) -> None:
        """
        Overview:
            store the visit count distributions and value of the root node after MCTS.
        """
        sum_visits = sum(visit_counts)
        if idx is None:
            self.child_visit_segment.append([visit_count / sum_visits for visit_count in visit_counts])
            self.root_value_segment.append(root_value)
            if self.sampled_algo:
                self.root_sampled_actions.append(root_sampled_actions)
            # store the improved policy in Gumbel Muzero: \pi'=softmax(logits + \sigma(CompletedQ))
            if self.gumbel_algo:
                self.improved_policy_probs.append(improved_policy)
        else:
            self.child_visit_segment[idx] = [visit_count / sum_visits for visit_count in visit_counts]
            self.root_value_segment[idx] = root_value
            self.improved_policy_probs[idx] = improved_policy

    def game_segment_to_array(self) -> None:
        """
        Overview:
            Post-process the data when a `GameSegment` block is full. This function converts various game segment
            elements into numpy arrays for easier manipulation and processing.
        Structure:
            The structure and shapes of different game segment elements are as follows. Let's assume
            `game_segment_length`=20, `stack`=4, `num_unroll_steps`=5, `td_steps`=5:

            - obs:            game_segment_length + stack + num_unroll_steps, 20+4+5
            - action:         game_segment_length -> 20
            - reward:         game_segment_length + num_unroll_steps + td_steps -1  20+5+5-1
            - root_values:    game_segment_length + num_unroll_steps + td_steps -> 20+5+5
            - child_visits:   game_segment_length + num_unroll_steps -> 20+5
            - to_play:        game_segment_length -> 20
            - action_mask:    game_segment_length -> 20
        Examples:
            Here is an illustration of the structure of `obs` and `rew` for two consecutive game segments
            (game_segment_i and game_segment_i+1):

            - game_segment_i (obs):     4       20        5
                                      ----|----...----|-----|
            - game_segment_i+1 (obs):              4       20        5
                                                  ----|----...----|-----|

            - game_segment_i (rew):        20        5      4
                                      ----...----|------|-----|
            - game_segment_i+1 (rew):                 20        5    4
                                                 ----...----|------|-----|

        Postprocessing:
            - self.obs_segment (:obj:`numpy.ndarray`): A numpy array version of the original obs_segment.
            - self.action_segment (:obj:`numpy.ndarray`): A numpy array version of the original action_segment.
            - self.reward_segment (:obj:`numpy.ndarray`): A numpy array version of the original reward_segment.
            - self.child_visit_segment (:obj:`numpy.ndarray`): A numpy array version of the original child_visit_segment.
            - self.root_value_segment (:obj:`numpy.ndarray`): A numpy array version of the original root_value_segment.
            - self.improved_policy_probs (:obj:`numpy.ndarray`): A numpy array version of the original improved_policy_probs.
            - self.action_mask_segment (:obj:`numpy.ndarray`): A numpy array version of the original action_mask_segment.
            - self.to_play_segment (:obj:`numpy.ndarray`): A numpy array version of the original to_play_segment.
            - self.chance_segment (:obj:`numpy.ndarray`, optional): A numpy array version of the original chance_segment. Only
               created if `self.use_ture_chance_label_in_chance_encoder` is True.

        .. note::
            For environments with a variable action space, such as board games, the elements in `child_visit_segment` may have
            different lengths. In such scenarios, it is necessary to use the object data type for `self.child_visit_segment`.
        """
        self.obs_segment = np.array(self.obs_segment)
        self.action_segment = np.array(self.action_segment)
        self.reward_segment = np.array(self.reward_segment)

        # Check if all elements in self.child_visit_segment have the same length
        if all(len(x) == len(self.child_visit_segment[0]) for x in self.child_visit_segment):
            self.child_visit_segment = np.array(self.child_visit_segment)
        else:
            # In the case of environments with a variable action space, such as board games,
            # the elements in child_visit_segment may have different lengths.
            # In such scenarios, it is necessary to use the object data type.
            self.child_visit_segment = np.array(self.child_visit_segment, dtype=object)

        self.root_value_segment = np.array(self.root_value_segment)
        self.improved_policy_probs = np.array(self.improved_policy_probs)

        self.action_mask_segment = np.array(self.action_mask_segment)
        self.to_play_segment = np.array(self.to_play_segment)
        if self.use_ture_chance_label_in_chance_encoder:
            self.chance_segment = np.array(self.chance_segment)

    def reset(self, init_observations: np.ndarray) -> None:
        """
        Overview:
            Initialize the game segment using ``init_observations``,
            which is the previous ``frame_stack_num`` stacked frames.
        Arguments:
            - init_observations (:obj:`list`): list of the stack observations in the previous time steps.
        """
        self.obs_segment = []
        self.action_segment = []
        self.reward_segment = []

        self.child_visit_segment = []
        self.root_value_segment = []

        self.action_mask_segment = []
        self.to_play_segment = []
        if self.use_ture_chance_label_in_chance_encoder:
            self.chance_segment = []

        assert len(init_observations) == self.frame_stack_num

        for observation in init_observations:
            self.obs_segment.append(copy.deepcopy(observation))

    def is_full(self) -> bool:
        """
        Overview:
            Check whether the current game segment is full, i.e. larger than the segment length.
        Returns:
            bool: True if the game segment is full, False otherwise.
        """
        return len(self.action_segment) >= self.game_segment_length

    def legal_actions(self):
        return [_ for _ in range(self.action_space.n)]

    def __len__(self):
        return len(self.action_segment)
