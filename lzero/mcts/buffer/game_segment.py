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
        self.config = config

        self.frame_stack_num = config.model.frame_stack_num
        self.discount_factor = config.discount_factor
        self.action_space_size = config.model.action_space_size
        if isinstance(config.model.observation_shape, int) or len(config.model.observation_shape) == 1:
            # for vector obs input, e.g. classical control ad box2d environments
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

        if self.config.sampled_algo:
            self.root_sampled_actions = []

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
        if self.config.transform2string:
            stacked_obs = [jpeg_data_decompressor(obs, self.config.gray_scale) for obs in stacked_obs]
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
        # TODO:
        timestep = timestep_obs
        timestep = timestep_reward
        stacked_obs = self.obs_segment[timestep:timestep + self.frame_stack_num]
        if self.config.transform2string:
            stacked_obs = [jpeg_data_decompressor(obs, self.config.gray_scale) for obs in stacked_obs]
        return stacked_obs

    def append(
            self,
            action: np.ndarray,
            obs: np.ndarray,
            reward: np.ndarray,
            action_mask: np.ndarray = None,
            to_play: int = -1
    ) -> None:
        """
        Overview:
            append a transition tuple, including a_t, o_{t+1}, r_{t}, action_mask_{t}, to_play_{t}
        """
        self.action_segment.append(action)
        self.obs_segment.append(obs)
        self.reward_segment.append(reward)

        self.action_mask_segment.append(action_mask)
        self.to_play_segment.append(to_play)

    def pad_over(
            self, next_segment_observations: List, next_segment_rewards: List, next_segment_root_values: List,
            next_segment_child_visits: List
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
        """
        assert len(next_segment_observations) <= self.config.num_unroll_steps
        assert len(next_segment_child_visits) <= self.config.num_unroll_steps
        assert len(next_segment_root_values) <= self.config.num_unroll_steps + self.config.td_steps
        assert len(next_segment_rewards) <= self.config.num_unroll_steps + self.config.td_steps - 1

        # NOTE: next block observation should start from (stacked_observation - 1) in next trajectory
        for observation in next_segment_observations:
            self.obs_segment.append(copy.deepcopy(observation))

        for reward in next_segment_rewards:
            self.reward_segment.append(reward)

        for value in next_segment_root_values:
            self.root_value_segment.append(value)

        for child_visits in next_segment_child_visits:
            self.child_visit_segment.append(child_visits)

    def get_targets(self, timestep: int) -> Tuple:
        """
        Overview:
            return the value/reward/policy targets at step timestep
        """
        return self.target_values[timestep], self.target_rewards[timestep], self.target_policies[timestep]

    def store_search_stats(
            self, visit_counts: List, root_value: List, root_sampled_actions=None, improved_policy=None, idx: int = None
    ) -> None:
        """
        Overview:
            store the visit count distributions and value of the root node after MCTS.
        """
        sum_visits = sum(visit_counts)
        if idx is None:
            self.child_visit_segment.append([visit_count / sum_visits for visit_count in visit_counts])
            self.root_value_segment.append(root_value)
            if self.config.sampled_algo:
                self.root_sampled_actions.append(root_sampled_actions)
            # store the improved policy in Gumbel Muzero: \pi'=softmax(logits + \sigma(CompletedQ))
            if improved_policy.all():
                self.improved_policy_probs.append(improved_policy)
        else:
            self.child_visit_segment[idx] = [visit_count / sum_visits for visit_count in visit_counts]
            self.root_value_segment[idx] = root_value

    def game_segment_to_array(self) -> None:
        """
        Overview:
            post processing the data when a ``GameSegment`` block is full.
        Note:
        game_segment element shape:
            e.g. game_segment_length=20, stack=4, num_unroll_steps=5, td_steps=5

            obs:            game_segment_length + stack + num_unroll_steps, 20+4+5
            action:         game_segment_length -> 20
            reward:         game_segment_length + num_unroll_steps + td_steps -1  20+5+5-1
            root_values:    game_segment_length + num_unroll_steps + td_steps -> 20+5+5
            child_visits：  game_segment_length + num_unroll_steps -> 20+5
            to_play:        game_segment_length -> 20
            action_mask:    game_segment_length -> 20

        game_segment_t:
            obs:  4       20        5
                 ----|----...----|-----|
        game_segment_t+1:
            obs:               4       20        5
                             ----|----...----|-----|

        game_segment_t:
            rew:     20        5      4
                 ----...----|------|-----|
        game_segment_t+1:
            rew:             20        5    4
                        ----...----|------|-----|
        """
        self.obs_segment = np.array(self.obs_segment)
        self.action_segment = np.array(self.action_segment)
        self.reward_segment = np.array(self.reward_segment)

        self.child_visit_segment = np.array(self.child_visit_segment)
        self.root_value_segment = np.array(self.root_value_segment)

        self.action_mask_segment = np.array(self.action_mask_segment)
        self.to_play_segment = np.array(self.to_play_segment)

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
