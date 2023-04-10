import copy

import numpy as np
from ding.utils.compression_helper import jpeg_data_decompressor




class GameSegment:
    """
        Overview:
            A segment of game segment from a full episode trajectories.
            The length of one episode in Atari games are quite large. Split the whole episode trajectory into several
            ``GameSegment`` segments.
        Interfaces:
            ``__init__``, ``__len__``,``init``, ``pad_over``, ``is_full``, ``legal_actions``, ``append``, ``obs``
            ``zero_obs``, ``step_obs``, ``get_targets``, ``game_segment_to_array``, ``store_search_stats``.
    """

    def __init__(self, action_space, game_segment_length=200, config=None):
        """
        Overview:
            Init the ``GameSegment`` according to the provided arguments.
        Arguments:
             action_space (:obj:`int`): action space
            - game_segment_length (:obj:`int`): the transition number of one ``GameSegment`` segment
        """
        self.action_space = action_space
        self.game_segment_length = game_segment_length
        self.config = config

        self.frame_stack_num = config.model.frame_stack_num
        self.discount_factor = config.discount_factor
        self.action_space_size = config.model.action_space_size
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

        if self.config.sampled_algo:
            self.root_sampled_actions = []

    def get_unroll_obs(self, index, num_unroll_steps=0, padding=False):
        """
        Overview:
            To obtain an observation of correct format: o[t, t + stack frames + extra len]
        Arguments:
            - index: int time step
            - num_unroll_steps: int extra len of the obs frames
            - padding: bool True -> padding frames if (t + stack frames) are out of trajectory
        """
        frames = self.obs_segment[index:index + self.frame_stack_num + num_unroll_steps]
        if padding:
            pad_len = self.frame_stack_num + num_unroll_steps - len(frames)
            if pad_len > 0:
                pad_frames = np.array([frames[-1] for _ in range(pad_len)])
                frames = np.concatenate((frames, pad_frames))
        if self.config.transform2string:
            frames = [jpeg_data_decompressor(obs, self.config.gray_scale) for obs in frames]
        return frames

    def zero_obs(self):
        """
        Overview:
            return a zero obs frame
        """
        return [np.zeros(self.zero_obs_shape, dtype=np.float32) for _ in range(self.frame_stack_num)]

    def get_obs(self):
        """
        Overview:
            return an observation in correct format for model inference
        """
        index = len(self.reward_segment)
        frames = self.obs_segment[index:index + self.frame_stack_num]
        if self.config.transform2string:
            frames = [jpeg_data_decompressor(obs, self.config.gray_scale) for obs in frames]
        return frames

    def append(self, action, obs, reward, action_mask=None, to_play=-1):
        """
        Overview:
            append a transition tuple, including a_t, o_{t+1}, r_{t}, action_mask_{t}, to_play_{t}
        """
        self.action_segment.append(action)
        self.obs_segment.append(obs)
        self.reward_segment.append(reward)

        self.action_mask_segment.append(action_mask)
        self.to_play_segment.append(to_play)

    def pad_over(self, next_segment_observations, next_segment_rewards, next_segment_root_values, next_segment_child_visits):
        """
        Overview:
            To make sure the correction of value targets, we need to add (o_t, r_t, etc) from the next segment segment
            , which is necessary for the bootstrapped values at the end states of this segment segment.
            Eg: len = 100; target value v_100 = r_100 + gamma^1 r_101 + ... + gamma^4 r_104 + gamma^5 v_105,
            but r_101, r_102, ... are from the next segment segment.
        Arguments:
            - next_segment_observations (:obj:`list`):  list o_t from the next segment segment
            - next_segment_rewards (:obj:`list`): list r_t from the next segment segment
            - next_segment_root_values (:obj:`list`): list root values of MCTS from the next segment segment
            - next_segment_child_visits (:obj:`list`): list root visit count distributions of MCTS from
            the next segment segment
        """
        assert len(next_segment_observations) <= self.config.num_unroll_steps
        assert len(next_segment_child_visits) <= self.config.num_unroll_steps
        assert len(next_segment_root_values) <= self.config.num_unroll_steps + self.config.td_steps
        assert len(next_segment_rewards) <= self.config.num_unroll_steps + self.config.td_steps - 1

        # NOTE: next segment observation should start from (stacked_observation - 1) in next trajectory
        for observation in next_segment_observations:
            self.obs_segment.append(copy.deepcopy(observation))

        for reward in next_segment_rewards:
            self.reward_segment.append(reward)

        for value in next_segment_root_values:
            self.root_value_segment.append(value)

        for child_visits in next_segment_child_visits:
            self.child_visit_segment.append(child_visits)




    def get_targets(self, i):
        """
        Overview:
            return the value/reward/policy targets at step i
        """
        return self.target_values[i], self.target_rewards[i], self.target_policies[i]

    def store_search_stats(self, visit_counts, root_value, root_sampled_actions=None, idx: int = None):
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
        else:
            self.child_visit_segment[idx] = [visit_count / sum_visits for visit_count in visit_counts]
            self.root_value_segment[idx] = root_value

    def game_segment_to_array(self):
        """
        Overview:
            post processing the data when a ``GameSegment`` segment is full.
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

    def reset(self, init_observations):
        """
        Overview:
            Initialize the game segment segment using ``init_observations``,
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

    def is_full(self):
        """
        Overview:
            check whether current game segment segment is full, i.e. larger than self.game_segment_length
        """
        return len(self.action_segment) >= self.game_segment_length

    def legal_actions(self):
        return [_ for _ in range(self.action_space.n)]

    def __len__(self):
        return len(self.action_segment)