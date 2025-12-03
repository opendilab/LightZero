import numpy as np
from typing import Optional, List, Any
from lzero.mcts.buffer.game_segment import GameSegment as OriginalGameSegment


class GameSegment(OriginalGameSegment):
    """
    [PRIORZERO-MODIFIED]
    Enhanced GameSegment that stores additional data for PriorZero training.

    New attributes:
        - mcts_policy_segment: List of MCTS visit count distributions (for SFT)
        - raw_obs_segment: List of raw text observations (for LLM prompts)
        - llm_prior_segment: List of LLM generated text (for debugging)
        - search_value_segment: List of MCTS search values (for priority)
    """

    def __init__(
        self,
        action_space,
        game_segment_length: int = 200,
        config: Optional[Any] = None,
        task_id: Optional[int] = None
    ):
        """
        Initialize enhanced GameSegment.

        Args:
            action_space: Action space from environment
            game_segment_length: Maximum length of the segment
            config: Policy configuration
            task_id: Task ID for multi-task learning
        """
        super().__init__(action_space, game_segment_length, config, task_id)

        self.raw_obs_segment = []          # Raw text observations
        self.history_obs_segment = []
        self.action_logprob_segment = []   # Logprob of chosen action (for PPO/RFT)

    def reset(self, init_observations: List[np.ndarray], init_raw_obs, init_history_obs, init_action_logprob) -> None:
        """
        [PRIORZERO-MODIFIED]
        Reset the segment with initial observations.

        Args:
            init_observations: List of initial frame stack observations
        """
        super().reset(init_observations)
        self.raw_obs_segment.clear()
        self.history_obs_segment.clear()
        self.action_logprob_segment.clear()
        
        self.raw_obs_segment.append(init_raw_obs) 
        self.history_obs_segment.append(init_history_obs)
        self.action_logprob_segment.append(init_action_logprob)  

    def append(
        self,
        action: int,
        obs: np.ndarray,
        reward: float,
        action_mask: np.ndarray,
        to_play: int,
        timestep: int = 0,
        chance: int = 0,
        raw_obs_text: Optional[str] = None,
        history_obs: Optional[List[str]] = None,
        action_logprob: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        [PRIORZERO-MODIFIED]
        Append a new transition to the segment.

        Args:
            action: Action taken
            obs: Observation received
            reward: Reward received
            action_mask: Valid action mask
            to_play: Player ID (for multi-agent)
            **kwargs: Additional arguments (timestep, chance, raw_obs_text, llm_prior_text)
        """
        # Call parent append with remaining kwargs
        super().append(action, obs, reward, action_mask, to_play, timestep, chance)
        self.raw_obs_segment.append(raw_obs_text)
        self.history_obs_segment.append(history_obs)
        self.action_logprob_segment.append(action_logprob)

    def store_search_stats(self, visit_counts: List, root_value: List) -> None:
        """
        [PRIORZERO-MODIFIED]
        Store MCTS search statistics.

        This method is called after MCTS search to store the visit count
        distribution and search value. These will be used for:
        - SFT training: MCTS policy as supervision signal for LLM
        - Priority calculation: Search value for prioritized replay

        Args:
            root_visit_dist: Visit count distribution from MCTS
            value: Search value from MCTS
            *args: Additional positional arguments (for compatibility)
            **kwargs: Additional keyword arguments (improved_policy, etc.)
        """
        super().store_search_stats(visit_counts, root_value)

    def game_segment_to_array(self) -> None:
        """
        [PRIORZERO-MODIFIED]
        Convert all segment lists to numpy arrays for efficient storage.

        This is called when the segment is full and ready to be stored in
        the replay buffer.
        """
        # Call parent method to convert standard segments
        super().game_segment_to_array()
        self.action_logprob_segment = np.asarray(self.action_logprob_segment)
    
    def pad_over(
            self, next_segment_observations: List, next_segment_rewards: List, next_segment_actions: List, next_segment_root_values: List,
            next_segment_child_visits: List, next_segment_improved_policy: List = None, next_chances: List = None, 
            next_segment_raw_obs: List = None, next_segment_history_obs: List = None, next_segment_action_logprob: List = None
    ) -> None:
        super().pad_over(
            next_segment_observations, next_segment_rewards, next_segment_actions, next_segment_root_values,
            next_segment_child_visits, next_segment_improved_policy, next_chances
        )
        assert len(next_segment_raw_obs) <= self.num_unroll_steps + self.td_steps
        assert len(next_segment_history_obs) <= self.num_unroll_steps + self.td_steps
        assert len(next_segment_action_logprob) <= self.num_unroll_steps + self.td_steps
        import copy
        for raw_obs in next_segment_raw_obs:
            self.raw_obs_segment.append(copy.deepcopy(raw_obs))
        for history_obs in next_segment_history_obs:
            self.history_obs_segment.append(copy.deepcopy(history_obs))
        for lp in next_segment_action_logprob:
            self.action_logprob_segment.append(copy.deepcopy(lp))

    def get_unroll_raw_obs(self, timestep: int, num_unroll_steps: int = 0, padding: bool = False) -> np.ndarray:
        """
        Overview:
            Get an observation of the correct format: o[t, t + stack frames + num_unroll_steps].
        Arguments:
            - timestep (int): The time step.
            - num_unroll_steps (int): The extra length of the observation frames.
            - padding (bool): If True, pad frames if (t + stack frames) is outside of the trajectory.
        """
        stacked_raw_obs = self.raw_obs_segment[timestep:timestep + self.frame_stack_num + num_unroll_steps]
        if padding:
            pad_len = self.frame_stack_num + num_unroll_steps - len(stacked_raw_obs)
            if pad_len > 0:
                pad_frames = np.array([stacked_raw_obs[-1] for _ in range(pad_len)])
                stacked_raw_obs = np.concatenate((stacked_raw_obs, pad_frames))
        return stacked_raw_obs

    def get_unroll_histroy_obs(self, timestep: int, num_unroll_steps: int = 0, padding: bool = False) -> np.ndarray:
        """
        Overview:
            Get an observation of the correct format: o[t, t + stack frames + num_unroll_steps].
        Arguments:
            - timestep (int): The time step.
            - num_unroll_steps (int): The extra length of the observation frames.
            - padding (bool): If True, pad frames if (t + stack frames) is outside of the trajectory.
        """
        stacked_histroy_obs = self.history_obs_segment[timestep:timestep + self.frame_stack_num + num_unroll_steps]
        if padding:
            pad_len = self.frame_stack_num + num_unroll_steps - len(stacked_histroy_obs)
            if pad_len > 0:
                pad_frames = np.array([stacked_histroy_obs[-1] for _ in range(pad_len)])
                stacked_histroy_obs = np.concatenate((stacked_histroy_obs, pad_frames))
        return stacked_histroy_obs

    def get_unroll_action_logprob(self, timestep: int, num_unroll_steps: int = 0, padding: bool = False) -> np.ndarray:
        """
        Return action logprobs aligned with actions for unroll window.
        """
        stacked_logprob = list(self.action_logprob_segment[timestep:timestep + self.frame_stack_num + num_unroll_steps])
        if padding:
            pad_len = self.frame_stack_num + num_unroll_steps - len(stacked_logprob)
            if pad_len > 0:
                pad_frames = np.array([stacked_logprob[-1] for _ in range(pad_len)])
                stacked_logprob = np.concatenate((stacked_logprob, pad_frames))
        return stacked_logprob

# ==============================================================================
# Utility Functions
# ==============================================================================
