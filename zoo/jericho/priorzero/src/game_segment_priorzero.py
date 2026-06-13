import numpy as np
from typing import Optional, List, Any
from lzero.mcts.buffer.game_segment import GameSegment as OriginalGameSegment


class GameSegment(OriginalGameSegment):

    def __init__(
        self,
        action_space,
        game_segment_length: int = 200,
        config: Optional[Any] = None,
        task_id: Optional[int] = None
    ):
        super().__init__(action_space, game_segment_length, config, task_id)

        self.raw_obs_segment = []          # Raw text observations
        self.history_obs_segment = []
        self.llm_prior_per_tok_segment = []   # LLM prior per token (for debugging)
        self.cot_prefix_segment = []       # CoT prefixes for reuse (optimization)
        self.llm_action_segment = []       # Actions selected by LLM 

    def reset(self, init_observations: List[np.ndarray], init_raw_obs, init_history_obs) -> None:
        """
        [PRIORZERO-MODIFIED]
        Reset the segment with initial observations.

        Args:
            init_observations: List of initial frame stack observations
            init_raw_obs: Initial raw text observation
            init_history_obs: Initial history observations
        """
        super().reset(init_observations)
        self.raw_obs_segment.clear()
        self.history_obs_segment.clear()
        self.llm_prior_per_tok_segment.clear()
        self.cot_prefix_segment.clear()  # Clear CoT prefix segment
        self.llm_action_segment.clear()

        # 以下结果均是第 t 时刻的结果
        self.raw_obs_segment.append(init_raw_obs) 
        self.history_obs_segment.append(init_history_obs) 
        self.llm_prior_per_tok_segment.append(None)  
        self.cot_prefix_segment.append(None)        
        self.llm_action_segment.append(None)      

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
        llm_prior_per_tok = None,
        cot_prefix: Optional[str] = None,
        llm_action: Optional[str] = None,
        **kwargs
    ) -> None:
        
        super().append(action, obs, reward, action_mask, to_play, timestep, chance)
        self.raw_obs_segment.append(raw_obs_text)
        self.history_obs_segment.append(history_obs)
        self.llm_prior_per_tok_segment.append(llm_prior_per_tok)
        self.cot_prefix_segment.append(cot_prefix)
        self.llm_action_segment.append(llm_action)

    def store_search_stats(self, visit_counts: List, root_value: List) -> None:
        super().store_search_stats(visit_counts, root_value)

    def game_segment_to_array(self) -> None:
        super().game_segment_to_array()
    
    def pad_over(
            self, next_segment_observations: List, next_segment_rewards: List, next_segment_actions: List, next_segment_root_values: List,
            next_segment_child_visits: List, next_segment_improved_policy: List = None, next_chances: List = None,
            next_segment_raw_obs: List = None, next_segment_history_obs: List = None, next_segment_llm_prior_per_tok: List = None,
            next_segment_cot_prefix: List = None, next_segment_llm_action: List = None
    ) -> None:
        super().pad_over(
            next_segment_observations, next_segment_rewards, next_segment_actions, next_segment_root_values,
            next_segment_child_visits, next_segment_improved_policy, next_chances
        )
        assert len(next_segment_raw_obs) <= self.num_unroll_steps + self.td_steps
        assert len(next_segment_history_obs) <= self.num_unroll_steps + self.td_steps
        assert len(next_segment_llm_prior_per_tok) <= self.num_unroll_steps + self.td_steps
        assert len(next_segment_cot_prefix) <= self.num_unroll_steps + self.td_steps
        assert len(next_segment_llm_action) <= self.num_unroll_steps + self.td_steps

        import copy
        if len(next_segment_history_obs) > 0:
            assert self.raw_obs_segment[-1] == next_segment_llm_prior_per_tok[0]['current_obs']
            assert self.history_obs_segment[-1] == next_segment_llm_prior_per_tok[0]['history']
            assert self.history_obs_segment[-1][-1][1] == self.llm_action_segment[-1]
            assert next_segment_history_obs[0][-1][1] == next_segment_llm_action[0]

        for raw_obs in next_segment_raw_obs:
            self.raw_obs_segment.append(copy.deepcopy(raw_obs))
        for history_obs in next_segment_history_obs:
            self.history_obs_segment.append(copy.deepcopy(history_obs))
        for lp in next_segment_llm_prior_per_tok:
            self.llm_prior_per_tok_segment.append(copy.deepcopy(lp))
        for action in next_segment_llm_action:
            self.llm_action_segment.append(copy.deepcopy(action))

        # Handle CoT prefix padding (optimization for CoT reuse)
        if next_segment_cot_prefix is not None:
            for cot_prefix in next_segment_cot_prefix:
                self.cot_prefix_segment.append(copy.deepcopy(cot_prefix))

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
                pad_frames = [stacked_raw_obs[-1] for _ in range(pad_len)]
                stacked_raw_obs = stacked_raw_obs + pad_frames
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
                pad_frames = [stacked_histroy_obs[-1] for _ in range(pad_len)]
                stacked_histroy_obs = stacked_histroy_obs + pad_frames
        return stacked_histroy_obs

    def get_unroll_llm_prior_per_tok(self, timestep: int, num_unroll_steps: int = 0, padding: bool = False) -> np.ndarray:
        """
        Return LLM prior per token aligned with actions for unroll window.
        """
        stacked_prior = list(self.llm_prior_per_tok_segment[timestep:timestep + self.frame_stack_num + num_unroll_steps])
        if padding:
            pad_len = self.frame_stack_num + num_unroll_steps - len(stacked_prior)
            if pad_len > 0:
                pad_frames = [stacked_prior[-1] for _ in range(pad_len)]
                stacked_prior = stacked_prior + pad_frames
        return stacked_prior

    def get_unroll_cot_prefix(self, timestep: int, num_unroll_steps: int = 0, padding: bool = False) -> List[str]:
        """
        Return CoT prefixes aligned with observations for unroll window (CoT reuse optimization).

        Args:
            timestep: The time step
            num_unroll_steps: The extra length of the CoT prefix frames
            padding: If True, pad frames if outside of trajectory

        Returns:
            List of CoT prefix strings
        """
        stacked_cot_prefix = list(self.cot_prefix_segment[timestep:timestep + self.frame_stack_num +num_unroll_steps])
        if padding:
            pad_len = self.frame_stack_num + num_unroll_steps - len(stacked_cot_prefix)
            if pad_len > 0:
                # Pad with empty strings or last prefix
                pad_frames = [stacked_cot_prefix[-1] for _ in range(pad_len)]
                stacked_cot_prefix = stacked_cot_prefix + pad_frames
        return stacked_cot_prefix

    def get_unroll_llm_action(self, timestep: int, num_unroll_steps: int = 0, padding: bool = False) -> List[str]:
        """
        Return LLM actions aligned with observations for unroll window.

        Args:
            timestep: The time step
            num_unroll_steps: The extra length of the CoT prefix frames
            padding: If True, pad frames if outside of trajectory

        Returns:
            List of LLM action strings
        """
        stacked_llm_action = list(self.llm_action_segment[timestep:timestep + self.frame_stack_num + num_unroll_steps])
        if padding:
            pad_len = self.frame_stack_num + num_unroll_steps - len(stacked_llm_action)
            if pad_len > 0:
                # Pad with empty strings or last action
                pad_frames = [stacked_llm_action[-1] for _ in range(pad_len)]
                stacked_llm_action = stacked_llm_action + pad_frames
        return stacked_llm_action