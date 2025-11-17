# game_segment_priorzero.py
"""
[PRIORZERO] Enhanced Game Segment for PriorZero

This module extends the standard GameSegment to store additional information
needed for LLM policy training (SFT + RFT).

Key Features:
- Store MCTS policy distributions for SFT training
- Store raw text observations for LLM prompt construction
- Store LLM generated priors for analysis and debugging
- Store search values for priority calculation

Author: PriorZero Team
Date: 2025-01-20
"""

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

        # [PRIORZERO-NEW] Additional segments for LLM training
        self.mcts_policy_segment = []      # MCTS visit count distributions
        self.raw_obs_segment = []          # Raw text observations
        self.llm_prior_segment = []        # LLM generated priors (for debugging)
        self.search_value_segment = []     # MCTS search values

    def reset(self, init_observations: List[np.ndarray]) -> None:
        """
        [PRIORZERO-MODIFIED]
        Reset the segment with initial observations.

        Args:
            init_observations: List of initial frame stack observations
        """
        super().reset(init_observations)

        # Clear PriorZero-specific segments
        self.mcts_policy_segment.clear()
        self.raw_obs_segment.clear()
        self.llm_prior_segment.clear()
        self.search_value_segment.clear()

    def append(
        self,
        action: int,
        obs: np.ndarray,
        reward: float,
        action_mask: np.ndarray,
        to_play: int,
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
        # [PRIORZERO-NEW] Extract PriorZero-specific kwargs before passing to parent
        raw_obs_text = kwargs.pop('raw_obs_text', None)
        llm_prior_text = kwargs.pop('llm_prior_text', None)

        # [DEBUG] Log first few appends to see what's being passed
        if len(self.raw_obs_segment) < 3:
            print(f"[SEGMENT_DEBUG] append() called: kwargs keys = {list(kwargs.keys())}")
            print(f"[SEGMENT_DEBUG] raw_obs_text = {raw_obs_text[:50] if raw_obs_text else 'None'}...")

        # Call parent append with remaining kwargs
        super().append(action, obs, reward, action_mask, to_play, **kwargs)

        # [PRIORZERO-NEW] Initialize placeholders for new segments
        # These will be filled in by store_search_stats()
        self.mcts_policy_segment.append(None)
        self.search_value_segment.append(None)

        # [PRIORZERO-NEW] Store raw text observation if provided
        self.raw_obs_segment.append(raw_obs_text)

        # [PRIORZERO-NEW] Store LLM prior text if provided (for debugging)
        self.llm_prior_segment.append(llm_prior_text)

    def store_search_stats(
        self,
        root_visit_dist: List[float],
        value: float,
        *args,
        **kwargs
    ) -> None:
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
        # [FIX] Handle NaN values
        import numpy as np
        if value is None or (isinstance(value, float) and np.isnan(value)):
            # Use 0.0 as default for NaN values
            value = 0.0

        # Call parent method to store standard statistics
        super().store_search_stats(root_visit_dist, value, *args, **kwargs)

        # [PRIORZERO-NEW] Store MCTS policy distribution
        # Convert to numpy array and normalize to probability distribution
        policy_array = np.array(root_visit_dist, dtype=np.float32)

        if policy_array.sum() > 0:
            policy_array = policy_array / policy_array.sum()
        else:
            # If no visits (shouldn't happen), use uniform distribution
            policy_array = np.ones_like(policy_array) / len(policy_array)

        # Update the most recent position (corresponding to last append)
        if len(self.mcts_policy_segment) > 0:
            self.mcts_policy_segment[-1] = policy_array

        # [PRIORZERO-NEW] Store search value
        if len(self.search_value_segment) > 0:
            self.search_value_segment[-1] = float(value)

    def game_segment_to_array(self) -> None:
        """
        [PRIORZERO-MODIFIED]
        Convert all segment lists to numpy arrays for efficient storage.

        This is called when the segment is full and ready to be stored in
        the replay buffer.
        """
        # Call parent method to convert standard segments
        super().game_segment_to_array()

        # [PRIORZERO-NEW] Convert PriorZero-specific segments to arrays
        # Use object dtype to handle variable-length arrays and None values
        self.mcts_policy_segment = np.array(self.mcts_policy_segment, dtype=object)
        self.search_value_segment = np.array(self.search_value_segment, dtype=np.float32)

        # For text data, keep as list (more flexible for variable-length strings)
        # self.raw_obs_segment and self.llm_prior_segment remain as lists

    def get_stats(self) -> dict:
        """
        [PRIORZERO-NEW]
        Get statistics about this game segment.

        Returns:
            stats: Dictionary of statistics
        """
        stats = {
            'segment_length': len(self.reward_segment) if hasattr(self, 'reward_segment') else 0,
            'total_reward': sum(self.reward_segment) if hasattr(self, 'reward_segment') else 0,
            'num_mcts_policies': sum(1 for p in self.mcts_policy_segment if p is not None),
            'num_raw_obs': sum(1 for o in self.raw_obs_segment if o is not None),
            'num_llm_priors': sum(1 for p in self.llm_prior_segment if p is not None),
            'avg_search_value': np.mean([v for v in self.search_value_segment if v is not None]) if any(v is not None for v in self.search_value_segment) else 0.0,
        }
        return stats

    def get_mcts_policy_for_training(self, index: int) -> Optional[np.ndarray]:
        """
        [PRIORZERO-NEW]
        Get MCTS policy at a specific index for training.

        Args:
            index: Index in the segment

        Returns:
            policy: MCTS policy distribution, or None if not available
        """
        if 0 <= index < len(self.mcts_policy_segment):
            return self.mcts_policy_segment[index]
        return None

    def get_raw_obs_for_training(self, index: int) -> Optional[str]:
        """
        [PRIORZERO-NEW]
        Get raw text observation at a specific index for training.

        Args:
            index: Index in the segment

        Returns:
            raw_obs: Raw text observation, or None if not available
        """
        if 0 <= index < len(self.raw_obs_segment):
            return self.raw_obs_segment[index]
        return None

    def get_history_for_training(self, index: int, history_length: int = 5) -> List[tuple]:
        """
        [PRIORZERO-NEW]
        Get history context for LLM prompting.

        Args:
            index: Current index in the segment
            history_length: Number of past transitions to include

        Returns:
            history: List of (obs, action, reward) tuples
        """
        history = []

        # Get recent transitions
        start_idx = max(0, index - history_length)
        for i in range(start_idx, index):
            if i < len(self.raw_obs_segment) and i < len(self.action_segment) and i < len(self.reward_segment):
                obs_text = self.raw_obs_segment[i]
                action_id = self.action_segment[i]
                reward = self.reward_segment[i]

                # Only add if observation is available
                if obs_text is not None:
                    history.append((obs_text, action_id, reward))

        return history

    def __repr__(self) -> str:
        """
        [PRIORZERO-MODIFIED]
        String representation with PriorZero statistics.
        """
        base_repr = super().__repr__()
        stats = self.get_stats()

        priorzero_info = (
            f"\n  MCTS policies: {stats['num_mcts_policies']}"
            f"\n  Raw observations: {stats['num_raw_obs']}"
            f"\n  LLM priors: {stats['num_llm_priors']}"
            f"\n  Avg search value: {stats['avg_search_value']:.3f}"
        )

        return base_repr + priorzero_info


# ==============================================================================
# Utility Functions
# ==============================================================================

def create_priorzero_game_segment(
    action_space,
    game_segment_length: int = 200,
    config: Optional[Any] = None,
    task_id: Optional[int] = None
) -> GameSegment:
    """
    Factory function to create a PriorZero GameSegment.

    Args:
        action_space: Action space from environment
        game_segment_length: Maximum length of the segment
        config: Policy configuration
        task_id: Task ID for multi-task learning

    Returns:
        segment: PriorZero GameSegment instance
    """
    return GameSegment(action_space, game_segment_length, config, task_id)


def validate_game_segment(segment: GameSegment) -> bool:
    """
    Validate that a GameSegment has consistent data.

    Args:
        segment: GameSegment to validate

    Returns:
        is_valid: True if segment is valid, False otherwise
    """
    try:
        # Check basic lengths
        if not hasattr(segment, 'obs_segment'):
            return False

        base_length = len(segment.obs_segment)

        # Check that all segments have compatible lengths
        if hasattr(segment, 'action_segment'):
            if len(segment.action_segment) != base_length:
                return False

        if hasattr(segment, 'reward_segment'):
            if len(segment.reward_segment) != base_length:
                return False

        # Check PriorZero-specific segments
        if len(segment.mcts_policy_segment) != base_length:
            return False

        if len(segment.raw_obs_segment) != base_length:
            return False

        # Check that MCTS policies are valid when present
        for policy in segment.mcts_policy_segment:
            if policy is not None:
                if not isinstance(policy, np.ndarray):
                    return False
                if policy.sum() < 0.99 or policy.sum() > 1.01:  # Should sum to ~1.0
                    return False
                if np.any(policy < 0):  # Should be non-negative
                    return False

        return True

    except Exception as e:
        print(f"Validation error: {e}")
        return False


# ==============================================================================
# Example Usage and Testing
# ==============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Testing PriorZero GameSegment")
    print("="*80)

    # Create a mock action space
    class MockActionSpace:
        def __init__(self, n):
            self.n = n

    # Create a mock config with all required attributes
    class MockConfig:
        def __init__(self):
            self.num_unroll_steps = 10
            self.td_steps = 5
            self.discount_factor = 0.99
            self.gray_scale = False
            self.transform2string = False
            self.sampled_algo = False
            self.gumbel_algo = False
            self.use_ture_chance_label_in_chance_encoder = False
            self.model = type('obj', (object,), {
                'frame_stack_num': 4,
                'action_space_size': 10,
                'observation_shape': (84, 84, 3),
                'image_channel': 3
            })()

    action_space = MockActionSpace(n=10)
    mock_config = MockConfig()

    # Create a game segment
    segment = GameSegment(action_space, game_segment_length=100, config=mock_config)

    # Reset with initial observations
    init_obs = [np.zeros((84, 84, 3)) for _ in range(4)]
    segment.reset(init_obs)

    print("\n1. Empty segment:")
    print(f"  Length: {len(segment.obs_segment)}")
    print(f"  MCTS policies: {len(segment.mcts_policy_segment)}")

    # Simulate some transitions
    print("\n2. Adding transitions...")
    for i in range(5):
        obs = np.random.rand(84, 84, 3)
        action = np.random.randint(0, 10)
        reward = np.random.randn()
        action_mask = np.ones(10)

        # Append transition
        segment.append(
            action, obs, reward, action_mask, to_play=0,
            raw_obs_text=f"You see a room. Step {i}.",
            llm_prior_text=f"Top actions: go north, take key"
        )

        # Store MCTS stats
        visit_dist = np.random.dirichlet([1.0] * 10).tolist()
        value = np.random.randn()
        segment.store_search_stats(visit_dist, value)

    print(f"  Added {len(segment.obs_segment)} transitions")

    # Get statistics
    print("\n3. Segment statistics:")
    stats = segment.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test retrieval functions
    print("\n4. Testing retrieval functions:")
    mcts_policy = segment.get_mcts_policy_for_training(2)
    print(f"  MCTS policy at index 2: {mcts_policy is not None}")
    if mcts_policy is not None:
        print(f"    Shape: {mcts_policy.shape}")
        print(f"    Sum: {mcts_policy.sum():.3f}")

    raw_obs = segment.get_raw_obs_for_training(2)
    print(f"  Raw obs at index 2: {raw_obs}")

    history = segment.get_history_for_training(4, history_length=3)
    print(f"  History for index 4: {len(history)} transitions")

    # Validate segment
    print("\n5. Validating segment:")
    is_valid = validate_game_segment(segment)
    print(f"  Is valid: {is_valid}")

    # Convert to array
    print("\n6. Converting to array:")
    segment.game_segment_to_array()
    print(f"  MCTS policy type: {type(segment.mcts_policy_segment)}")
    print(f"  Search value type: {type(segment.search_value_segment)}")

    # Print representation
    print("\n7. Segment representation:")
    print(segment)

    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)
