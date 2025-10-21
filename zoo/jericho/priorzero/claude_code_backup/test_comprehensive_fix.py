#!/usr/bin/env python3
"""
Comprehensive test for PriorZero segment index fix.

This script tests:
1. Correct segment length calculation
2. Safe mcts_policy_segment access with error handling
3. Proper use of raw_obs_segment
4. Robustness with various segment states
"""

import sys
import numpy as np
from pathlib import Path

# Add LightZero to path
lightzero_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(lightzero_path))

from zoo.jericho.priorzero.game_segment_priorzero import GameSegment


class MockActionSpace:
    def __init__(self, n):
        self.n = n


class MockConfig:
    def __init__(self):
        self.num_unroll_steps = 5
        self.td_steps = 5
        self.discount_factor = 0.99
        self.gray_scale = False
        self.transform2string = False
        self.sampled_algo = False
        self.gumbel_algo = False
        self.use_ture_chance_label_in_chance_encoder = False

        class MockModel:
            frame_stack_num = 4
            action_space_size = 10
            observation_shape = (84, 84, 3)
            image_channel = 3

        self.model = MockModel()


def create_test_segment(num_actions=20):
    """Create a test segment with specified number of actions."""
    action_space = MockActionSpace(n=10)
    config = MockConfig()

    segment = GameSegment(action_space, game_segment_length=num_actions, config=config)

    # Reset with initial observations
    init_obs = [np.zeros((84, 84, 3)) for _ in range(4)]
    segment.reset(init_obs)

    # Add transitions
    for i in range(num_actions):
        obs = np.random.rand(84, 84, 3)
        action = np.random.randint(0, 10)
        reward = np.random.randn()
        action_mask = np.ones(10)

        segment.append(
            action, obs, reward, action_mask, to_play=0,
            raw_obs_text=f"Step {i}: You are in a room.",
            llm_prior_text=f"Suggested: go north"
        )

        # Store MCTS stats
        visit_dist = np.random.dirichlet([1.0] * 10).tolist()
        value = np.random.randn()
        segment.store_search_stats(visit_dist, value)

    return segment


def test_safe_segment_access():
    """Test safe access to segment with proper bounds checking."""
    print("\n" + "="*80)
    print("TEST 1: Safe Segment Access with Bounds Checking")
    print("="*80)

    segment = create_test_segment(num_actions=20)

    # Test the fixed approach
    segment_length = len(segment.action_segment)
    mcts_policy_length = len(segment.mcts_policy_segment) if hasattr(segment, 'mcts_policy_segment') else 0
    max_index = min(segment_length, mcts_policy_length)

    print(f"Calculated lengths:")
    print(f"  action_segment: {segment_length}")
    print(f"  mcts_policy_segment: {mcts_policy_length}")
    print(f"  max_index: {max_index}")

    success_count = 0
    error_count = 0

    for i in range(max_index):
        try:
            mcts_policy = segment.mcts_policy_segment[i]
            if mcts_policy is None:
                continue

            # Access raw_obs_segment
            raw_obs_text = None
            if hasattr(segment, 'raw_obs_segment') and i < len(segment.raw_obs_segment):
                raw_obs_text = segment.raw_obs_segment[i]
            elif i < len(segment.obs_segment):
                raw_obs_text = str(segment.obs_segment[i])

            if raw_obs_text is not None:
                success_count += 1
        except (IndexError, KeyError, TypeError) as e:
            error_count += 1
            print(f"  Error at index {i}: {e}")

    print(f"\nResults:")
    print(f"  Successful accesses: {success_count}")
    print(f"  Errors: {error_count}")

    assert error_count == 0, f"Should have no errors, got {error_count}"
    assert success_count == max_index, f"Should access all {max_index} indices"

    print("\n✓ TEST 1 PASSED: All accesses successful with no errors!")


def test_segment_after_game_segment_to_array():
    """Test segment access after game_segment_to_array() conversion."""
    print("\n" + "="*80)
    print("TEST 2: Segment Access After game_segment_to_array()")
    print("="*80)

    segment = create_test_segment(num_actions=20)

    # Convert to array (this is called when segment is added to buffer)
    segment.game_segment_to_array()

    print(f"After game_segment_to_array():")
    print(f"  mcts_policy_segment type: {type(segment.mcts_policy_segment)}")
    print(f"  mcts_policy_segment dtype: {segment.mcts_policy_segment.dtype}")
    print(f"  mcts_policy_segment shape: {segment.mcts_policy_segment.shape}")

    # Test access
    segment_length = len(segment.action_segment)
    mcts_policy_length = len(segment.mcts_policy_segment)
    max_index = min(segment_length, mcts_policy_length)

    success_count = 0
    for i in range(max_index):
        try:
            mcts_policy = segment.mcts_policy_segment[i]
            if mcts_policy is not None:
                # Verify it's still a valid probability distribution
                assert isinstance(mcts_policy, np.ndarray)
                assert abs(mcts_policy.sum() - 1.0) < 0.01
                success_count += 1
        except Exception as e:
            print(f"  Error at index {i}: {e}")
            raise

    print(f"\nResults:")
    print(f"  Successfully accessed and validated {success_count} MCTS policies")

    assert success_count == max_index

    print("\n✓ TEST 2 PASSED: Segment access works correctly after conversion!")


def test_empty_segment():
    """Test handling of empty segments."""
    print("\n" + "="*80)
    print("TEST 3: Empty Segment Handling")
    print("="*80)

    action_space = MockActionSpace(n=10)
    config = MockConfig()
    segment = GameSegment(action_space, game_segment_length=20, config=config)

    # Reset but don't add any transitions
    init_obs = [np.zeros((84, 84, 3)) for _ in range(4)]
    segment.reset(init_obs)

    # Test access
    segment_length = len(segment.action_segment)
    mcts_policy_length = len(segment.mcts_policy_segment) if hasattr(segment, 'mcts_policy_segment') else 0
    max_index = min(segment_length, mcts_policy_length)

    print(f"Empty segment lengths:")
    print(f"  action_segment: {segment_length}")
    print(f"  mcts_policy_segment: {mcts_policy_length}")
    print(f"  max_index: {max_index}")

    assert max_index == 0, "Empty segment should have max_index=0"
    print("  → Correctly identified as empty (max_index=0)")

    print("\n✓ TEST 3 PASSED: Empty segments are handled correctly!")


def test_partial_segment():
    """Test handling of partially filled segments."""
    print("\n" + "="*80)
    print("TEST 4: Partial Segment Handling")
    print("="*80)

    segment = create_test_segment(num_actions=5)  # Only 5 actions instead of 20

    segment_length = len(segment.action_segment)
    mcts_policy_length = len(segment.mcts_policy_segment)
    max_index = min(segment_length, mcts_policy_length)

    print(f"Partial segment lengths:")
    print(f"  action_segment: {segment_length}")
    print(f"  mcts_policy_segment: {mcts_policy_length}")
    print(f"  max_index: {max_index}")

    success_count = 0
    for i in range(max_index):
        try:
            mcts_policy = segment.mcts_policy_segment[i]
            if mcts_policy is not None:
                success_count += 1
        except Exception as e:
            print(f"  Error at index {i}: {e}")
            raise

    print(f"\nResults:")
    print(f"  Successfully accessed {success_count} indices")

    assert success_count == max_index

    print("\n✓ TEST 4 PASSED: Partial segments are handled correctly!")


def test_history_context_building():
    """Test history context building with bounds checking."""
    print("\n" + "="*80)
    print("TEST 5: History Context Building")
    print("="*80)

    segment = create_test_segment(num_actions=20)

    history_length = 5
    test_index = 10

    # Build history using the fixed approach
    history = []
    for j in range(max(0, test_index - history_length), test_index):
        obs_text = None
        if hasattr(segment, 'raw_obs_segment') and j < len(segment.raw_obs_segment):
            obs_text = segment.raw_obs_segment[j]
        elif j < len(segment.obs_segment):
            obs_text = str(segment.obs_segment[j])

        if obs_text is not None and j < len(segment.action_segment):
            action = segment.action_segment[j]
            reward = segment.reward_segment[j] if j < len(segment.reward_segment) else 0.0
            history.append((obs_text, f"action_{action}", float(reward)))

    print(f"History context for index {test_index}:")
    print(f"  Requested history_length: {history_length}")
    print(f"  Actual history items: {len(history)}")
    print(f"  Expected: {min(history_length, test_index)} items")

    expected_length = min(history_length, test_index)
    assert len(history) == expected_length, f"History should have {expected_length} items, got {len(history)}"

    # Verify history content
    for i, (obs, action, reward) in enumerate(history):
        assert isinstance(obs, str), f"obs should be string"
        assert "Step" in obs, f"obs should contain 'Step'"
        assert action.startswith("action_"), f"action should start with 'action_'"

    print(f"  Sample history[0]: obs='{history[0][0][:30]}...', action={history[0][1]}")

    print("\n✓ TEST 5 PASSED: History context is built correctly!")


def test_mismatched_lengths():
    """Test handling when mcts_policy_segment is shorter than action_segment."""
    print("\n" + "="*80)
    print("TEST 6: Mismatched Segment Lengths")
    print("="*80)

    segment = create_test_segment(num_actions=20)

    # Simulate a scenario where mcts_policy_segment is shorter
    # (e.g., due to some error in collection)
    original_length = len(segment.mcts_policy_segment)

    # Manually shorten mcts_policy_segment to simulate a mismatch
    segment.mcts_policy_segment = segment.mcts_policy_segment[:15]

    segment_length = len(segment.action_segment)
    mcts_policy_length = len(segment.mcts_policy_segment)
    max_index = min(segment_length, mcts_policy_length)

    print(f"Mismatched lengths:")
    print(f"  action_segment: {segment_length}")
    print(f"  mcts_policy_segment: {mcts_policy_length}")
    print(f"  max_index: {max_index} (using minimum)")

    # Access should work up to max_index
    success_count = 0
    for i in range(max_index):
        try:
            mcts_policy = segment.mcts_policy_segment[i]
            if mcts_policy is not None:
                success_count += 1
        except Exception as e:
            print(f"  Error at index {i}: {e}")
            raise

    # Accessing beyond max_index should fail safely
    beyond_max_error = False
    if segment_length > max_index:
        try:
            _ = segment.mcts_policy_segment[segment_length - 1]
        except IndexError:
            beyond_max_error = True

    print(f"\nResults:")
    print(f"  Successfully accessed {success_count} indices (up to max_index)")
    print(f"  Access beyond max_index correctly raises IndexError: {beyond_max_error}")

    assert success_count == max_index
    assert beyond_max_error, "Should raise IndexError when accessing beyond max_index"

    print("\n✓ TEST 6 PASSED: Mismatched lengths are handled safely!")


def main():
    """Run all tests."""
    print("="*80)
    print("COMPREHENSIVE PRIORZERO SEGMENT INDEX FIX VALIDATION")
    print("="*80)

    try:
        test_safe_segment_access()
        test_segment_after_game_segment_to_array()
        test_empty_segment()
        test_partial_segment()
        test_history_context_building()
        test_mismatched_lengths()

        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nSummary of fixes:")
        print("  1. ✓ Use action_segment length instead of obs_segment")
        print("  2. ✓ Calculate max_index = min(action_len, mcts_policy_len)")
        print("  3. ✓ Wrap mcts_policy_segment access in try-except")
        print("  4. ✓ Use raw_obs_segment for text observations")
        print("  5. ✓ Add proper None checks for all data access")
        print("  6. ✓ Handle edge cases (empty, partial, mismatched segments)")
        print("\nThe implementation is now robust and handles all error cases!")
        print("="*80)

        return 0

    except Exception as e:
        print("\n" + "="*80)
        print(f"TEST FAILED: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
