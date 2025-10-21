#!/usr/bin/env python3
"""
Test script to verify the segment index fix for PriorZero policy.

This script tests that:
1. segment_length is correctly calculated from action_segment
2. mcts_policy_segment indexing works correctly
3. raw_obs_segment is used for text observations
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


def test_segment_lengths():
    """Test that segment lengths are correct."""
    print("\n" + "="*80)
    print("TEST 1: Segment Length Calculations")
    print("="*80)

    action_space = MockActionSpace(n=10)
    config = MockConfig()

    # Create segment with game_segment_length=20
    segment = GameSegment(action_space, game_segment_length=20, config=config)

    # Reset with initial observations (frame_stack_num=4)
    init_obs = [np.zeros((84, 84, 3)) for _ in range(4)]
    segment.reset(init_obs)

    print(f"Initial state:")
    print(f"  obs_segment length: {len(segment.obs_segment)} (should be 4)")
    print(f"  action_segment length: {len(segment.action_segment)} (should be 0)")
    print(f"  mcts_policy_segment length: {len(segment.mcts_policy_segment)} (should be 0)")

    # Add exactly 20 transitions (game_segment_length)
    for i in range(20):
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

    print(f"\nAfter adding 20 transitions:")
    print(f"  obs_segment length: {len(segment.obs_segment)} (should be 24 = 4 + 20)")
    print(f"  action_segment length: {len(segment.action_segment)} (should be 20)")
    print(f"  reward_segment length: {len(segment.reward_segment)} (should be 20)")
    print(f"  mcts_policy_segment length: {len(segment.mcts_policy_segment)} (should be 20)")
    print(f"  raw_obs_segment length: {len(segment.raw_obs_segment)} (should be 20)")

    # Verify lengths
    assert len(segment.obs_segment) == 24, f"obs_segment should be 24, got {len(segment.obs_segment)}"
    assert len(segment.action_segment) == 20, f"action_segment should be 20, got {len(segment.action_segment)}"
    assert len(segment.mcts_policy_segment) == 20, f"mcts_policy_segment should be 20, got {len(segment.mcts_policy_segment)}"
    assert len(segment.raw_obs_segment) == 20, f"raw_obs_segment should be 20, got {len(segment.raw_obs_segment)}"

    print("\n✓ TEST 1 PASSED: Segment lengths are correct!")
    return segment


def test_index_access(segment):
    """Test that we can access all indices correctly."""
    print("\n" + "="*80)
    print("TEST 2: Index Access Validation")
    print("="*80)

    # Test using action_segment length (correct approach)
    segment_length = len(segment.action_segment)
    print(f"\nUsing action_segment length: {segment_length}")

    success_count = 0
    for i in range(segment_length):
        try:
            # This should NOT raise IndexError
            mcts_policy = segment.mcts_policy_segment[i]
            raw_obs = segment.raw_obs_segment[i]
            action = segment.action_segment[i]

            assert mcts_policy is not None, f"MCTS policy at index {i} is None"
            assert raw_obs is not None, f"Raw obs at index {i} is None"

            success_count += 1
        except IndexError as e:
            print(f"  ✗ IndexError at index {i}: {e}")
            raise

    print(f"  ✓ Successfully accessed all {success_count} indices")

    # Test what would happen with obs_segment length (incorrect approach)
    obs_segment_length = len(segment.obs_segment)
    print(f"\nTesting with obs_segment length (WRONG): {obs_segment_length}")

    would_fail = False
    for i in range(obs_segment_length):
        try:
            _ = segment.mcts_policy_segment[i]
        except IndexError:
            if i >= len(segment.action_segment):
                would_fail = True
                print(f"  ✗ Would fail at index {i} (>= {len(segment.action_segment)})")
                break

    assert would_fail, "Expected IndexError when using obs_segment length"

    print("\n✓ TEST 2 PASSED: Index access works correctly!")


def test_raw_obs_segment_access(segment):
    """Test that raw_obs_segment contains correct text data."""
    print("\n" + "="*80)
    print("TEST 3: Raw Observation Segment Access")
    print("="*80)

    segment_length = len(segment.action_segment)

    # Check all raw observations
    for i in range(segment_length):
        raw_obs = segment.raw_obs_segment[i]

        # Verify it's a string
        assert isinstance(raw_obs, str), f"raw_obs at {i} should be str, got {type(raw_obs)}"

        # Verify it contains expected content
        assert f"Step {i}" in raw_obs, f"raw_obs at {i} should contain 'Step {i}'"

    print(f"  ✓ All {segment_length} raw observations are valid strings")
    print(f"  Sample raw_obs[0]: {segment.raw_obs_segment[0]}")
    print(f"  Sample raw_obs[10]: {segment.raw_obs_segment[10]}")

    print("\n✓ TEST 3 PASSED: Raw observations are correctly stored!")


def test_mcts_policy_segment_access(segment):
    """Test that mcts_policy_segment contains valid probability distributions."""
    print("\n" + "="*80)
    print("TEST 4: MCTS Policy Segment Validation")
    print("="*80)

    segment_length = len(segment.action_segment)

    # Check all MCTS policies
    for i in range(segment_length):
        mcts_policy = segment.mcts_policy_segment[i]

        # Verify it's a numpy array
        assert isinstance(mcts_policy, np.ndarray), f"MCTS policy at {i} should be ndarray, got {type(mcts_policy)}"

        # Verify it's a valid probability distribution
        assert abs(mcts_policy.sum() - 1.0) < 0.01, f"MCTS policy at {i} should sum to 1.0, got {mcts_policy.sum()}"
        assert np.all(mcts_policy >= 0), f"MCTS policy at {i} should be non-negative"

    print(f"  ✓ All {segment_length} MCTS policies are valid probability distributions")
    print(f"  Sample policy[0] shape: {segment.mcts_policy_segment[0].shape}")
    print(f"  Sample policy[0] sum: {segment.mcts_policy_segment[0].sum():.6f}")

    print("\n✓ TEST 4 PASSED: MCTS policies are correctly normalized!")


def main():
    """Run all tests."""
    print("="*80)
    print("PRIORZERO SEGMENT INDEX FIX VALIDATION")
    print("="*80)

    try:
        # Test 1: Segment lengths
        segment = test_segment_lengths()

        # Test 2: Index access
        test_index_access(segment)

        # Test 3: Raw observations
        test_raw_obs_segment_access(segment)

        # Test 4: MCTS policies
        test_mcts_policy_segment_access(segment)

        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nSummary:")
        print("  1. ✓ Segment lengths are correctly calculated")
        print("  2. ✓ Using action_segment.length prevents IndexError")
        print("  3. ✓ raw_obs_segment stores text observations correctly")
        print("  4. ✓ mcts_policy_segment stores valid probability distributions")
        print("\nThe fix resolves the original IndexError:")
        print("  - Changed: segment_length = len(segment.obs_segment)")
        print("  - To:      segment_length = len(segment.action_segment)")
        print("  - Added:   Use raw_obs_segment for text observations")
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
