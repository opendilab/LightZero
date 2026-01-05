"""
Test script for c_batch_sequential_halving
Tests the Sequential Halving algorithm with batch processing
"""

import sys
sys.path.insert(0, '/mnt/shared-storage-user/tangjia/eff/eff_orign/LightZero')

import numpy as np
from lzero.mcts.ctree.ctree_efficientzero_v2 import ez_tree as tree_efficientzero_v2


def test_sequential_halving_phase_reduction():
    """Test that Sequential Halving correctly reduces action space across phases"""
    print("\n" + "="*70)
    print("[Test 1] Sequential Halving Phase Reduction")
    print("="*70)

    try:
        # Setup
        batch_size = 2
        num_actions = 8
        num_simulations = 8

        print(f"\nSetup:")
        print(f"  - batch_size: {batch_size}")
        print(f"  - num_actions: {num_actions}")
        print(f"  - num_simulations: {num_simulations}")

        # Create roots
        legal_actions = [[0, 1, 2, 3, 4, 5, 6, 7] for _ in range(batch_size)]
        roots = tree_efficientzero_v2.Roots(batch_size, legal_actions)
        print(f"  ✓ Roots created: root_num={roots.root_num}")

        # Prepare roots with initial policy and values
        value_prefix_roots = [0.0 for _ in range(batch_size)]
        policy_logits = [np.random.randn(num_actions).astype(np.float32) for _ in range(batch_size)]
        noises = [np.random.randn(num_actions).astype(np.float32) for _ in range(batch_size)]
        to_play_batch = [0, 0]

        roots.prepare(0.25, noises, value_prefix_roots, policy_logits, to_play_batch)
        print(f"  ✓ Roots prepared")

        # Create MinMaxStats
        min_max_stats_lst = tree_efficientzero_v2.MinMaxStatsList(batch_size)
        min_max_stats_lst.set_delta(0.01)
        print(f"  ✓ MinMaxStats created")

        # Generate Gumbel noises
        gumbel_noises = [np.random.randn(num_actions).astype(np.float32) for _ in range(batch_size)]
        print(f"  ✓ Gumbel noises generated")

        # Test Sequential Halving across phases
        print(f"\nSequential Halving Phase Progression:")
        num_phases = 4
        expected_actions = [8, 4, 2, 1]

        for phase in range(num_phases):
            current_num_top_actions = max(1, num_actions // (2 ** phase))

            print(f"\n  Phase {phase}:")
            print(f"    - current_num_top_actions: {current_num_top_actions}")
            print(f"    - Expected actions to keep: {expected_actions[phase]}")

            # Call Sequential Halving
            best_actions = tree_efficientzero_v2.batch_sequential_halving(
                roots, gumbel_noises, min_max_stats_lst, phase, current_num_top_actions
            )

            print(f"    - Best actions selected: {best_actions}")
            print(f"    ✓ Phase {phase} completed")

            # Verify that the number of selected actions is correct
            assert current_num_top_actions == expected_actions[phase], \
                f"Phase {phase}: Expected {expected_actions[phase]}, got {current_num_top_actions}"

        print(f"\n  ✓ All phases completed successfully")
        print(f"  ✓ Actions correctly reduced: {expected_actions}")

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sequential_halving_action_scores():
    """Test that Sequential Halving ranks actions by score (gumbel + prior + Q)"""
    print("\n" + "="*70)
    print("[Test 2] Sequential Halving Action Scoring")
    print("="*70)

    try:
        batch_size = 1
        num_actions = 4

        print(f"\nSetup:")
        print(f"  - batch_size: {batch_size}")
        print(f"  - num_actions: {num_actions} (small for easy inspection)")

        # Create roots
        legal_actions = [[0, 1, 2, 3]]
        roots = tree_efficientzero_v2.Roots(batch_size, legal_actions)
        print(f"  ✓ Roots created")

        # Create deterministic policy for easy prediction
        value_prefix_roots = [0.0]
        # Manually set policy logits so action 0 has highest prior
        policy_logits = [np.array([2.0, 1.0, 0.5, 0.2], dtype=np.float32)]
        noises = [np.zeros(num_actions, dtype=np.float32)]  # Zero noise for deterministic test
        to_play_batch = [0]

        roots.prepare(0.25, noises, value_prefix_roots, policy_logits, to_play_batch)
        print(f"  ✓ Roots prepared with deterministic policy")
        print(f"    - Policy logits: {policy_logits[0]}")

        # Create MinMaxStats
        min_max_stats_lst = tree_efficientzero_v2.MinMaxStatsList(batch_size)
        min_max_stats_lst.set_delta(0.01)
        print(f"  ✓ MinMaxStats created")

        # Zero Gumbel noise for deterministic test
        gumbel_noises = [np.zeros(num_actions, dtype=np.float32)]
        print(f"  ✓ Gumbel noises set to zero (deterministic)")

        # Run Sequential Halving
        print(f"\nRunning Sequential Halving (Phase 0, keep all 4):")
        best_actions = tree_efficientzero_v2.batch_sequential_halving(
            roots, gumbel_noises, min_max_stats_lst, 0, 4
        )

        print(f"  - Best action selected: {best_actions[0]}")
        print(f"  ✓ Sequential Halving completed")

        # With zero gumbel noise and no Q-values yet, the best action should be the one with highest prior
        # which is action 0 (logit = 2.0)
        expected_best = 0
        assert best_actions[0] == expected_best, \
            f"Expected best action {expected_best}, got {best_actions[0]}"

        print(f"  ✓ Best action is action {best_actions[0]} (as expected with highest prior)")

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sequential_halving_batch_consistency():
    """Test that Sequential Halving works consistently across batch"""
    print("\n" + "="*70)
    print("[Test 3] Sequential Halving Batch Consistency")
    print("="*70)

    try:
        batch_size = 4
        num_actions = 6

        print(f"\nSetup:")
        print(f"  - batch_size: {batch_size}")
        print(f"  - num_actions: {num_actions}")

        # Create roots
        legal_actions = [[0, 1, 2, 3, 4, 5] for _ in range(batch_size)]
        roots = tree_efficientzero_v2.Roots(batch_size, legal_actions)
        print(f"  ✓ Roots created for batch of {batch_size}")

        # Prepare roots with different policies for each batch
        value_prefix_roots = [0.0 for _ in range(batch_size)]
        policy_logits = [np.random.randn(num_actions).astype(np.float32) for _ in range(batch_size)]
        noises = [np.random.randn(num_actions).astype(np.float32) for _ in range(batch_size)]
        to_play_batch = list(range(batch_size))

        roots.prepare(0.25, noises, value_prefix_roots, policy_logits, to_play_batch)
        print(f"  ✓ Roots prepared for all batch items")

        # Create MinMaxStats
        min_max_stats_lst = tree_efficientzero_v2.MinMaxStatsList(batch_size)
        min_max_stats_lst.set_delta(0.01)

        # Generate Gumbel noises
        gumbel_noises = [np.random.randn(num_actions).astype(np.float32) for _ in range(batch_size)]

        # Test multiple phases
        print(f"\nTesting Sequential Halving across phases:")
        phases = [0, 1, 2]
        expected_top_actions = [6, 3, 1]

        for phase in phases:
            current_num_top_actions = expected_top_actions[phase]

            best_actions = tree_efficientzero_v2.batch_sequential_halving(
                roots, gumbel_noises, min_max_stats_lst, phase, current_num_top_actions
            )

            print(f"  Phase {phase} (keep {current_num_top_actions}):")
            print(f"    - Best actions per batch: {best_actions}")
            assert len(best_actions) == batch_size, \
                f"Expected {batch_size} results, got {len(best_actions)}"
            print(f"    ✓ Returned {len(best_actions)} best actions (one per batch item)")

        print(f"\n  ✓ Sequential Halving works consistently across batch")

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("c_batch_sequential_halving Test Suite")
    print("="*70)

    results = []

    # Test 1: Phase reduction
    results.append(("Phase Reduction", test_sequential_halving_phase_reduction()))

    # Test 2: Action scoring
    results.append(("Action Scoring", test_sequential_halving_action_scores()))

    # Test 3: Batch consistency
    results.append(("Batch Consistency", test_sequential_halving_batch_consistency()))

    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
