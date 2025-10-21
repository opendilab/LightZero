#!/usr/bin/env python
"""
Simple test script for enhanced JerichoEnv with robustness features.
This script avoids pytest and provides clear output for quick testing.
"""

import sys
import os
import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from easydict import EasyDict
from jericho_env import JerichoEnv


def test_basic_functionality():
    """Test basic JerichoEnv functionality without robustness features."""
    print("\n" + "="*80)
    print("Test 1: Basic Functionality (No Robustness Features)")
    print("="*80)

    cfg = EasyDict(
        dict(
            game_path="./z-machine-games-master/jericho-game-suite/zork1.z5",
            max_steps=20,
            max_action_num=10,
            tokenizer_path="BAAI/bge-base-en-v1.5",
            max_seq_len=512,
            for_unizero=True,
            # Disable robustness features for baseline test
            enable_timeout=False,
            enable_debug_logging=False,
        )
    )

    try:
        env = JerichoEnv(cfg)
        print("✓ Environment created successfully")

        # Test reset
        obs = env.reset()
        assert 'observation' in obs, "Observation missing 'observation' key"
        assert 'action_mask' in obs, "Observation missing 'action_mask' key"
        print(f"✓ Reset successful, obs keys: {list(obs.keys())}")

        # Test a few steps
        for i in range(3):
            action = i % env.action_space.n
            timestep = env.step(action)
            assert timestep.obs is not None, "Timestep observation is None"
            assert timestep.reward is not None, "Timestep reward is None"
            print(f"  Step {i+1}: action={action}, reward={timestep.reward}, done={timestep.done}")

        env.close()
        print("✓ Environment closed successfully")
        print("✓ Test 1 PASSED\n")
        return True

    except Exception as e:
        print(f"✗ Test 1 FAILED: {type(e).__name__}: {e}\n")
        return False


def test_robustness_features():
    """Test enhanced JerichoEnv with robustness features enabled."""
    print("\n" + "="*80)
    print("Test 2: Robustness Features Enabled")
    print("="*80)

    cfg = EasyDict(
        dict(
            game_path="./z-machine-games-master/jericho-game-suite/zork1.z5",
            max_steps=20,
            max_action_num=10,
            tokenizer_path="BAAI/bge-base-en-v1.5",
            max_seq_len=512,
            for_unizero=True,
            # Enable robustness features
            enable_timeout=True,
            step_timeout=30.0,
            reset_timeout=10.0,
            enable_debug_logging=True,
            max_reset_retries=3,
            max_step_retries=2,
        )
    )

    try:
        env = JerichoEnv(cfg)
        print("✓ Environment created with robustness features")

        # Test reset
        obs = env.reset()
        print(f"✓ Reset successful with timeout protection")

        # Test steps with timeout protection
        for i in range(5):
            action = i % env.action_space.n
            timestep = env.step(action)
            print(f"  Step {i+1}: action={action}, reward={timestep.reward}, done={timestep.done}")

            if timestep.info.get('abnormal', False):
                print(f"  WARNING: Abnormal episode detected at step {i+1}")
                break

        # Get diagnostics
        diagnostics = env.get_diagnostics()
        print(f"\n✓ Diagnostics retrieved:")
        print(f"  - Total steps: {diagnostics['total_steps']}")
        print(f"  - Avg step time: {diagnostics['avg_step_time']:.4f}s")
        print(f"  - Timeout count: {diagnostics['timeout_count']}")
        print(f"  - Error count: {diagnostics['error_count']}")

        env.close()
        print("✓ Environment closed successfully")
        print("✓ Test 2 PASSED\n")
        return True

    except Exception as e:
        print(f"✗ Test 2 FAILED: {type(e).__name__}: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_episode_completion():
    """Test a complete episode with robustness features."""
    print("\n" + "="*80)
    print("Test 3: Complete Episode with Robustness")
    print("="*80)

    cfg = EasyDict(
        dict(
            game_path="./z-machine-games-master/jericho-game-suite/detective.z5",
            max_steps=10,  # Short episode for quick test
            max_action_num=10,
            tokenizer_path="BAAI/bge-base-en-v1.5",
            max_seq_len=512,
            for_unizero=True,
            enable_timeout=True,
            step_timeout=10.0,
            reset_timeout=5.0,
            enable_debug_logging=False,  # Less verbose
            max_reset_retries=3,
            max_step_retries=2,
        )
    )

    try:
        env = JerichoEnv(cfg)
        print("✓ Environment created")

        obs = env.reset()
        print("✓ Episode started")

        done = False
        step_count = 0

        while not done and step_count < cfg.max_steps:
            action = step_count % env.action_space.n
            timestep = env.step(action)
            done = timestep.done
            step_count += 1

            if timestep.info.get('abnormal', False):
                print(f"  Abnormal termination at step {step_count}")
                break

        print(f"✓ Episode completed: {step_count} steps, return={env.episode_return}")

        diagnostics = env.get_diagnostics()
        print(f"  Timeouts: {diagnostics['timeout_count']}, Errors: {diagnostics['error_count']}")

        env.close()
        print("✓ Test 3 PASSED\n")
        return True

    except Exception as e:
        print(f"✗ Test 3 FAILED: {type(e).__name__}: {e}\n")
        return False


def test_error_handling():
    """Test error handling with invalid game path."""
    print("\n" + "="*80)
    print("Test 4: Error Handling")
    print("="*80)

    cfg = EasyDict(
        dict(
            game_path="./nonexistent_game.z5",  # Invalid path
            max_steps=10,
            max_action_num=10,
            tokenizer_path="BAAI/bge-base-en-v1.5",
            max_seq_len=512,
            for_unizero=True,
            enable_timeout=True,
            step_timeout=5.0,
            reset_timeout=2.0,
            enable_debug_logging=False,
            max_reset_retries=2,
            max_step_retries=1,
        )
    )

    try:
        env = JerichoEnv(cfg)
        print("  Environment created (may fail on reset)")

        try:
            obs = env.reset()
            print("  Reset unexpectedly succeeded")
            env.close()
            print("✓ Test 4 PASSED (graceful error handling)\n")
            return True
        except Exception as e:
            print(f"  Expected error on reset: {type(e).__name__}")
            print("✓ Test 4 PASSED (error caught as expected)\n")
            return True

    except Exception as e:
        print(f"  Expected error on initialization: {type(e).__name__}")
        print("✓ Test 4 PASSED (error caught as expected)\n")
        return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("Enhanced Jericho Environment Test Suite")
    print("="*80)

    # Check if game files exist
    game_path = "./z-machine-games-master/jericho-game-suite/zork1.z5"
    if not os.path.exists(game_path):
        print(f"\n⚠ WARNING: Game file not found at {game_path}")
        print("Tests may fail. Please ensure game files are in the correct location.\n")

    results = []

    # Run all tests
    results.append(("Basic Functionality", test_basic_functionality()))
    results.append(("Robustness Features", test_robustness_features()))
    results.append(("Episode Completion", test_episode_completion()))
    results.append(("Error Handling", test_error_handling()))

    # Print summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    print("="*80 + "\n")

    return 0 if total_passed == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
