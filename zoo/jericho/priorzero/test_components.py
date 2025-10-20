#!/usr/bin/env python
"""
test_components.py

Lightweight component testing script that doesn't require full environment setup.
This avoids mujoco and other heavy dependencies.
"""

import sys
import traceback


def test_config():
    """Test configuration generation."""
    print("\n" + "="*80)
    print("TEST 1: Configuration Generation")
    print("="*80)

    try:
        from priorzero_config import (
            get_priorzero_config,
            get_priorzero_config_for_quick_test,
            get_config_pure_unizero,
            get_jericho_action_mapping
        )

        # Test action mapping
        print("\n1.1 Testing action mapping...")
        action_map, action_inv_map = get_jericho_action_mapping('zork1.z5')
        print(f"  ✓ Action space size: {len(action_map)}")
        print(f"  ✓ Sample actions: {list(action_map.keys())[:3]}")

        # Test standard config
        print("\n1.2 Testing standard config...")
        main_cfg, create_cfg = get_priorzero_config(env_id='zork1.z5', seed=0)
        print(f"  ✓ Exp name: {main_cfg.exp_name}")
        print(f"  ✓ LLM model: {main_cfg.policy.llm_policy_cfg.pretrain_llm_path}")
        print(f"  ✓ Batch size: {main_cfg.policy.batch_size}")
        print(f"  ✓ Action map loaded: {hasattr(main_cfg.policy, 'action_map')}")

        # Test quick config
        print("\n1.3 Testing quick test config...")
        test_cfg, _ = get_priorzero_config_for_quick_test()
        print(f"  ✓ Batch size (reduced): {test_cfg.policy.batch_size}")
        print(f"  ✓ Num simulations (reduced): {test_cfg.policy.num_simulations}")

        # Test pure unizero config
        print("\n1.4 Testing pure UniZero config...")
        unizero_cfg, _ = get_config_pure_unizero()
        print(f"  ✓ LLM loss weight: {unizero_cfg.policy.llm_policy_cfg.llm_loss_weight}")

        print("\n✅ Configuration test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Configuration test FAILED: {e}")
        traceback.print_exc()
        return False


def test_game_segment():
    """Test game segment."""
    print("\n" + "="*80)
    print("TEST 2: Game Segment")
    print("="*80)

    try:
        import numpy as np
        from game_segment_priorzero import GameSegment, validate_game_segment

        # Create mock objects
        class MockActionSpace:
            def __init__(self, n):
                self.n = n

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
                    'observation_shape': (4, 84, 84),
                    'image_channel': 4
                })()

        action_space = MockActionSpace(n=10)
        config = MockConfig()

        print("\n2.1 Creating game segment...")
        segment = GameSegment(action_space, game_segment_length=50, config=config)
        print(f"  ✓ Segment created with max length: 50")

        print("\n2.2 Resetting segment...")
        init_obs = [np.zeros((4, 84, 84)) for _ in range(4)]
        segment.reset(init_obs)
        print(f"  ✓ Segment reset with {len(segment.obs_segment)} initial observations")

        print("\n2.3 Adding transitions...")
        for i in range(5):
            obs = np.random.rand(4, 84, 84).astype(np.float32)
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

        print(f"  ✓ Added 5 transitions")

        print("\n2.4 Checking statistics...")
        stats = segment.get_stats()
        print(f"  ✓ Segment length: {stats['segment_length']}")
        print(f"  ✓ MCTS policies: {stats['num_mcts_policies']}")
        print(f"  ✓ Raw observations: {stats['num_raw_obs']}")

        print("\n2.5 Validating segment...")
        is_valid = validate_game_segment(segment)
        print(f"  ✓ Segment valid: {is_valid}")

        print("\n✅ Game segment test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Game segment test FAILED: {e}")
        traceback.print_exc()
        return False


def test_policy_helpers():
    """Test policy helper functions."""
    print("\n" + "="*80)
    print("TEST 3: Policy Helper Functions")
    print("="*80)

    try:
        import numpy as np
        from priorzero_policy import (
            parse_llm_action_ranking,
            format_mcts_policy_to_text,
            build_llm_prompt
        )

        # Test parse_llm_action_ranking
        print("\n3.1 Testing parse_llm_action_ranking...")
        action_map = {
            "go north": 0,
            "go south": 1,
            "take key": 2,
            "open door": 3
        }

        llm_text = "1. go north\n2. take key\n3. open door"
        policy = parse_llm_action_ranking(llm_text, action_map, 4)
        print(f"  ✓ Parsed policy shape: {policy.shape}")
        print(f"  ✓ Policy sum: {policy.sum():.3f}")
        print(f"  ✓ Top action: {np.argmax(policy)}")

        # Test format_mcts_policy_to_text
        print("\n3.2 Testing format_mcts_policy_to_text...")
        mcts_policy = np.array([0.5, 0.3, 0.15, 0.05])
        action_inv_map = {0: "go north", 1: "go south", 2: "take key", 3: "open door"}
        text = format_mcts_policy_to_text(mcts_policy, action_inv_map, top_k=3)
        print(f"  ✓ Generated text:\n{text}")

        # Test build_llm_prompt
        print("\n3.3 Testing build_llm_prompt...")
        current_obs = "You are in a dark room. There is a key on the floor."
        history = [
            ("You are in a hallway.", "go north", 0.0),
            ("You see a door.", "open door", 1.0)
        ]
        prompt = build_llm_prompt(current_obs, history, use_cot=True)
        print(f"  ✓ Prompt length: {len(prompt)} chars")
        print(f"  ✓ Contains CoT: {'Think step-by-step' in prompt}")

        print("\n✅ Policy helpers test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Policy helpers test FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PriorZero Component Tests")
    print("="*80)
    print("\nThis script tests individual components without requiring")
    print("full environment setup (avoids mujoco, gym, etc.)")
    print("="*80)

    results = []

    # Run tests
    results.append(("Configuration", test_config()))
    results.append(("Game Segment", test_game_segment()))
    results.append(("Policy Helpers", test_policy_helpers()))

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20s}: {status}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\nNext steps:")
        print("1. Fix numpy version: bash fix_environment.sh")
        print("2. Test full pipeline: python priorzero_entry.py --quick_test")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("="*80)
        print("\nPlease fix the errors above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
