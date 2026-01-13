"""
Test script for GAE computation in MuZeroCollector.

This script tests both the original (_batch_compute_gae_for_pool_bak) and new 
(_batch_compute_gae_for_pool) implementations to ensure they produce the same results.
"""

import numpy as np
import torch
from easydict import EasyDict
from unittest.mock import Mock, MagicMock

# Import the collector class
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lzero.worker.muzero_collector import MuZeroCollector
from lzero.mcts.buffer.game_segment import GameSegment


class MockGameSegment:
    """Mock GameSegment for testing"""
    def __init__(self, episode_id, values, rewards, actions):
        self.episode_id = episode_id
        self.root_value_segment = values.copy()
        self.reward_segment = rewards.copy()
        self.action_segment = actions.copy()
        self.advantage_segment = []
        self.return_segment = []
    
    def __len__(self):
        return len(self.action_segment)


def create_test_data():
    """Create test data for GAE computation"""
    
    # Test case 1: Simple episode with 5 steps
    # Episode 0: 3 segments
    episode_0_segments = [
        MockGameSegment(
            episode_id=0,
            values=[1.0, 2.0, 3.0],
            rewards=[0.1, 0.2, 0.3],
            actions=[0, 1, 0]
        ),
        MockGameSegment(
            episode_id=0,
            values=[4.0, 5.0],
            rewards=[0.4, 0.5],
            actions=[1, 0]
        ),
        MockGameSegment(
            episode_id=0,
            values=[6.0],
            rewards=[0.6],
            actions=[0]
        ),
    ]
    
    # Test case 2: Another episode with 4 steps
    # Episode 1: 2 segments
    episode_1_segments = [
        MockGameSegment(
            episode_id=1,
            values=[10.0, 11.0],
            rewards=[1.0, 1.1],
            actions=[0, 1]
        ),
        MockGameSegment(
            episode_id=1,
            values=[12.0, 13.0],
            rewards=[1.2, 1.3],
            actions=[1, 0]
        ),
    ]
    
    return episode_0_segments, episode_1_segments


def create_mock_collector():
    """Create a mock MuZeroCollector instance"""
    collector = Mock(spec=MuZeroCollector)
    
    # Set up PPO parameters
    collector.ppo_gamma = 0.99
    collector.ppo_gae_lambda = 0.95
    
    # Create logger mock
    collector._logger = Mock()
    collector._logger.info = Mock()
    
    return collector


def test_gae_computation():
    """Test GAE computation with both implementations"""
    
    print("=" * 80)
    print("Testing GAE Computation")
    print("=" * 80)
    
    # Create test data
    episode_0_segments, episode_1_segments = create_test_data()
    
    # Create game_segment_pool
    game_segment_pool = []
    priorities = 1.0
    done_flag = False
    
    # Add episode 0 segments
    for seg in episode_0_segments:
        game_segment_pool.append((seg, priorities, done_flag))
    
    # Add episode 1 segments
    for seg in episode_1_segments:
        game_segment_pool.append((seg, priorities, done_flag))
    
    # Make deep copies for comparison
    def copy_segment(seg):
        new_seg = MockGameSegment(
            seg.episode_id, 
            seg.root_value_segment.copy() if isinstance(seg.root_value_segment, list) else seg.root_value_segment.copy(),
            seg.reward_segment.copy() if isinstance(seg.reward_segment, list) else seg.reward_segment.copy(),
            seg.action_segment.copy() if isinstance(seg.action_segment, list) else seg.action_segment.copy()
        )
        return new_seg
    
    pool_bak = [(copy_segment(seg), priorities, done_flag) for seg, _, _ in game_segment_pool]
    pool_new = [(copy_segment(seg), priorities, done_flag) for seg, _, _ in game_segment_pool]
    
    collector_bak = create_mock_collector()
    collector_bak.game_segment_pool = pool_bak
    
    collector_new = create_mock_collector()
    collector_new.game_segment_pool = pool_new
    
    # Test original implementation
    print("\n[1] Testing original implementation (_batch_compute_gae_for_pool_bak)...")
    MuZeroCollector._batch_compute_gae_for_pool_bak(collector_bak)
    
    # Test new implementation
    print("[2] Testing new implementation (_batch_compute_gae_for_pool)...")
    MuZeroCollector._batch_compute_gae_for_pool(collector_new)
    
    # Compare results
    print("\n[3] Comparing results...")
    print("-" * 80)
    
    all_match = True
    for i, ((seg_bak, _, _), (seg_new, _, _)) in enumerate(zip(pool_bak, pool_new)):
        print(f"\nSegment {i} (Episode {seg_bak.episode_id}):")
        print(f"  Length: {len(seg_bak.action_segment)}")
        
        # Compare advantages
        adv_bak = np.array(seg_bak.advantage_segment)
        adv_new = np.array(seg_new.advantage_segment)
        
        if len(adv_bak) != len(adv_new):
            print(f"  ❌ Advantage length mismatch: {len(adv_bak)} vs {len(adv_new)}")
            all_match = False
        else:
            max_diff = np.max(np.abs(adv_bak - adv_new))
            print(f"  Advantages - Max difference: {max_diff:.6f}")
            if max_diff > 1e-5:
                print(f"  ❌ Advantages don't match!")
                print(f"    Original: {adv_bak}")
                print(f"    New:      {adv_new}")
                all_match = False
            else:
                print(f"  ✓ Advantages match")
        
        # Compare returns
        ret_bak = np.array(seg_bak.return_segment)
        ret_new = np.array(seg_new.return_segment)
        
        if len(ret_bak) != len(ret_new):
            print(f"  ❌ Return length mismatch: {len(ret_bak)} vs {len(ret_new)}")
            all_match = False
        else:
            max_diff = np.max(np.abs(ret_bak - ret_new))
            print(f"  Returns - Max difference: {max_diff:.6f}")
            if max_diff > 1e-5:
                print(f"  ❌ Returns don't match!")
                print(f"    Original: {ret_bak}")
                print(f"    New:      {ret_new}")
                all_match = False
            else:
                print(f"  ✓ Returns match")
        
        # Print detailed values for first segment of each episode
        if i == 0 or i == len(episode_0_segments):
            print(f"\n  Detailed values for Segment {i}:")
            print(f"    Values:  {seg_bak.root_value_segment}")
            print(f"    Rewards: {seg_bak.reward_segment}")
            print(f"    Advantages (original): {adv_bak}")
            print(f"    Advantages (new):      {adv_new}")
            print(f"    Returns (original):    {ret_bak}")
            print(f"    Returns (new):         {ret_new}")
    
    print("\n" + "=" * 80)
    if all_match:
        print("✓ All tests passed! Both implementations produce identical results.")
    else:
        print("❌ Tests failed! Implementations produce different results.")
    print("=" * 80)
    
    return all_match


def test_manual_gae_verification():
    """Manually verify GAE computation for a simple case"""
    print("\n" + "=" * 80)
    print("Manual GAE Verification (Simple Case)")
    print("=" * 80)
    
    # Simple case: 3 steps, gamma=0.99, lambda=0.95
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    rewards = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    gamma = 0.99
    gae_lambda = 0.95
    
    print(f"\nInput:")
    print(f"  Values:  {values}")
    print(f"  Rewards: {rewards}")
    print(f"  Gamma:   {gamma}")
    print(f"  Lambda:  {gae_lambda}")
    
    # Manual computation
    advantages = np.zeros_like(rewards)
    gae_val = 0.0
    
    print(f"\nManual computation (backward):")
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        gae_val = delta + gamma * gae_lambda * gae_val
        advantages[t] = gae_val
        print(f"  t={t}: delta={delta:.6f}, gae={gae_val:.6f}")
    
    returns = advantages + values
    
    print(f"\nResults:")
    print(f"  Advantages: {advantages}")
    print(f"  Returns:    {returns}")
    
    # Test with ding library
    print(f"\nDing library computation:")
    from ding.rl_utils import gae_data, gae
    
    value = torch.tensor(values, dtype=torch.float32)
    next_value = torch.cat([value[1:], torch.tensor([0.0], dtype=torch.float32)])
    reward = torch.tensor(rewards, dtype=torch.float32)
    done = torch.tensor([False, False, True], dtype=torch.bool)
    
    compute_adv_data = gae_data(value, next_value, reward, done, None)
    advantages_ding = gae(compute_adv_data, gamma, gae_lambda)
    returns_ding = advantages_ding + value
    
    print(f"  Advantages: {advantages_ding.cpu().numpy()}")
    print(f"  Returns:    {returns_ding.cpu().numpy()}")
    
    # Compare
    max_diff_adv = np.max(np.abs(advantages - advantages_ding.cpu().numpy()))
    max_diff_ret = np.max(np.abs(returns - returns_ding.cpu().numpy()))
    
    print(f"\nComparison:")
    print(f"  Advantages max diff: {max_diff_adv:.6f}")
    print(f"  Returns max diff:    {max_diff_ret:.6f}")
    
    if max_diff_adv < 1e-5 and max_diff_ret < 1e-5:
        print("  ✓ Manual and ding library results match!")
    else:
        print("  ❌ Results don't match!")


if __name__ == "__main__":
    # Run manual verification first
    test_manual_gae_verification()
    
    # Run full test
    success = test_gae_computation()
    
    if success:
        print("\n✅ All tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed!")
        exit(1)

