"""
Simple test for Batch MCTS module
"""
import sys
import os
import numpy as np

# Add module path - use absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(script_dir, 'lzero/mcts/ctree/ctree_alphazero/build')
sys.path.insert(0, module_path)
print(f"Looking for module in: {module_path}")

try:
    import mcts_alphazero_batch
    print("="*70)
    print("Batch MCTS Module Test")
    print("="*70)
except ImportError as e:
    print(f"❌ Failed to import module: {e}")
    sys.exit(1)

def test_roots_creation():
    """Test 1: Create batch roots"""
    print("\n[Test 1] Creating Batch Roots...")

    batch_size = 4
    legal_actions_list = [[0, 1, 2, 3, 4, 5, 6, 7, 8] for _ in range(batch_size)]

    roots = mcts_alphazero_batch.Roots(batch_size, legal_actions_list)

    assert roots.num == batch_size, f"Expected {batch_size} roots, got {roots.num}"
    print(f"  ✓ Created {batch_size} roots successfully")

    return roots

def test_roots_prepare():
    """Test 2: Prepare roots with noise"""
    print("\n[Test 2] Preparing Roots...")

    batch_size = 4
    action_space = 9
    legal_actions_list = [[i for i in range(action_space)] for _ in range(batch_size)]

    roots = mcts_alphazero_batch.Roots(batch_size, legal_actions_list)

    # Generate noise
    noises = []
    for _ in range(batch_size):
        noise = np.random.dirichlet([0.3] * action_space)
        noises.append(noise.tolist())

    # Generate policy
    values = [0.0] * batch_size
    policy_logits_pool = []
    for _ in range(batch_size):
        policy = np.random.randn(action_space)
        policy = np.exp(policy) / np.exp(policy).sum()
        policy_logits_pool.append(policy.tolist())

    # Prepare
    roots.prepare(0.25, noises, values, policy_logits_pool)
    print(f"  ✓ Roots prepared with noise")

    return roots

def test_batch_traverse():
    """Test 3: Batch traverse"""
    print("\n[Test 3] Batch Traverse...")

    batch_size = 4
    action_space = 9
    legal_actions_list = [[i for i in range(action_space)] for _ in range(batch_size)]

    roots = mcts_alphazero_batch.Roots(batch_size, legal_actions_list)

    # Prepare
    noises = [np.random.dirichlet([0.3] * action_space).tolist() for _ in range(batch_size)]
    values = [0.0] * batch_size
    policy_logits_pool = []
    for _ in range(batch_size):
        policy = np.random.randn(action_space)
        policy = np.exp(policy) / np.exp(policy).sum()
        policy_logits_pool.append(policy.tolist())

    roots.prepare(0.25, noises, values, policy_logits_pool)

    # Traverse
    current_legal_actions = [[i for i in range(action_space)] for _ in range(batch_size)]
    results = mcts_alphazero_batch.batch_traverse(
        roots, 19652, 1.25, current_legal_actions
    )

    print(f"  ✓ Traverse completed")
    print(f"    - Latent state indices: {results.latent_state_index_in_search_path}")
    print(f"    - Batch indices: {results.latent_state_index_in_batch}")
    print(f"    - Last actions: {results.last_actions}")

    assert len(results.last_actions) == batch_size

    return roots, results

def test_batch_backpropagate():
    """Test 4: Batch backpropagate"""
    print("\n[Test 4] Batch Backpropagate...")

    batch_size = 4
    action_space = 9
    legal_actions_list = [[i for i in range(action_space)] for _ in range(batch_size)]

    roots = mcts_alphazero_batch.Roots(batch_size, legal_actions_list)

    # Prepare
    noises = [np.random.dirichlet([0.3] * action_space).tolist() for _ in range(batch_size)]
    values = [0.0] * batch_size
    policy_logits_pool = []
    for _ in range(batch_size):
        policy = np.random.randn(action_space)
        policy = np.exp(policy) / np.exp(policy).sum()
        policy_logits_pool.append(policy.tolist())

    roots.prepare(0.25, noises, values, policy_logits_pool)

    # Traverse
    current_legal_actions = [[i for i in range(action_space)] for _ in range(batch_size)]
    results = mcts_alphazero_batch.batch_traverse(
        roots, 19652, 1.25, current_legal_actions
    )

    # Backpropagate
    values = [0.5, -0.3, 0.8, 0.1]
    policy_logits_batch = []
    for _ in range(batch_size):
        policy = np.random.randn(action_space)
        policy = np.exp(policy) / np.exp(policy).sum()
        policy_logits_batch.append(policy.tolist())

    legal_actions_batch = [[i for i in range(action_space)] for _ in range(batch_size)]

    mcts_alphazero_batch.batch_backpropagate(
        results, values, policy_logits_batch, legal_actions_batch, "play_with_bot_mode"
    )

    print(f"  ✓ Backpropagate completed")

    # Check distributions
    distributions = roots.get_distributions()
    print(f"    - Example distribution: {[f'{p:.3f}' for p in distributions[0][:5]]}...")

    return roots

def test_full_mcts():
    """Test 5: Full MCTS simulation"""
    print("\n[Test 5] Full MCTS Simulation...")

    batch_size = 8
    num_simulations = 10
    action_space = 9
    legal_actions_list = [[i for i in range(action_space)] for _ in range(batch_size)]

    roots = mcts_alphazero_batch.Roots(batch_size, legal_actions_list)

    # Initialize
    noises = [np.random.dirichlet([0.3] * action_space).tolist() for _ in range(batch_size)]
    values = [0.0] * batch_size
    policy_logits_pool = []
    for _ in range(batch_size):
        policy = np.random.randn(action_space)
        policy = np.exp(policy) / np.exp(policy).sum()
        policy_logits_pool.append(policy.tolist())

    roots.prepare(0.25, noises, values, policy_logits_pool)

    # Run simulations
    for sim_idx in range(num_simulations):
        # Traverse
        current_legal_actions = [[i for i in range(action_space)] for _ in range(batch_size)]
        results = mcts_alphazero_batch.batch_traverse(
            roots, 19652, 1.25, current_legal_actions
        )

        # Mock network inference
        values = np.random.randn(batch_size).tolist()
        policy_logits_batch = []
        for _ in range(batch_size):
            policy = np.random.randn(action_space)
            policy = np.exp(policy) / np.exp(policy).sum()
            policy_logits_batch.append(policy.tolist())

        legal_actions_batch = [[i for i in range(action_space)] for _ in range(batch_size)]

        # Backpropagate
        mcts_alphazero_batch.batch_backpropagate(
            results, values, policy_logits_batch, legal_actions_batch, "play_with_bot_mode"
        )

    # Get results
    distributions = roots.get_distributions()
    root_values = roots.get_values()

    print(f"  ✓ Completed {num_simulations} simulations for {batch_size} environments")
    print(f"    - Example distribution: {[f'{p:.3f}' for p in distributions[0][:5]]}...")
    print(f"    - Root values: {[f'{v:.3f}' for v in root_values]}")

    # Verify all distributions sum to ~1.0
    for i, dist in enumerate(distributions):
        dist_sum = sum(dist)
        assert abs(dist_sum - 1.0) < 0.01, f"Distribution {i} sum is {dist_sum}, expected ~1.0"

    print(f"  ✓ All distributions sum to 1.0")

    return roots

def main():
    try:
        test_roots_creation()
        test_roots_prepare()
        test_batch_traverse()
        test_batch_backpropagate()
        test_full_mcts()

        print("\n" + "="*70)
        print("✓ All tests passed!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Try: python test_performance_comparison.py")
        print("  2. Use alphazero_batch policy in your training config")
        print("  3. See ALPHAZERO_BATCH_IMPLEMENTATION_GUIDE.md for details")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
