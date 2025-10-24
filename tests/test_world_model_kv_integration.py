"""
Integration Tests for KV Cache Manager with World Model
========================================================

Tests to validate that KVCacheManager works correctly with the actual
world_model.py patterns and KeysValues objects.

Run with: pytest tests/test_world_model_kv_integration.py -v
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock
from lzero.model.unizero_world_models.kv_cache_manager import (
    KVCachePool,
    KVCacheManager,
    EvictionStrategy,
    CacheStats,
)
from lzero.model.unizero_world_models.kv_caching import KeysValues


class TestKVCacheManagerWithRealKeysValues:
    """Test KVCacheManager with actual KeysValues objects."""

    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.game_segment_length = 100
        config.num_simulations = 50
        return config

    @pytest.fixture
    def real_keys_values(self):
        """Create a real KeysValues object similar to what world_model uses."""
        # Simulate a batch of keys and values for a transformer
        batch_size = 4
        seq_len = 10
        num_heads = 8
        embed_dim = 512
        num_layers = 4
        max_tokens = 20
        device = torch.device('cpu')

        # Create KeysValues using the correct constructor
        keys_values = KeysValues(
            num_samples=batch_size,
            num_heads=num_heads,
            max_tokens=max_tokens,
            embed_dim=embed_dim,
            num_layers=num_layers,
            device=device
        )

        # Update with some initial data
        for layer in range(num_layers):
            keys = torch.randn(batch_size, num_heads, seq_len, embed_dim // num_heads)
            values = torch.randn(batch_size, num_heads, seq_len, embed_dim // num_heads)
            keys_values[layer].update(keys, values)

        return keys_values

    def test_basic_set_and_get_with_real_kv(self, mock_config, real_keys_values):
        """Test basic set and get operations with real KeysValues."""
        manager = KVCacheManager(config=mock_config, env_num=4, enable_stats=True)

        # Generate a cache key (simulate hash_state)
        cache_key = hash(torch.randn(10).numpy().tobytes())

        # Set cache
        index = manager.set_init_cache(env_id=0, cache_key=cache_key, kv_cache=real_keys_values)
        assert isinstance(index, int)
        assert 0 <= index < 100

        # Get cache
        retrieved = manager.get_init_cache(env_id=0, cache_key=cache_key)
        assert retrieved is not None
        assert isinstance(retrieved, KeysValues)

        # Verify the structure (KeysValues is a collection of KVCache objects)
        assert len(retrieved) == len(real_keys_values)
        assert retrieved.size == real_keys_values.size

    def test_cache_key_generation_consistency(self, mock_config, real_keys_values):
        """Test that cache keys generated from latent states are consistent."""
        manager = KVCacheManager(config=mock_config, env_num=4, enable_stats=True)

        # Simulate latent state
        latent_state = torch.randn(4, 768)

        # Generate cache key (simulate hash_state function)
        def hash_state(state):
            """Simulate the hash_state function from world_model."""
            if isinstance(state, torch.Tensor):
                state_np = state.detach().cpu().numpy()
            else:
                state_np = state
            return hash(state_np.tobytes())

        cache_key_1 = hash_state(latent_state)
        cache_key_2 = hash_state(latent_state)

        # Same state should produce same key
        assert cache_key_1 == cache_key_2

        # Store and retrieve
        manager.set_init_cache(env_id=0, cache_key=cache_key_1, kv_cache=real_keys_values)
        retrieved_1 = manager.get_init_cache(env_id=0, cache_key=cache_key_1)
        retrieved_2 = manager.get_init_cache(env_id=0, cache_key=cache_key_2)

        assert retrieved_1 is not None
        assert retrieved_2 is not None
        assert retrieved_1 is retrieved_2  # Should be the same object

    def test_multi_environment_isolation(self, mock_config, real_keys_values):
        """Test that caches from different environments are isolated."""
        manager = KVCacheManager(config=mock_config, env_num=4, enable_stats=True)

        # Create different KeysValues for different envs
        kv_env0 = real_keys_values
        kv_env1 = KeysValues(
            num_samples=4,
            num_heads=8,
            max_tokens=20,
            embed_dim=512,
            num_layers=4,
            device=torch.device('cpu')
        )
        # Update with different values
        for layer in range(4):
            keys = torch.randn(4, 8, 10, 64) * 2.0  # Different values
            values = torch.randn(4, 8, 10, 64) * 2.0
            kv_env1[layer].update(keys, values)

        cache_key = 12345

        # Set different caches for different environments
        manager.set_init_cache(env_id=0, cache_key=cache_key, kv_cache=kv_env0)
        manager.set_init_cache(env_id=1, cache_key=cache_key, kv_cache=kv_env1)

        # Retrieve and verify isolation
        retrieved_env0 = manager.get_init_cache(env_id=0, cache_key=cache_key)
        retrieved_env1 = manager.get_init_cache(env_id=1, cache_key=cache_key)

        assert retrieved_env0 is kv_env0
        assert retrieved_env1 is kv_env1
        assert retrieved_env0 is not retrieved_env1

    def test_cache_fallback_pattern(self, mock_config, real_keys_values):
        """Test the init->recur fallback pattern used in world_model."""
        manager = KVCacheManager(config=mock_config, env_num=4, enable_stats=True)

        cache_key = 99999

        # Scenario 1: Cache exists in init cache
        manager.set_init_cache(env_id=0, cache_key=cache_key, kv_cache=real_keys_values)

        # Try init cache first
        matched = manager.get_init_cache(env_id=0, cache_key=cache_key)
        assert matched is not None

        # Scenario 2: Cache doesn't exist in init, but exists in recur
        cache_key_2 = 88888
        kv_recur = KeysValues(
            num_samples=4,
            num_heads=8,
            max_tokens=20,
            embed_dim=512,
            num_layers=4,
            device=torch.device('cpu')
        )
        for layer in range(4):
            keys = torch.randn(4, 8, 10, 64)
            values = torch.randn(4, 8, 10, 64)
            kv_recur[layer].update(keys, values)

        manager.set_recur_cache(cache_key=cache_key_2, kv_cache=kv_recur)

        # Try init cache first (should miss)
        matched = manager.get_init_cache(env_id=0, cache_key=cache_key_2)
        assert matched is None

        # Fallback to recur cache
        matched = manager.get_recur_cache(cache_key=cache_key_2)
        assert matched is kv_recur

    def test_cache_eviction_with_real_data(self, mock_config):
        """Test cache eviction with realistic data volumes."""
        # Set small pool size to trigger eviction
        mock_config.game_segment_length = 5
        manager = KVCacheManager(config=mock_config, env_num=1, enable_stats=True)

        # Fill the pool beyond capacity
        stored_keys = []
        for i in range(10):
            kv = KeysValues(
                num_samples=2,
                num_heads=4,
                max_tokens=10,
                embed_dim=128,
                num_layers=2,
                device=torch.device('cpu')
            )
            for layer in range(2):
                keys = torch.randn(2, 4, 5, 32)
                values = torch.randn(2, 4, 5, 32)
                kv[layer].update(keys, values)

            cache_key = 1000 + i
            stored_keys.append(cache_key)
            manager.set_init_cache(env_id=0, cache_key=cache_key, kv_cache=kv)

        # First 5 should be evicted (FIFO)
        for i in range(5):
            retrieved = manager.get_init_cache(env_id=0, cache_key=stored_keys[i])
            assert retrieved is None  # Evicted

        # Last 5 should still be there
        for i in range(5, 10):
            retrieved = manager.get_init_cache(env_id=0, cache_key=stored_keys[i])
            assert retrieved is not None  # Still cached

    def test_statistics_tracking(self, mock_config, real_keys_values):
        """Test that statistics are correctly tracked."""
        manager = KVCacheManager(config=mock_config, env_num=2, enable_stats=True)

        cache_key = 7777
        manager.set_init_cache(env_id=0, cache_key=cache_key, kv_cache=real_keys_values)

        # Initial state
        stats = manager.get_stats_summary()
        assert stats['stats_enabled'] is True

        # Generate some hits and misses
        manager.get_init_cache(env_id=0, cache_key=cache_key)  # Hit
        manager.get_init_cache(env_id=0, cache_key=cache_key)  # Hit
        manager.get_init_cache(env_id=0, cache_key=9999)  # Miss

        # Check stats
        pool_stats = manager.init_pools[0].stats
        assert pool_stats.hits == 2
        assert pool_stats.misses == 1
        assert pool_stats.total_queries == 3
        assert abs(pool_stats.hit_rate - 2/3) < 1e-10

    def test_clear_operations(self, mock_config, real_keys_values):
        """Test various clear operations."""
        manager = KVCacheManager(config=mock_config, env_num=3, enable_stats=True)

        # Populate all cache types
        manager.set_init_cache(env_id=0, cache_key=1, kv_cache=real_keys_values)
        manager.set_init_cache(env_id=1, cache_key=2, kv_cache=real_keys_values)
        manager.set_recur_cache(cache_key=3, kv_cache=real_keys_values)
        manager.set_wm_cache(cache_key=4, kv_cache=real_keys_values)

        # Test selective clear
        manager.clear_init_caches()
        assert manager.get_init_cache(env_id=0, cache_key=1) is None
        assert manager.get_init_cache(env_id=1, cache_key=2) is None
        assert manager.get_recur_cache(cache_key=3) is not None  # Still there
        assert manager.get_wm_cache(cache_key=4) is not None  # Still there

        # Test clear all
        manager.set_init_cache(env_id=0, cache_key=1, kv_cache=real_keys_values)
        manager.clear_all()
        assert manager.get_init_cache(env_id=0, cache_key=1) is None
        assert manager.get_recur_cache(cache_key=3) is None
        assert manager.get_wm_cache(cache_key=4) is None


class TestCacheCopySemantics:
    """Test that cache copy semantics match the original implementation."""

    @pytest.fixture
    def real_keys_values(self):
        """Create a KeysValues object."""
        kv = KeysValues(
            num_samples=4,
            num_heads=8,
            max_tokens=20,
            embed_dim=512,
            num_layers=2,
            device=torch.device('cpu')
        )
        for layer in range(2):
            keys = torch.randn(4, 8, 10, 64)
            values = torch.randn(4, 8, 10, 64)
            kv[layer].update(keys, values)
        return kv

    def test_reference_vs_copy(self, real_keys_values):
        """Test whether KVCachePool stores references or copies."""
        pool = KVCachePool(pool_size=10, name="test")

        cache_key = 123
        # Get the first layer's first key tensor before storing
        original_key_tensor = real_keys_values[0].get()[0]  # Returns (keys, values) tuple
        original_key_data = original_key_tensor.clone()

        # Store the KeysValues
        pool.set(cache_key, real_keys_values)

        # Retrieve it
        retrieved = pool.get(cache_key)

        # In the original implementation, it stores references
        # So modifying the original should affect the cached version
        # Note: We need to modify the actual cache tensor, not create a new one
        retrieved_key_tensor = retrieved[0].get()[0]

        # Verify they reference the same data
        assert retrieved is real_keys_values

    def test_update_existing_entry(self, real_keys_values):
        """Test updating an existing cache entry."""
        pool = KVCachePool(pool_size=10, name="test")

        cache_key = 456

        # First set
        kv1 = real_keys_values
        index1 = pool.set(cache_key, kv1)

        # Create a different KeysValues
        kv2 = KeysValues(
            num_samples=4,
            num_heads=8,
            max_tokens=20,
            embed_dim=512,
            num_layers=2,
            device=torch.device('cpu')
        )
        for layer in range(2):
            keys = torch.ones(4, 8, 10, 64) * 100  # Different values
            values = torch.ones(4, 8, 10, 64) * 100
            kv2[layer].update(keys, values)

        # Update the same key
        index2 = pool.set(cache_key, kv2)

        # Should use the same index
        assert index1 == index2

        # Should have the new value
        retrieved = pool.get(cache_key)
        assert retrieved is kv2
        # Verify the data is from kv2
        retrieved_keys, retrieved_values = retrieved[0].get()
        expected_keys, expected_values = kv2[0].get()
        assert torch.allclose(retrieved_keys, expected_keys)


class TestRealisticWorkflow:
    """Test realistic workflows from world_model usage patterns."""

    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.game_segment_length = 50
        config.num_simulations = 25
        return config

    def test_imagine_function_workflow(self, mock_config):
        """Simulate the workflow in the imagine function."""
        manager = KVCacheManager(config=mock_config, env_num=4, enable_stats=True)

        # Simulate a batch of observations being processed
        batch_size = 4

        for i in range(batch_size):
            # Simulate encoding an observation to get a latent state
            latent_state = torch.randn(768)
            cache_key = hash(latent_state.numpy().tobytes())

            # Try to retrieve from init cache
            matched_kv = manager.get_init_cache(env_id=i, cache_key=cache_key)

            if matched_kv is None:
                # Cache miss: try recurrent cache
                matched_kv = manager.get_recur_cache(cache_key=cache_key)

                if matched_kv is None:
                    # Generate new KV
                    new_kv = KeysValues(
                        num_samples=1,
                        num_heads=8,
                        max_tokens=20,
                        embed_dim=512,
                        num_layers=4,
                        device=torch.device('cpu')
                    )
                    for layer in range(4):
                        keys = torch.randn(1, 8, 10, 64)
                        values = torch.randn(1, 8, 10, 64)
                        new_kv[layer].update(keys, values)

                    # Store in init cache for this environment
                    manager.set_init_cache(env_id=i, cache_key=cache_key, kv_cache=new_kv)
                    matched_kv = new_kv

            # At this point, matched_kv should always be available
            assert matched_kv is not None

    def test_mcts_search_workflow(self, mock_config):
        """Simulate MCTS search using recurrent cache."""
        manager = KVCacheManager(config=mock_config, env_num=4, enable_stats=True)

        # Simulate multiple MCTS simulations
        num_simulations = 20

        for sim_id in range(num_simulations):
            # Each simulation generates a state
            state = torch.randn(768)
            cache_key = hash(state.numpy().tobytes())

            # Try to retrieve from recurrent cache
            kv = manager.get_recur_cache(cache_key=cache_key)

            if kv is None:
                # Generate new KV
                kv = KeysValues(
                    num_samples=1,
                    num_heads=8,
                    max_tokens=10,
                    embed_dim=256,
                    num_layers=4,
                    device=torch.device('cpu')
                )
                for layer in range(4):
                    keys = torch.randn(1, 8, 5, 32)
                    values = torch.randn(1, 8, 5, 32)
                    kv[layer].update(keys, values)

                # Store in recurrent cache
                manager.set_recur_cache(cache_key=cache_key, kv_cache=kv)

        # Check that we have some cache hits from repeated states
        stats = manager.recur_pool.stats
        assert stats.total_queries == num_simulations
        # In a real scenario, some states would be revisited, leading to hits


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
