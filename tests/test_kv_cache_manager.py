"""
Unit tests for KV Cache Manager
================================

Tests to ensure the correctness of the refactored KV cache management system.

Run with: pytest test_kv_cache_manager.py -v
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock
from lzero.model.unizero_world_models.kv_cache_manager import (
    KVCachePool,
    KVCacheManager,
    EvictionStrategy,
    CacheStats,
)


# Mock KeysValues for testing
class MockKeysValues:
    """Mock KeysValues object for testing."""
    def __init__(self, value: int):
        self.value = value

    def __eq__(self, other):
        if not isinstance(other, MockKeysValues):
            return False
        return self.value == other.value


class TestCacheStats:
    """Test CacheStats class."""

    def test_initialization(self):
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.total_queries == 0

    def test_hit_rate_calculation(self):
        stats = CacheStats()
        stats.hits = 7
        stats.misses = 3
        stats.total_queries = 10
        assert stats.hit_rate == 0.7
        assert abs(stats.miss_rate - 0.3) < 1e-10  # Use approximate comparison for floating point

    def test_hit_rate_zero_queries(self):
        stats = CacheStats()
        assert stats.hit_rate == 0.0
        assert stats.miss_rate == 1.0

    def test_reset(self):
        stats = CacheStats(hits=10, misses=5, evictions=2, total_queries=15)
        stats.reset()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.total_queries == 0


class TestKVCachePool:
    """Test KVCachePool class."""

    def test_initialization(self):
        pool = KVCachePool(pool_size=10, name="test")
        assert pool.pool_size == 10
        assert len(pool) == 0
        assert pool.name == "test"

    def test_invalid_pool_size(self):
        with pytest.raises(ValueError):
            KVCachePool(pool_size=0)

        with pytest.raises(ValueError):
            KVCachePool(pool_size=-5)

    def test_simple_set_and_get(self):
        pool = KVCachePool(pool_size=5)
        kv = MockKeysValues(42)
        cache_key = 123

        # Set
        index = pool.set(cache_key, kv)
        assert isinstance(index, int)
        assert 0 <= index < 5

        # Get
        retrieved = pool.get(cache_key)
        assert retrieved == kv
        assert len(pool) == 1

    def test_cache_miss(self):
        pool = KVCachePool(pool_size=5)
        result = pool.get(999)  # Non-existent key
        assert result is None

    def test_cache_update(self):
        pool = KVCachePool(pool_size=5)
        cache_key = 100

        kv1 = MockKeysValues(1)
        kv2 = MockKeysValues(2)

        # First set
        index1 = pool.set(cache_key, kv1)
        assert pool.get(cache_key) == kv1

        # Update same key
        index2 = pool.set(cache_key, kv2)
        assert index1 == index2  # Same index
        assert pool.get(cache_key) == kv2  # Updated value
        assert len(pool) == 1  # Still one entry

    def test_fifo_eviction(self):
        pool = KVCachePool(pool_size=3, eviction_strategy=EvictionStrategy.FIFO)

        # Fill the pool
        pool.set(1, MockKeysValues(1))
        pool.set(2, MockKeysValues(2))
        pool.set(3, MockKeysValues(3))
        assert len(pool) == 3

        # Add fourth item, should evict first
        pool.set(4, MockKeysValues(4))
        assert len(pool) == 3
        assert pool.get(1) is None  # Evicted
        assert pool.get(2) is not None
        assert pool.get(3) is not None
        assert pool.get(4) is not None

    def test_lru_eviction(self):
        pool = KVCachePool(pool_size=3, eviction_strategy=EvictionStrategy.LRU)

        # Fill the pool
        pool.set(1, MockKeysValues(1))
        pool.set(2, MockKeysValues(2))
        pool.set(3, MockKeysValues(3))

        # Access key 1 to make it recently used
        pool.get(1)

        # Add fourth item, should evict LRU (key 2)
        pool.set(4, MockKeysValues(4))
        assert pool.get(2) is None  # Evicted (LRU)
        assert pool.get(1) is not None  # Still there
        assert pool.get(3) is not None
        assert pool.get(4) is not None

    def test_statistics_collection(self):
        pool = KVCachePool(pool_size=5, enable_stats=True)

        # Cache miss
        pool.get(999)
        assert pool.stats.misses == 1
        assert pool.stats.hits == 0
        assert pool.stats.total_queries == 1

        # Cache hit
        pool.set(100, MockKeysValues(1))
        pool.get(100)
        assert pool.stats.hits == 1
        assert pool.stats.total_queries == 2

        # Eviction
        for i in range(6):  # Overflow pool
            pool.set(i, MockKeysValues(i))
        assert pool.stats.evictions >= 1

    def test_clear(self):
        pool = KVCachePool(pool_size=5)
        pool.set(1, MockKeysValues(1))
        pool.set(2, MockKeysValues(2))
        assert len(pool) == 2

        pool.clear()
        assert len(pool) == 0
        assert pool.get(1) is None
        assert pool.get(2) is None


class TestKVCacheManager:
    """Test KVCacheManager class."""

    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.game_segment_length = 10
        config.num_simulations = 50
        return config

    @pytest.fixture
    def manager(self, mock_config):
        return KVCacheManager(config=mock_config, env_num=3, enable_stats=True)

    def test_initialization(self, manager):
        assert manager.env_num == 3
        assert len(manager.init_pools) == 3
        assert manager.recur_pool is not None
        assert manager.wm_pool is not None

    def test_init_cache_operations(self, manager):
        kv = MockKeysValues(42)
        cache_key = 123

        # Set in env 0
        manager.set_init_cache(env_id=0, cache_key=cache_key, kv_cache=kv)

        # Get from env 0
        retrieved = manager.get_init_cache(env_id=0, cache_key=cache_key)
        assert retrieved == kv

        # Should not exist in env 1
        assert manager.get_init_cache(env_id=1, cache_key=cache_key) is None

    def test_invalid_env_id(self, manager):
        kv = MockKeysValues(1)

        with pytest.raises(ValueError):
            manager.set_init_cache(env_id=-1, cache_key=1, kv_cache=kv)

        with pytest.raises(ValueError):
            manager.set_init_cache(env_id=999, cache_key=1, kv_cache=kv)

        with pytest.raises(ValueError):
            manager.get_init_cache(env_id=-1, cache_key=1)

    def test_recur_cache_operations(self, manager):
        kv = MockKeysValues(99)
        cache_key = 456

        manager.set_recur_cache(cache_key=cache_key, kv_cache=kv)
        retrieved = manager.get_recur_cache(cache_key=cache_key)
        assert retrieved == kv

    def test_wm_cache_operations(self, manager):
        kv = MockKeysValues(77)
        cache_key = 789

        manager.set_wm_cache(cache_key=cache_key, kv_cache=kv)
        retrieved = manager.get_wm_cache(cache_key=cache_key)
        assert retrieved == kv

    def test_clear_all(self, manager):
        # Populate all caches
        manager.set_init_cache(0, 1, MockKeysValues(1))
        manager.set_recur_cache(2, MockKeysValues(2))
        manager.set_wm_cache(3, MockKeysValues(3))

        # Clear all
        manager.clear_all()

        # Verify all cleared
        assert manager.get_init_cache(0, 1) is None
        assert manager.get_recur_cache(2) is None
        assert manager.get_wm_cache(3) is None

    def test_clear_specific_caches(self, manager):
        manager.set_init_cache(0, 1, MockKeysValues(1))
        manager.set_recur_cache(2, MockKeysValues(2))

        # Clear only init caches
        manager.clear_init_caches()
        assert manager.get_init_cache(0, 1) is None
        assert manager.get_recur_cache(2) is not None

        # Clear recur cache
        manager.clear_recur_cache()
        assert manager.get_recur_cache(2) is None

    def test_stats_collection(self, manager):
        # Perform some operations
        manager.set_init_cache(0, 1, MockKeysValues(1))
        manager.get_init_cache(0, 1)  # Hit
        manager.get_init_cache(0, 999)  # Miss

        # Get stats
        stats = manager.get_stats_summary()
        assert stats["stats_enabled"] is True
        assert "init_pools" in stats
        assert "env_0" in stats["init_pools"]

    def test_stats_reset(self, manager):
        # Generate some stats
        manager.get_init_cache(0, 999)  # Miss
        assert manager.init_pools[0].stats.misses > 0

        # Reset
        manager.reset_stats()
        assert manager.init_pools[0].stats.misses == 0


# Integration test
class TestIntegration:
    """Integration tests with more realistic scenarios."""

    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.game_segment_length = 100
        config.num_simulations = 50
        return config

    def test_realistic_workflow(self, mock_config):
        manager = KVCacheManager(config=mock_config, env_num=4)

        # Simulate episode collection
        for env_id in range(4):
            for step in range(50):
                cache_key = hash((env_id, step))
                kv = MockKeysValues(step)
                manager.set_init_cache(env_id, cache_key, kv)

        # Simulate MCTS search
        for sim in range(20):
            cache_key = hash(("mcts", sim))
            kv = MockKeysValues(sim)
            manager.set_recur_cache(cache_key, kv)

        # Verify retrieval
        cache_key = hash((0, 10))
        retrieved = manager.get_init_cache(0, cache_key)
        assert retrieved is not None
        assert retrieved.value == 10

        # Check stats
        stats = manager.get_stats_summary()
        assert stats["stats_enabled"] is True

    def test_cache_overflow_behavior(self, mock_config):
        # Small pool to trigger overflow
        mock_config.game_segment_length = 5
        manager = KVCacheManager(config=mock_config, env_num=1)

        # Add more items than pool size
        for i in range(10):
            manager.set_init_cache(0, i, MockKeysValues(i))

        # First items should be evicted
        assert manager.get_init_cache(0, 0) is None
        assert manager.get_init_cache(0, 9) is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
