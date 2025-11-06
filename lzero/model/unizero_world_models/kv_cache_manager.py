"""
KV Cache Manager for UniZero World Model
=========================================

This module provides a unified, robust, and extensible KV cache management system
for the UniZero world model. It replaces the scattered cache logic with a clean,
well-tested abstraction.

Author: Claude Code
Date: 2025-10-24
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import torch
from collections import OrderedDict

# Assuming kv_caching is in the same directory or accessible
from .kv_caching import KeysValues


logger = logging.getLogger(__name__)


class EvictionStrategy(Enum):
    """Cache eviction strategies."""
    FIFO = "fifo"  # First In First Out (循环覆盖)
    LRU = "lru"    # Least Recently Used
    PRIORITY = "priority"  # 基于优先级


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_queries: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        if self.total_queries == 0:
            return 0.0
        return self.hits / self.total_queries

    @property
    def miss_rate(self) -> float:
        """Calculate miss rate."""
        return 1.0 - self.hit_rate

    def reset(self):
        """Reset all statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_queries = 0

    def __repr__(self) -> str:
        return (f"CacheStats(hits={self.hits}, misses={self.misses}, "
                f"evictions={self.evictions}, hit_rate={self.hit_rate:.2%})")


class KVCachePool:
    """
    A fixed-size pool for storing KeysValues objects.

    This class manages a pre-allocated pool of KeysValues objects and provides
    efficient storage and retrieval mechanisms with configurable eviction strategies.

    Args:
        pool_size: Maximum number of KV caches to store
        eviction_strategy: Strategy for cache eviction
        enable_stats: Whether to collect statistics
        name: Name for this cache pool (for logging)
    """

    def __init__(
        self,
        pool_size: int,
        eviction_strategy: EvictionStrategy = EvictionStrategy.FIFO,
        enable_stats: bool = True,
        name: str = "default"
    ):
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {pool_size}")

        self.pool_size = pool_size
        self.eviction_strategy = eviction_strategy
        self.enable_stats = enable_stats
        self.name = name

        # Core data structures
        self._pool: List[Optional[KeysValues]] = [None] * pool_size
        self._key_to_index: Dict[int, int] = {}  # cache_key -> pool_index
        self._index_to_key: List[Optional[int]] = [None] * pool_size  # pool_index -> cache_key

        # Eviction strategy specific data
        self._next_index: int = 0  # For FIFO
        self._access_order: OrderedDict = OrderedDict()  # For LRU
        self._priorities: Dict[int, float] = {}  # For PRIORITY

        # Statistics
        self.stats = CacheStats() if enable_stats else None

        logger.info(f"Initialized KVCachePool '{name}' with size={pool_size}, "
                    f"strategy={eviction_strategy.value}")

    def get(self, cache_key: int) -> Optional[KeysValues]:
        """
        Retrieve a cached KeysValues object.

        Args:
            cache_key: The hash key for the cache

        Returns:
            The cached KeysValues object if found, None otherwise
        """
        if self.enable_stats:
            self.stats.total_queries += 1

        pool_index = self._key_to_index.get(cache_key)

        if pool_index is not None:
            # Cache hit
            if self.enable_stats:
                self.stats.hits += 1

            # Update access order for LRU
            if self.eviction_strategy == EvictionStrategy.LRU:
                self._access_order.move_to_end(cache_key)

            logger.debug(f"[{self.name}] Cache HIT for key={cache_key}, index={pool_index}")
            return self._pool[pool_index]
        else:
            # Cache miss
            if self.enable_stats:
                self.stats.misses += 1

            logger.debug(f"[{self.name}] Cache MISS for key={cache_key}")
            return None

    def set(self, cache_key: int, kv_cache: KeysValues) -> int:
        """
        Store a KeysValues object in the cache.

        Args:
            cache_key: The hash key for the cache
            kv_cache: The KeysValues object to store

        Returns:
            The pool index where the cache was stored
        """
        # ==================== BUG FIX: Defensive Deep Copy ====================
        # CRITICAL: Always clone the input to prevent cache corruption.
        # This provides an additional layer of protection in case the caller
        # forgets to clone. The clone operation ensures that the stored cache
        # is independent from the caller's object, preventing unintended mutations.
        kv_cache_copy = kv_cache.clone()
        # =======================================================================

        # Check if key already exists
        if cache_key in self._key_to_index:
            # Update existing entry
            pool_index = self._key_to_index[cache_key]
            self._pool[pool_index] = kv_cache_copy  # Store cloned copy

            if self.eviction_strategy == EvictionStrategy.LRU:
                self._access_order.move_to_end(cache_key)

            logger.debug(f"[{self.name}] Updated cache for key={cache_key} at index={pool_index}")
            return pool_index

        # Find a slot for new entry
        pool_index = self._find_slot_for_new_entry(cache_key)

        # Evict old entry if necessary
        old_key = self._index_to_key[pool_index]
        if old_key is not None:
            self._evict(old_key, pool_index)

        # Store new entry (already cloned above)
        self._pool[pool_index] = kv_cache_copy
        self._key_to_index[cache_key] = pool_index
        self._index_to_key[pool_index] = cache_key

        # Update access tracking for LRU
        if self.eviction_strategy == EvictionStrategy.LRU:
            self._access_order[cache_key] = True

        logger.debug(f"[{self.name}] Stored cache for key={cache_key} at index={pool_index}")
        return pool_index

    def _find_slot_for_new_entry(self, cache_key: int) -> int:
        """Find an appropriate slot for a new cache entry based on eviction strategy."""
        if self.eviction_strategy == EvictionStrategy.FIFO:
            # Simple circular buffer
            pool_index = self._next_index
            self._next_index = (self._next_index + 1) % self.pool_size
            return pool_index

        elif self.eviction_strategy == EvictionStrategy.LRU:
            # Find LRU slot
            if len(self._key_to_index) < self.pool_size:
                # Pool not full, find first empty slot
                for i in range(self.pool_size):
                    if self._index_to_key[i] is None:
                        return i

            # Evict LRU (first item in OrderedDict)
            lru_key = next(iter(self._access_order))
            return self._key_to_index[lru_key]

        elif self.eviction_strategy == EvictionStrategy.PRIORITY:
            # Find lowest priority slot
            if len(self._key_to_index) < self.pool_size:
                # Pool not full
                for i in range(self.pool_size):
                    if self._index_to_key[i] is None:
                        return i

            # Evict lowest priority
            min_priority_key = min(self._priorities, key=self._priorities.get)
            return self._key_to_index[min_priority_key]

        else:
            raise ValueError(f"Unknown eviction strategy: {self.eviction_strategy}")

    def _evict(self, cache_key: int, pool_index: int):
        """Evict a cache entry."""
        if self.enable_stats:
            self.stats.evictions += 1

        # Remove from tracking structures
        del self._key_to_index[cache_key]
        self._index_to_key[pool_index] = None

        if self.eviction_strategy == EvictionStrategy.LRU:
            self._access_order.pop(cache_key, None)

        if self.eviction_strategy == EvictionStrategy.PRIORITY:
            self._priorities.pop(cache_key, None)

        logger.debug(f"[{self.name}] Evicted key={cache_key} from index={pool_index}")

    def clear(self):
        """Clear all cache entries."""
        self._pool = [None] * self.pool_size
        self._key_to_index.clear()
        self._index_to_key = [None] * self.pool_size
        self._next_index = 0
        self._access_order.clear()
        self._priorities.clear()

        if self.enable_stats:
            # Don't reset stats on clear, user can call stats.reset() explicitly
            pass

        # logger.info(f"[{self.name}] Cleared all cache entries")

    def __len__(self) -> int:
        """Return the number of cached entries."""
        return len(self._key_to_index)

    def __repr__(self) -> str:
        stats_str = f", {self.stats}" if self.enable_stats else ""
        return (f"KVCachePool(name='{self.name}', size={len(self)}/{self.pool_size}, "
                f"strategy={self.eviction_strategy.value}{stats_str})")


class KVCacheManager:
    """
    Unified KV Cache Manager for World Model.

    This class manages multiple cache pools for different inference scenarios:
    - Initial inference caches (per-environment)
    - Recurrent inference caches (for MCTS)
    - World model caches (temporary batch caches)

    Args:
        config: World model configuration
        env_num: Number of environments
        enable_stats: Whether to enable statistics collection
        clear_recur_log_freq: How often to log 'clear_recur_cache' calls.
        clear_all_log_freq: How often to log 'clear_all' calls.
    """

    def __init__(
        self,
        config,
        env_num: int,
        enable_stats: bool = True,
        clear_recur_log_freq: int = 1000, # <--- RENAMED & MODIFIED
        clear_all_log_freq: int = 100     # <--- NEW
    ):
        self.config = config
        self.env_num = env_num
        self.enable_stats = enable_stats
        
        # --- Throttling parameters and counters ---
        self.clear_recur_log_freq = clear_recur_log_freq
        self.clear_all_log_freq = clear_all_log_freq
        self._clear_recur_counter = 0
        self._clear_all_counter = 0  # <--- NEW

        # Initialize cache pools
        self._init_cache_pools()

        # These lists store KeysValues objects, not integers
        # Used in world model's trim_and_pad_kv_cache for batch processing
        self.keys_values_wm_list: List[KeysValues] = []
        self.keys_values_wm_size_list: List[int] = []

        logger.info(f"Initialized KVCacheManager for {env_num} environments")

    def _init_cache_pools(self):
        """Initialize all cache pools."""
        # Initial inference pools (one per environment)
        init_pool_size = int(self.config.game_segment_length)
        self.init_pools: List[KVCachePool] = []
        for env_id in range(self.env_num):
            pool = KVCachePool(
                pool_size=init_pool_size,
                eviction_strategy=EvictionStrategy.FIFO,
                enable_stats=self.enable_stats,
                name=f"init_env{env_id}"
            )
            self.init_pools.append(pool)

        # Recurrent inference pool (shared across all environments)
        num_simulations = getattr(self.config, 'num_simulations', 50)
        recur_pool_size = int(num_simulations * self.env_num)
        self.recur_pool = KVCachePool(
            pool_size=recur_pool_size,
            eviction_strategy=EvictionStrategy.FIFO,
            enable_stats=self.enable_stats,
            name="recurrent"
        )

        # World model pool (temporary)
        wm_pool_size = self.env_num
        self.wm_pool = KVCachePool(
            pool_size=wm_pool_size,
            eviction_strategy=EvictionStrategy.FIFO,
            enable_stats=self.enable_stats,
            name="world_model"
        )

    def get_init_cache(self, env_id: int, cache_key: int) -> Optional[KeysValues]:
        """Get cache from initial inference pool."""
        if env_id < 0 or env_id >= self.env_num:
            raise ValueError(f"Invalid env_id: {env_id}, must be in [0, {self.env_num})")
        return self.init_pools[env_id].get(cache_key)

    def set_init_cache(self, env_id: int, cache_key: int, kv_cache: KeysValues) -> int:
        """Set cache in initial inference pool."""
        if env_id < 0 or env_id >= self.env_num:
            raise ValueError(f"Invalid env_id: {env_id}, must be in [0, {self.env_num})")
        return self.init_pools[env_id].set(cache_key, kv_cache)

    def get_recur_cache(self, cache_key: int) -> Optional[KeysValues]:
        """Get cache from recurrent inference pool."""
        return self.recur_pool.get(cache_key)

    def set_recur_cache(self, cache_key: int, kv_cache: KeysValues) -> int:
        """Set cache in recurrent inference pool."""
        return self.recur_pool.set(cache_key, kv_cache)

    def get_wm_cache(self, cache_key: int) -> Optional[KeysValues]:
        """Get cache from world model pool."""
        return self.wm_pool.get(cache_key)

    def set_wm_cache(self, cache_key: int, kv_cache: KeysValues) -> int:
        """Set cache in world model pool."""
        return self.wm_pool.set(cache_key, kv_cache)

    def hierarchical_get(self, env_id: int, cache_key: int) -> Optional[KeysValues]:
        """
        Perform hierarchical cache lookup: init_pool -> recur_pool.

        This method encapsulates the two-level lookup strategy:
        1. First try to find in environment-specific init_infer cache
        2. If not found, fallback to global recurrent_infer cache

        Arguments:
            - env_id (:obj:`int`): Environment ID for init cache lookup
            - cache_key (:obj:`int`): Cache key to lookup

        Returns:
            - kv_cache (:obj:`Optional[KeysValues]`): Found cache or None
        """
        # Step 1: Try init_infer cache first (per-environment)
        kv_cache = self.get_init_cache(env_id, cache_key)
        if kv_cache is not None:
            return kv_cache

        # Step 2: If not found, try recurrent_infer cache (global)
        return self.get_recur_cache(cache_key)

    def clear_all(self): # <--- MODIFIED METHOD
        """Clear all cache pools, with throttled logging."""
        # Core clearing actions always execute.
        for pool in self.init_pools:
            pool.clear()
        self.recur_pool.clear()
        self.wm_pool.clear()
        self.keys_values_wm_list.clear()
        self.keys_values_wm_size_list.clear()
        
        # --- Throttled Logging Logic ---
        self._clear_all_counter += 1
        if self.clear_all_log_freq > 0 and self._clear_all_counter % self.clear_all_log_freq == 0:
            logger.info(
                f"Cleared all KV caches (this message appears every "
                f"{self.clear_all_log_freq} calls, total calls: {self._clear_all_counter})"
            )

    def clear_init_caches(self):
        """Clear only initial inference caches."""
        for pool in self.init_pools:
            pool.clear()
        logger.info("Cleared initial inference caches")

    def clear_recur_cache(self):
        """Clear only recurrent inference cache, with throttled logging."""
        # The core cache clearing action always executes.
        self.recur_pool.clear()
        
        # --- Throttled Logging Logic ---
        self._clear_recur_counter += 1
        # Only log if frequency is positive and the counter is a multiple of the frequency.
        if self.clear_recur_log_freq > 0 and self._clear_recur_counter % self.clear_recur_log_freq == 0:
            logger.info(
                f"Cleared recurrent inference cache (this message appears every "
                f"{self.clear_recur_log_freq} calls, total calls: {self._clear_recur_counter})"
            )

    def get_stats_summary(self) -> Dict[str, Any]:
        """Get statistics summary for all pools."""
        if not self.enable_stats:
            return {"stats_enabled": False}

        summary = {
            "stats_enabled": True,
            "init_pools": {},
            "recur_pool": str(self.recur_pool.stats),
            "wm_pool": str(self.wm_pool.stats),
        }

        for env_id, pool in enumerate(self.init_pools):
            summary["init_pools"][f"env_{env_id}"] = str(pool.stats)

        return summary

    def reset_stats(self):
        """Reset statistics for all pools."""
        if not self.enable_stats:
            return

        for pool in self.init_pools:
            pool.stats.reset()
        self.recur_pool.stats.reset()
        self.wm_pool.stats.reset()
        logger.info("Reset all cache statistics")

    def __repr__(self) -> str:
        init_sizes = [len(pool) for pool in self.init_pools]
        return (f"KVCacheManager(env_num={self.env_num}, "
                f"init_caches={init_sizes}, "
                f"recur_cache={len(self.recur_pool)}/{self.recur_pool.pool_size}, "
                f"wm_cache={len(self.wm_pool)}/{self.wm_pool.pool_size})")