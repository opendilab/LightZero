# KV Cache 重构指南

## 目标
将现有的分散的KV cache管理逻辑重构为统一的、可测试的、可维护的系统。

## 前期准备

### 1. 备份已完成 ✅
```bash
cp world_model.py world_model.py.backup_20251023_143124
```

### 2. 新模块已创建 ✅
- `kv_cache_manager.py`: 核心管理类
- `test_kv_cache_manager.py`: 单元测试 (24/24 通过)

## 重构步骤

### Step 1: 导入新模块

在 `world_model.py` 顶部添加:
```python
from .kv_cache_manager import KVCacheManager
```

### Step 2: 替换初始化逻辑

#### 原代码 (第197-209行):
```python
def _initialize_cache_structures(self) -> None:
    """Initialize cache structures for past keys and values."""
    from collections import defaultdict

    self.past_kv_cache_recurrent_infer = {}
    self.pool_idx_to_key_map_recur_infer = [None] * self.shared_pool_size_recur
    self.past_kv_cache_init_infer_envs = [{} for _ in range(self.env_num)]
    self.pool_idx_to_key_map_init_envs = [[None] * self.shared_pool_size_init for _ in range(self.env_num)]

    self.keys_values_wm_list = []
    self.keys_values_wm_size_list = []
```

#### 新代码:
```python
def _initialize_cache_structures(self) -> None:
    """Initialize unified KV cache manager."""
    # 创建统一的cache manager
    self.kv_cache_manager = KVCacheManager(
        config=self.config,
        env_num=self.env_num,
        enable_stats=True  # 启用统计以便监控
    )

    # 保持向后兼容 (稍后会逐步移除)
    self.keys_values_wm_list = self.kv_cache_manager.keys_values_wm_list
    self.keys_values_wm_size_list = self.kv_cache_manager.keys_values_wm_size_list
```

### Step 3: 重构 cache 复制函数

#### 移除重复的复制函数

删除或标记为deprecated:
- `custom_copy_kv_cache_to_shared_init_envs` (第341-376行)
- `custom_copy_kv_cache_to_shared_recur` (第377-415行)
- `custom_copy_kv_cache_to_shared_wm` (第415-450行)

#### 创建统一的复制辅助函数:
```python
def _copy_kv_to_pool(self, src_kv: KeysValues, pool_type: str, env_id: Optional[int] = None) -> int:
    """
    统一的KV cache复制辅助函数

    Args:
        src_kv: 源KeysValues对象
        pool_type: "init", "recur", 或 "wm"
        env_id: 环境ID (仅对init类型需要)

    Returns:
        Pool中的索引
    """
    # 生成cache key
    # 注意: 这里需要根据实际的state生成key
    # 暂时返回dummy实现,实际使用时需要传入state
    raise NotImplementedError("需要实际的state来生成cache_key")
```

### Step 4: 重构 `imagine` 函数中的 cache 操作

#### 4.1 存储cache的位置 (约1422-1450行)

##### 原代码:
```python
# 3. 如果存在旧 key，就从主 cache map 中删除它
old_key_to_evict = self.pool_idx_to_key_map_init_envs[i][index_to_write]
if old_key_to_evict is not None:
    if old_key_to_evict in self.past_kv_cache_init_infer_envs[i]:
        del self.past_kv_cache_init_infer_envs[i][old_key_to_evict]

# Copy to shared pool
cache_index = self.custom_copy_kv_cache_to_shared_init_envs(self.keys_values_wm_single_env, i)

# 4. 在主 cache map 和辅助列表中同时更新新的映射关系
self.past_kv_cache_init_infer_envs[i][cache_key] = cache_index
self.pool_idx_to_key_map_init_envs[i][index_to_write] = cache_key
```

##### 新代码:
```python
# 使用统一的cache manager
self.kv_cache_manager.set_init_cache(
    env_id=i,
    cache_key=cache_key,
    kv_cache=self.keys_values_wm_single_env
)
```

#### 4.2 检索cache的位置 (约1455-1510行)

##### 原代码:
```python
cache_index = self.past_kv_cache_init_infer_envs[index].get(cache_key)
if cache_index is not None:
    matched_value = self.shared_pool_init_infer[index][cache_index]
    ...
# Fallback to recurrent cache
recur_cache_index = self.past_kv_cache_recurrent_infer.get(cache_key)
if recur_cache_index is not None:
    matched_value = self.shared_pool_recur_infer[recur_cache_index]
```

##### 新代码:
```python
# 首先尝试从init cache获取
matched_value = self.kv_cache_manager.get_init_cache(index, cache_key)

if matched_value is None:
    # 回退到recurrent cache
    matched_value = self.kv_cache_manager.get_recur_cache(cache_key)

    if matched_value is None:
        # Cache miss, 需要生成新的cache
        print(f"[CACHE MISS] key={cache_key}. Generating new cache.")
```

### Step 5: 重构 `clear_caches` 函数

#### 原代码 (第2181-2189行):
```python
def clear_caches(self):
    """Clears the caches of the world model."""
    for kv_cache_dict_env in self.past_kv_cache_init_infer_envs:
        kv_cache_dict_env.clear()
    self.past_kv_cache_recurrent_infer.clear()
    self.keys_values_wm_list.clear()
    print(f'Cleared {self.__class__.__name__} past_kv_cache.')
```

#### 新代码:
```python
def clear_caches(self):
    """Clears all KV caches of the world model."""
    self.kv_cache_manager.clear_all()
    print(f'Cleared {self.__class__.__name__} KV caches.')

    # 可选: 打印统计信息
    if hasattr(self.kv_cache_manager, 'get_stats_summary'):
        stats = self.kv_cache_manager.get_stats_summary()
        print(f'Cache stats before clear: {stats}')
        self.kv_cache_manager.reset_stats()
```

### Step 6: 添加监控和日志

#### 在适当位置添加cache性能监控:
```python
def _log_cache_stats(self):
    """定期记录cache统计信息"""
    if self.config.enable_cache_logging:
        stats = self.kv_cache_manager.get_stats_summary()
        logger.info(f"KV Cache Stats: {stats}")
```

#### 在training loop中调用:
```python
# 例如每1000步记录一次
if global_step % 1000 == 0:
    self._log_cache_stats()
```

## 详细映射表

### 数据结构映射

| 旧结构 | 新结构 | 说明 |
|--------|--------|------|
| `past_kv_cache_init_infer_envs` | `kv_cache_manager.init_pools[env_id]._key_to_index` | Per-env字典→Pool管理 |
| `past_kv_cache_recurrent_infer` | `kv_cache_manager.recur_pool._key_to_index` | 全局字典→Pool管理 |
| `shared_pool_init_infer` | `kv_cache_manager.init_pools[env_id]._pool` | Pool实现 |
| `shared_pool_recur_infer` | `kv_cache_manager.recur_pool._pool` | Pool实现 |
| `pool_idx_to_key_map_*` | 内部于Pool中管理 | 简化接口 |

### 函数映射

| 旧函数 | 新函数 | 说明 |
|--------|--------|------|
| `custom_copy_kv_cache_to_shared_init_envs` | `kv_cache_manager.set_init_cache` | 统一接口 |
| `custom_copy_kv_cache_to_shared_recur` | `kv_cache_manager.set_recur_cache` | 统一接口 |
| `custom_copy_kv_cache_to_shared_wm` | `kv_cache_manager.set_wm_cache` | 统一接口 |
| 手动检索逻辑 | `kv_cache_manager.get_*_cache` | 封装 |

## 注意事项

### ⚠️ 关键兼容性问题

1. **KeysValues 复制逻辑**:
   - 现有代码有复杂的 `copy_()` 操作
   - 需要确保新系统保持相同的复制语义
   - 建议: 在KVCacheManager内部处理复制细节

2. **Cache Key 生成**:
   - 现有使用 `hash_state(state)`
   - 需要确保hash函数一致性
   - 建议: 将hash逻辑集成到KVCacheManager

3. **Position Embedding 调整**:
   - 现有代码在cache trimming时调整position embedding
   - 这部分逻辑可能需要保留在外部
   - 建议: 先保持分离,后续优化

4. **多线程/多进程**:
   - 如果有并发访问,需要添加锁
   - 当前实现假设单线程
   - 建议: 如需要,使用 `threading.Lock`

### 🔍 测试检查点

在每个步骤后,运行以下测试:

```bash
# 1. 单元测试
pytest tests/test_kv_cache_manager.py -v

# 2. 集成测试 (需要创建)
pytest tests/test_world_model_integration.py -v

# 3. 性能测试
python benchmarks/benchmark_kv_cache.py

# 4. 功能一致性测试
python tests/compare_old_new_cache.py
```

### 📊 性能验证

创建benchmark脚本:
```python
# benchmarks/benchmark_kv_cache.py

import time
import torch

def benchmark_old_cache():
    # 使用backup版本
    ...

def benchmark_new_cache():
    # 使用重构版本
    ...

if __name__ == "__main__":
    old_time = benchmark_old_cache()
    new_time = benchmark_new_cache()

    print(f"Old: {old_time:.4f}s")
    print(f"New: {new_time:.4f}s")
    print(f"Speedup: {old_time/new_time:.2f}x")
```

## 回滚计划

如果重构出现问题:

```bash
# 1. 停止使用新代码
git stash

# 2. 恢复备份
cp world_model.py.backup_20251023_143124 world_model.py

# 3. 验证功能
pytest tests/ -v

# 4. 分析问题
# 查看日志、错误信息等
```

## 渐进式迁移策略

推荐采用渐进式迁移:

### Phase 1: 并行运行 (推荐)
```python
# 同时保留旧系统和新系统
self.use_new_cache_manager = config.get('use_new_cache_manager', False)

if self.use_new_cache_manager:
    # 使用新系统
    self.kv_cache_manager = KVCacheManager(...)
else:
    # 使用旧系统
    self._initialize_cache_structures_old()
```

### Phase 2: 对比验证
```python
# 同时运行两个系统,对比结果
if config.get('cache_validation_mode', False):
    result_old = self._get_cache_old(key)
    result_new = self.kv_cache_manager.get_cache(key)
    assert torch.allclose(result_old, result_new), "Cache mismatch!"
```

### Phase 3: 完全切换
```python
# 移除所有旧代码
# 仅保留新系统
```

## 下一步

1. ✅ 创建KVCacheManager - 完成
2. ✅ 编写单元测试 - 完成 (24/24通过)
3. ⏭️ 创建集成测试
4. ⏭️ 实施Phase 1迁移 (并行运行)
5. ⏭️ 性能benchmark
6. ⏭️ Phase 2验证
7. ⏭️ Phase 3完全切换

## 总结

重构的核心优势:
- ✅ **可测试性**: 独立模块,易于单元测试
- ✅ **可维护性**: 清晰的接口,减少代码重复
- ✅ **可扩展性**: 易于添加新的eviction策略
- ✅ **可监控性**: 内置统计和日志
- ✅ **鲁棒性**: 完善的参数验证和错误处理
