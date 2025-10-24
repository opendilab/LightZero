# Phase 1.5 实施指南: 存储层替换

## 概述

本文档提供 Phase 1.5 的具体实施步骤，展示如何在三个关键方法中将旧 cache 系统调用替换为 KVCacheManager 调用。

---

## 修改 1: `retrieve_or_generate_kvcache()`

### 位置
world_model.py: Line 1472-1550

### 修改内容

#### Before (Line 1493-1518):
```python
if self.reanalyze_phase:
    # TODO: check if this is correct
    matched_value = None
else:
    # Try to retrieve the cached value from past_kv_cache_init_infer_envs
    cache_index = self.past_kv_cache_init_infer_envs[index].get(cache_key)
    if cache_index is not None:
        matched_value = self.shared_pool_init_infer[index][cache_index]
    else:
        matched_value = None

    # ==================== TODO ====================
    # 步骤 2: 仅当在 init_infer 中未找到时，才尝试从 recurrent_infer 缓存中查找
    if matched_value is None:
        # 2.1 安全地从字典中获取索引，它可能返回 None
        recur_cache_index = self.past_kv_cache_recurrent_infer.get(cache_key)
        # 2.2 只有在索引有效（不是 None）的情况下，才使用它来从物理池中检索值
        if recur_cache_index is not None:
            matched_value = self.shared_pool_recur_infer[recur_cache_index]

        if recur_cache_index is None:
            print(f"[CACHE MISS]  Not found for key={cache_key} in recurrent infer. Generating new cache.")
```

#### After:
```python
if self.reanalyze_phase:
    # TODO: check if this is correct
    matched_value = None
else:
    # ==================== Phase 1.5: Cache System Selection ====================
    if self.use_new_cache_manager:
        # NEW SYSTEM: Use KVCacheManager for hierarchical cache lookup
        # Step 1: Try init_infer cache first (per-environment)
        matched_value = self.kv_cache_manager.get_init_cache(env_id=index, cache_key=cache_key)

        # Step 2: If not found, try recurrent_infer cache (global)
        if matched_value is None:
            matched_value = self.kv_cache_manager.get_recur_cache(cache_key=cache_key)

        # Step 3: Log cache miss (统计由 KVCacheManager 自动处理)
        if matched_value is None:
            logging.debug(f"[NEW CACHE MISS] Not found for key={cache_key} in both init and recurrent cache.")
    else:
        # OLD SYSTEM: Use legacy cache dictionaries and pools
        # Try to retrieve the cached value from past_kv_cache_init_infer_envs
        cache_index = self.past_kv_cache_init_infer_envs[index].get(cache_key)
        if cache_index is not None:
            matched_value = self.shared_pool_init_infer[index][cache_index]
        else:
            matched_value = None

        # 仅当在 init_infer 中未找到时，才尝试从 recurrent_infer 缓存中查找
        if matched_value is None:
            # 安全地从字典中获取索引，它可能返回 None
            recur_cache_index = self.past_kv_cache_recurrent_infer.get(cache_key)
            # 只有在索引有效（不是 None）的情况下，才使用它来从物理池中检索值
            if recur_cache_index is not None:
                matched_value = self.shared_pool_recur_infer[recur_cache_index]

            if recur_cache_index is None:
                logging.debug(f"[OLD CACHE MISS] Not found for key={cache_key} in recurrent infer. Generating new cache.")
    # =============================================================================
```

### 关键点说明

1. **两级查找保持一致**:
   - 新旧系统都是先查 init_cache，再查 recur_cache
   - 查找逻辑完全相同，只是底层存储不同

2. **统计自动化**:
   - 新系统: KVCacheManager 在 get 方法内部自动记录 hit/miss
   - 旧系统: 继续使用 `self.hit_count` (见 Line 1522)

3. **日志改进**:
   - 使用 `logging.debug()` 替代 `print()`
   - 区分新旧系统的日志前缀

4. **向后兼容**:
   - 旧系统代码完全保留
   - 只在 `use_new_cache_manager=True` 时使用新系统

---

## 修改 2: `update_cache_context()`

### 位置
world_model.py: Line 1432-1448

### 修改内容

#### Before (Line 1432-1448):
```python
if is_init_infer:
    # TODO
    # ==================== 主动淘汰修复逻辑 ====================
    # 1. 获取即将被覆写的物理索引
    index_to_write = self.shared_pool_index_init_envs[i]
    # 2. 使用辅助列表查找该索引上存储的旧的 key
    old_key = self.pool_idx_to_key_map_init_envs[i][index_to_write]
    # 3. 如果该索引已经存储过某个键（不是 None），则需要先删除字典中的映射
    if old_key is not None:
        del self.past_kv_cache_init_infer_envs[i][old_key]
    # 4. 将新的 key 存储到字典和辅助列表
    cache_index = self.custom_copy_kv_cache_to_shared_init_envs(self.keys_values_wm_single_env, i)
    self.past_kv_cache_init_infer_envs[i][cache_key] = cache_index
else:
    # Store the latest key-value cache for recurrent inference
    cache_index = self.custom_copy_kv_cache_to_shared_recur(self.keys_values_wm_single_env)
    self.past_kv_cache_recurrent_infer[cache_key] = cache_index
```

#### After:
```python
# ==================== Phase 1.5: Cache System Selection ====================
if self.use_new_cache_manager:
    # NEW SYSTEM: Use KVCacheManager for cache storage
    if is_init_infer:
        # Store to per-environment init cache pool
        self.kv_cache_manager.set_init_cache(
            env_id=i,
            cache_key=cache_key,
            kv_cache=self.keys_values_wm_single_env
        )
        # Note: KVCacheManager 自动处理淘汰逻辑 (FIFO/LRU)
    else:
        # Store to global recurrent cache pool
        self.kv_cache_manager.set_recur_cache(
            cache_key=cache_key,
            kv_cache=self.keys_values_wm_single_env
        )
else:
    # OLD SYSTEM: Use legacy cache with manual eviction
    if is_init_infer:
        # ==================== 主动淘汰修复逻辑 ====================
        # 1. 获取即将被覆写的物理索引
        index_to_write = self.shared_pool_index_init_envs[i]
        # 2. 使用辅助列表查找该索引上存储的旧的 key
        old_key = self.pool_idx_to_key_map_init_envs[i][index_to_write]
        # 3. 如果该索引已经存储过某个键（不是 None），则需要先删除字典中的映射
        if old_key is not None:
            del self.past_kv_cache_init_infer_envs[i][old_key]
        # 4. 将新的 key 存储到字典和辅助列表
        cache_index = self.custom_copy_kv_cache_to_shared_init_envs(self.keys_values_wm_single_env, i)
        self.past_kv_cache_init_infer_envs[i][cache_key] = cache_index
    else:
        # Store the latest key-value cache for recurrent inference
        cache_index = self.custom_copy_kv_cache_to_shared_recur(self.keys_values_wm_single_env)
        self.past_kv_cache_recurrent_infer[cache_key] = cache_index
# =============================================================================
```

### 关键点说明

1. **简化淘汰逻辑**:
   - 新系统: KVCacheManager 自动处理淘汰 (通过 EvictionStrategy)
   - 旧系统: 手动管理 `pool_idx_to_key_map_init_envs`

2. **深拷贝处理**:
   - 新系统: `set_init_cache()` 内部自动深拷贝
   - 旧系统: `custom_copy_kv_cache_to_shared_init_envs()` 手动深拷贝

3. **无需索引管理**:
   - 新系统: 不返回 cache_index，由 KVCachePool 内部管理
   - 旧系统: 返回 cache_index 用于 `pool_idx_to_key_map` 映射

---

## 修改 3: `trim_and_pad_kv_cache()`

### 位置
world_model.py: Line 1235-1285

### 修改内容

**无需修改!**

**原因**:
1. 此方法操作的是 `self.keys_values_wm_list` 和 `self.keys_values_wm`
2. 这些是 WorldModel 的**临时批处理 cache**，不是持久化存储
3. 与 cache 存储系统 (旧/新) 无关

**架构说明**:
```
┌─────────────────────────────────────────────────┐
│  WorldModel                                     │
│  ┌───────────────────────────────────────────┐ │
│  │  Batch Processing Caches (临时)           │ │
│  │  - keys_values_wm_list  (multi-env)       │ │
│  │  - keys_values_wm       (stacked)         │ │
│  │  - keys_values_wm_single_env (single)     │ │
│  │                                            │ │
│  │  Used by:                                  │ │
│  │  - trim_and_pad_kv_cache()  ← 无需修改    │ │
│  │  - forward() for batched inference        │ │
│  └───────────────────────────────────────────┘ │
│                                                 │
│  ┌───────────────────────────────────────────┐ │
│  │  Persistent Caches (持久化)               │ │
│  │                                            │ │
│  │  OLD System:                               │ │
│  │  - past_kv_cache_init_infer_envs          │ │
│  │  - past_kv_cache_recurrent_infer          │ │
│  │                                            │ │
│  │  NEW System:                               │ │
│  │  - kv_cache_manager                        │ │
│  │    - init_pools                            │ │
│  │    - recur_pool                            │ │
│  │                                            │ │
│  │  Used by:                                  │ │
│  │  - retrieve_or_generate_kvcache() ← 修改   │ │
│  │  - update_cache_context()         ← 修改   │ │
│  └───────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

---

## 测试策略

### 1. 单元测试

创建 `tests/test_phase1_5_integration.py`:

```python
"""
Phase 1.5 集成测试: 验证三个方法在新旧系统下的一致性
"""
import torch
import numpy as np
from lzero.model.unizero_world_models.world_model import WorldModel
from lzero.model.unizero_world_models.transformer import TransformerConfig

def test_retrieve_or_generate_consistency():
    """测试 retrieve_or_generate_kvcache 在新旧系统下的一致性"""

    # 创建配置和模型
    config_old = create_test_config(use_new_cache=False)
    config_new = create_test_config(use_new_cache=True)

    model_old = create_test_model(config_old)
    model_new = create_test_model(config_new)

    # 准备测试数据
    latent_state = [np.random.randn(1, 768) for _ in range(4)]
    ready_env_num = 4
    start_pos = torch.zeros(4, 1)

    # 旧系统执行
    model_old.keys_values_wm_list.clear()
    model_old.keys_values_wm_size_list.clear()
    sizes_old = model_old.retrieve_or_generate_kvcache(
        latent_state, ready_env_num, start_pos=start_pos
    )

    # 新系统执行
    model_new.keys_values_wm_list.clear()
    model_new.keys_values_wm_size_list.clear()
    sizes_new = model_new.retrieve_or_generate_kvcache(
        latent_state, ready_env_num, start_pos=start_pos
    )

    # 验证
    assert len(sizes_old) == len(sizes_new)
    assert sizes_old == sizes_new
    assert len(model_old.keys_values_wm_list) == len(model_new.keys_values_wm_list)

    print("✅ retrieve_or_generate_kvcache 一致性测试通过")


def test_update_cache_context_consistency():
    """测试 update_cache_context 在新旧系统下的一致性"""

    config_old = create_test_config(use_new_cache=False)
    config_new = create_test_config(use_new_cache=True)

    model_old = create_test_model(config_old)
    model_new = create_test_model(config_new)

    # 准备测试数据
    latent_state = torch.randn(4, 1, 768)

    # 旧系统执行
    model_old.update_cache_context(latent_state, is_init_infer=True)

    # 新系统执行
    model_new.update_cache_context(latent_state, is_init_infer=True)

    # 验证 cache 是否存储成功
    # (需要检查 cache key 是否在新旧系统中都能找到)
    # ...

    print("✅ update_cache_context 一致性测试通过")
```

### 2. 集成测试

使用现有的 `test_kv_cache_consistency.py`，添加新的测试用例:

```python
def test_retrieve_generate_with_new_cache():
    """测试 retrieve_or_generate_kvcache 在新系统下的行为"""

    config = create_minimal_config(use_new_cache=True)
    model = WorldModel(config, mock_tokenizer)

    # 第一次调用: 应该 miss, 生成新 cache
    latent_state = [np.random.randn(1, 768) for _ in range(2)]
    sizes = model.retrieve_or_generate_kvcache(latent_state, ready_env_num=2, start_pos=torch.zeros(2, 1))

    assert len(sizes) == 2
    assert len(model.keys_values_wm_list) == 2

    # 检查统计
    stats = model.kv_cache_manager.get_stats_summary()
    assert stats['init_pools']['env_0']['misses'] == 1  # 第一次应该 miss

    # 第二次调用相同的 latent_state: 应该 hit
    model.keys_values_wm_list.clear()
    sizes2 = model.retrieve_or_generate_kvcache(latent_state, ready_env_num=2, start_pos=torch.zeros(2, 1))

    stats2 = model.kv_cache_manager.get_stats_summary()
    assert stats2['init_pools']['env_0']['hits'] >= 1  # 应该有 hit

    print("✅ retrieve_generate_with_new_cache 测试通过")
```

### 3. 性能测试

```python
def benchmark_cache_operations():
    """对比新旧系统的 cache 操作性能"""
    import time

    config_old = create_test_config(use_new_cache=False)
    config_new = create_test_config(use_new_cache=True)

    model_old = create_test_model(config_old)
    model_new = create_test_model(config_new)

    latent_state = [np.random.randn(1, 768) for _ in range(8)]

    # 旧系统
    start = time.time()
    for _ in range(100):
        model_old.keys_values_wm_list.clear()
        model_old.retrieve_or_generate_kvcache(latent_state, ready_env_num=8, start_pos=torch.zeros(8, 1))
    time_old = time.time() - start

    # 新系统
    start = time.time()
    for _ in range(100):
        model_new.keys_values_wm_list.clear()
        model_new.retrieve_or_generate_kvcache(latent_state, ready_env_num=8, start_pos=torch.zeros(8, 1))
    time_new = time.time() - start

    print(f"旧系统: {time_old:.3f}s")
    print(f"新系统: {time_new:.3f}s")
    print(f"性能差异: {(time_new / time_old - 1) * 100:.1f}%")
```

---

## 实施 Checklist

### 准备阶段
- [ ] 阅读完整的分析文档 (KV_CACHE_INTEGRATION_ANALYSIS.md)
- [ ] 理解新旧系统的架构差异
- [ ] 确认 Phase 1 已完成 (新旧系统并行运行)

### 实施阶段
- [ ] 修改 `retrieve_or_generate_kvcache()` (world_model.py:1493-1518)
- [ ] 修改 `update_cache_context()` (world_model.py:1432-1448)
- [ ] 确认 `trim_and_pad_kv_cache()` 无需修改

### 测试阶段
- [ ] 运行现有一致性测试 (`test_kv_cache_consistency.py`)
- [ ] 创建 Phase 1.5 专用测试 (`test_phase1_5_integration.py`)
- [ ] 执行性能基准测试
- [ ] 验证 hit/miss 统计准确性

### 验证阶段
- [ ] 使用旧系统运行训练 (use_new_cache_manager=False)
- [ ] 使用新系统运行训练 (use_new_cache_manager=True)
- [ ] 对比训练曲线和最终性能
- [ ] 检查内存使用情况
- [ ] 收集 cache 命中率数据

### 文档阶段
- [ ] 更新 PHASE1_INTEGRATION_REPORT.md
- [ ] 创建 PHASE1_5_COMPLETION_REPORT.md
- [ ] 更新 KV_CACHE_CONFIG_GUIDE.md

---

## 预期结果

### 功能性
- ✅ 新旧系统在三个方法中都能正确工作
- ✅ Cache hit/miss 行为一致
- ✅ 训练结果一致 (相同随机种子)

### 性能
- ✅ 新系统性能不低于旧系统 (目标: ±5%)
- ✅ 内存使用相当或更优
- ✅ Cache 命中率相当或更高

### 可维护性
- ✅ 代码更清晰 (统一的 cache 接口)
- ✅ 统计更完善 (自动 hit/miss 记录)
- ✅ 易于扩展 (新 eviction 策略)

---

## 回滚方案

如果新系统出现问题:

1. **立即回滚**: 设置 `use_new_cache_manager=False`
2. **无需代码更改**: 旧系统代码完全保留
3. **数据兼容**: 新旧系统独立存储，互不影响

---

**文档版本**: 1.0
**作者**: Claude
**日期**: 2025-10-23
**状态**: 实施指南
