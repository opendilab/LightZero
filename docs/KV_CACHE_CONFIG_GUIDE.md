# KV Cache 系统切换配置示例

本文档展示如何在配置中切换使用新旧 KV Cache 系统。

## 使用旧系统 (默认)

如果不设置任何配置,默认使用旧的 KV Cache 系统:

```python
# 配置文件中不需要额外设置
# 或显式设置:
world_model_cfg=dict(
    # ... 其他配置 ...
    use_new_cache_manager=False,  # 使用旧系统 (默认)
)
```

## 使用新系统 (Phase 1)

要启用新的 KVCacheManager,只需在配置中添加一个标志:

```python
world_model_cfg=dict(
    # ... 其他配置 ...
    use_new_cache_manager=True,  # ✨ 启用新的 KVCacheManager

    # 新系统相关配置 (可选)
    # game_segment_length=400,  # 用于计算 init cache pool 大小
    # num_simulations=50,        # 用于计算 recur cache pool 大小
    # env_num=8,                 # 环境数量
)
```

## 配置示例

### 示例 1: 默认配置 (旧系统)

```python
from easydict import EasyDict

config = dict(
    policy=dict(
        model=dict(
            world_model_cfg=dict(
                max_blocks=10,
                max_tokens=20,
                context_length=8,
                device='cuda',
                env_num=8,
                num_simulations=50,
                # 不设置 use_new_cache_manager,默认使用旧系统
            ),
        ),
    ),
)

config = EasyDict(config)
```

### 示例 2: 启用新系统

```python
from easydict import EasyDict

config = dict(
    policy=dict(
        model=dict(
            world_model_cfg=dict(
                max_blocks=10,
                max_tokens=20,
                context_length=8,
                device='cuda',
                env_num=8,
                num_simulations=50,
                game_segment_length=400,

                # ✨ 启用新的 KV Cache Manager
                use_new_cache_manager=True,
            ),
        ),
    ),
)

config = EasyDict(config)
```

## 运行时日志

### 旧系统日志
```
INFO: Using OLD cache system (original implementation)
...
Cleared WorldModel past_kv_cache (OLD system).
```

### 新系统日志
```
INFO: ✓ Using NEW KVCacheManager for cache management
INFO: Initialized KVCachePool 'init_env0' with size=400, strategy=fifo
INFO: Initialized KVCachePool 'init_env1' with size=400, strategy=fifo
...
INFO: Initialized KVCachePool 'recurrent' with size=400, strategy=fifo
INFO: Initialized KVCachePool 'world_model' with size=8, strategy=fifo
INFO: Initialized KVCacheManager for 8 environments
...
Cleared WorldModel KV caches (NEW system).
```

## 验证系统切换

运行以下代码验证系统是否正确切换:

```python
# 在训练脚本中添加:
print(f"Using new cache manager: {world_model.use_new_cache_manager}")

if world_model.use_new_cache_manager:
    # 新系统的统计信息
    stats = world_model.kv_cache_manager.get_stats_summary()
    print(f"Cache stats: {stats}")
else:
    # 旧系统
    print("Using legacy cache system")
```

## 性能对比

要对比新旧系统的性能,可以运行相同的实验两次:

### 实验 1: 旧系统
```bash
python train.py --config config_old.py
```

### 实验 2: 新系统
```bash
python train.py --config config_new.py
```

然后比较:
- 训练时间
- 内存使用
- Cache命中率 (新系统提供)
- 最终性能指标

## 注意事项

1. **向后兼容**: 新系统完全向后兼容,不会影响现有代码
2. **统计信息**: 仅新系统提供 cache hit/miss 统计
3. **性能**: 新系统性能预计与旧系统相当或更好
4. **调试**: 新系统提供更详细的日志,便于调试

## 回滚方案

如果新系统出现问题,只需修改配置:

```python
world_model_cfg=dict(
    # ... 其他配置 ...
    use_new_cache_manager=False,  # 切回旧系统
)
```

无需任何代码更改。

## 后续步骤

一旦验证新系统工作正常:
- **Phase 2**: 添加对比验证逻辑
- **Phase 3**: 完全移除旧代码

---

**文档版本**: 1.0
**最后更新**: 2025-10-23
**状态**: Phase 1 实施完成
