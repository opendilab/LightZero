# KV Cache 重构项目 - README

## 📋 项目概述

本项目对 UniZero World Model 的 KV Cache 管理系统进行了全面重构,将分散的缓存逻辑统一为可测试、可扩展、鲁棒的系统。

**状态**: ✅ 设计和测试完成,待实际集成

## 🎯 主要目标

1. **统一接口**: 替换3套独立的缓存系统为单一管理器
2. **可测试性**: 从0%提升到95%+的测试覆盖
3. **可扩展性**: 支持多种缓存驱逐策略 (FIFO/LRU/PRIORITY)
4. **可监控性**: 内置命中率、未命中率、驱逐次数等统计
5. **鲁棒性**: 完善的参数验证和错误处理

## 📁 文件结构

```
LightZero/
├── lzero/model/unizero_world_models/
│   ├── world_model.py                      # 原文件
│   ├── world_model.py.backup_20251023_143124  # 备份 (120KB)
│   └── kv_cache_manager.py                 # ✨ 新模块 (904行)
├── tests/
│   ├── test_kv_cache_manager.py            # 单元测试 (24 tests)
│   └── test_world_model_kv_integration.py  # 集成测试 (11 tests)
└── docs/
    ├── kv_cache_refactoring_analysis.md   # 架构分析
    ├── kv_cache_refactoring_guide.md      # 重构指南
    ├── KV_CACHE_REFACTORING_SUMMARY.md    # 工作总结
    ├── test_results.txt                    # 测试结果
    └── README_KV_CACHE.md                  # 本文件
```

## 🚀 快速开始

### 1. 运行测试

```bash
# 运行所有测试
pytest tests/test_kv_cache_manager.py tests/test_world_model_kv_integration.py -v

# 仅单元测试
pytest tests/test_kv_cache_manager.py -v

# 仅集成测试
pytest tests/test_world_model_kv_integration.py -v
```

**预期结果**: 35/35 测试通过

### 2. 使用示例

```python
from lzero.model.unizero_world_models.kv_cache_manager import KVCacheManager

# 创建管理器
manager = KVCacheManager(
    config=config,
    env_num=4,
    enable_stats=True
)

# 存储缓存
manager.set_init_cache(env_id=0, cache_key=123, kv_cache=kv_object)

# 检索缓存
kv = manager.get_init_cache(env_id=0, cache_key=123)

# 查看统计
stats = manager.get_stats_summary()
print(stats)
# {
#   'stats_enabled': True,
#   'init_pools': {'env_0': 'CacheStats(hits=10, misses=2, ...)'},
#   'recur_pool': 'CacheStats(...)',
#   'wm_pool': 'CacheStats(...)'
# }
```

## 📊 测试结果

### 单元测试 (24 tests)

✅ **CacheStats** (4 tests)
- 初始化
- 命中率计算
- 零查询情况
- 重置功能

✅ **KVCachePool** (9 tests)
- 基本 set/get 操作
- FIFO 驱逐策略
- LRU 驱逐策略
- 统计收集
- 缓存更新
- 缓存清除

✅ **KVCacheManager** (9 tests)
- 初始化
- 多环境隔离
- 缓存操作 (init/recur/wm)
- 统计管理
- 选择性清除

✅ **集成场景** (2 tests)
- 现实工作流模拟
- 缓存溢出行为

### 集成测试 (11 tests)

✅ **与 KeysValues 集成** (7 tests)
- 基本操作
- Cache key 生成一致性
- 多环境隔离
- Init→Recur 回退模式
- 缓存驱逐
- 统计跟踪
- 清除操作

✅ **缓存语义** (2 tests)
- 引用 vs 复制
- 更新现有条目

✅ **现实工作流** (2 tests)
- Imagine 函数模拟
- MCTS 搜索模拟

## 🏗️ 架构设计

### 核心类

#### 1. `EvictionStrategy`
```python
class EvictionStrategy(Enum):
    FIFO = "fifo"       # 先进先出 (默认)
    LRU = "lru"         # 最近最少使用
    PRIORITY = "priority"  # 基于优先级
```

#### 2. `CacheStats`
```python
@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_queries: int = 0

    @property
    def hit_rate(self) -> float: ...
    @property
    def miss_rate(self) -> float: ...
```

#### 3. `KVCachePool`
```python
class KVCachePool:
    def __init__(self, pool_size: int, eviction_strategy: EvictionStrategy, ...):
        self._pool: List[Optional[KeysValues]] = [None] * pool_size
        self._key_to_index: Dict[int, int] = {}
        self.stats = CacheStats()

    def get(self, cache_key: int) -> Optional[KeysValues]: ...
    def set(self, cache_key: int, kv_cache: KeysValues) -> int: ...
    def clear(self): ...
```

#### 4. `KVCacheManager`
```python
class KVCacheManager:
    def __init__(self, config, env_num: int, enable_stats: bool = True):
        self.init_pools: List[KVCachePool] = []  # Per-environment
        self.recur_pool: KVCachePool  # Shared for MCTS
        self.wm_pool: KVCachePool  # Temporary

    def get_init_cache(self, env_id: int, cache_key: int) -> Optional[KeysValues]
    def set_init_cache(self, env_id: int, cache_key: int, kv_cache: KeysValues) -> int
    # ... similar for recur and wm caches
```

## 🔄 集成路线图

### Phase 1: 并行运行 (推荐首先实施)

**目标**: 新旧系统共存,可配置切换

```python
# In world_model.py
def _initialize_cache_structures(self):
    self.use_new_cache = self.config.get('use_new_cache_manager', False)

    if self.use_new_cache:
        self.kv_cache_manager = KVCacheManager(
            config=self.config,
            env_num=self.env_num,
            enable_stats=True
        )
    else:
        # 保留旧代码
        self._initialize_cache_structures_old()
```

**优势**:
- ✅ 零风险,旧系统仍可用
- ✅ 通过配置快速切换
- ✅ 易于回滚

### Phase 2: 对比验证

**目标**: 确保新旧系统行为一致

```python
if self.config.get('cache_validation_mode', False):
    result_old = self._get_cache_old(key)
    result_new = self.kv_cache_manager.get_cache(key)
    assert torch.allclose(result_old, result_new), "Mismatch!"
```

### Phase 3: 完全切换

**目标**: 移除旧代码,仅保留新系统

- 删除旧缓存相关代码
- 更新所有引用
- 更新文档

## 📈 性能考虑

### 预期性能

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| get  | O(1)      | O(1)      |
| set  | O(1)      | O(1)      |
| FIFO eviction | O(1) | O(n) |
| LRU eviction | O(1) | O(n) |

### 内存开销

- **新增**: `CacheStats` (~40 bytes per pool)
- **新增**: `OrderedDict` for LRU (~8 bytes per entry)
- **节省**: 减少重复代码和数据结构

### 性能测试 (待实施)

创建 `benchmarks/benchmark_kv_cache.py` 对比:
- Cache 操作延迟
- 内存使用
- 命中率统计
- Throughput

## 🐛 已知问题和限制

### 当前限制

1. **单线程假设**: 当前实现未考虑并发访问
   - **解决方案**: 如需多线程,添加 `threading.Lock`

2. **固定 Pool 大小**: Pool 大小在初始化时确定
   - **影响**: 无法动态调整
   - **解决方案**: 未来可添加动态扩展功能

3. **缓存复制语义**: 存储引用而非深拷贝
   - **原因**: 性能考虑,与原实现一致
   - **注意**: 修改原对象会影响缓存

### 已解决问题

✅ **浮点精度**: 测试中的浮点比较 (使用近似比较)
✅ **KeysValues 构造**: 正确使用所有必需参数
✅ **缓存隔离**: 多环境缓存完全隔离

## 🔧 故障排除

### 测试失败

```bash
# 清理缓存重新运行
pytest tests/ --cache-clear -v

# 查看详细错误
pytest tests/ -vv --tb=long
```

### 导入错误

确保在项目根目录:
```bash
cd /mnt/nfs/zhangjinouwen/puyuan/LightZero
python -c "from lzero.model.unizero_world_models.kv_cache_manager import KVCacheManager"
```

### 集成问题

参考集成测试示例:
```bash
cat tests/test_world_model_kv_integration.py
```

## 📚 文档

- **架构分析**: `docs/kv_cache_refactoring_analysis.md`
  - 现有系统分析
  - 识别的问题
  - 改进方向

- **重构指南**: `docs/kv_cache_refactoring_guide.md`
  - 分步骤重构指南
  - 代码映射表
  - 注意事项

- **工作总结**: `docs/KV_CACHE_REFACTORING_SUMMARY.md`
  - 完成状态
  - 测试结果
  - 下一步计划

## 🤝 贡献指南

### 添加新的驱逐策略

1. 在 `EvictionStrategy` 枚举中添加新策略
2. 在 `KVCachePool._find_slot_for_new_entry()` 中实现逻辑
3. 添加相应的单元测试

示例:
```python
class EvictionStrategy(Enum):
    # ...
    CUSTOM = "custom"

# In KVCachePool._find_slot_for_new_entry()
elif self.eviction_strategy == EvictionStrategy.CUSTOM:
    # Your custom logic here
    pass
```

### 添加新统计指标

1. 在 `CacheStats` 中添加字段
2. 在相应位置更新计数
3. 更新测试

## 📞 支持

如有问题或建议:

1. 查看文档: `docs/` 目录
2. 运行测试: `pytest tests/ -v`
3. 检查备份: `world_model.py.backup_20251023_143124`

## 📜 变更日志

### 2025-10-23 - v1.0

✅ **完成**:
- 创建 `KVCacheManager` 核心模块
- 实现 FIFO/LRU 驱逐策略
- 编写 24 个单元测试 (100% 通过)
- 编写 11 个集成测试 (100% 通过)
- 创建详细文档和指南
- 备份原始文件

⏭️ **待完成**:
- 实际集成到 `world_model.py`
- 性能对比测试
- 长时间运行测试

## ⚖️ 许可证

遵循 LightZero 项目许可证

---

**版本**: 1.0
**最后更新**: 2025-10-23
**维护者**: Claude Code
**状态**: 设计和测试完成 ✅
