# PriorZero Async Training

异步训练功能允许collect、train和eval任务并发执行，以提高训练吞吐量。

## 功能特性

### 1. 可配置的异步程度 (off_policy_degree)

`off_policy_degree` 参数控制collect和train之间的异步程度：

- **0**: 完全同步模式 (默认)
  - 严格串行执行：collect -> train -> eval
  - 与原始训练流程完全一致
  - 最低off-policy偏差

- **1-10**: 低异步模式
  - train最多可以落后collect 1-10个batch
  - 适度提升吞吐量
  - 较低off-policy偏差

- **10-50**: 中等异步模式
  - train最多可以落后collect 10-50个batch
  - 显著提升吞吐量
  - 中等off-policy偏差

- **>50**: 高异步模式
  - 最大吞吐量
  - 最高off-policy偏差
  - 可能影响训练稳定性

- **-1**: 自动调优
  - 根据buffer_size和batch_size自动计算
  - 允许lag约为buffer容量的10%

### 2. 异步Evaluation

`enable_async_eval` 参数控制evaluation的执行方式：

- **False** (默认): 同步eval
  - eval会阻塞训练流程
  - 确保eval结果及时用于决策（如early stopping）

- **True**: 异步eval
  - eval在后台运行，不阻塞训练
  - 提高训练吞吐量
  - 适合eval时间较长的场景

## 使用方法

### 1. 修改配置文件

在 `priorzero_config.py` 中设置异步参数：

```python
policy_config = dict(
    # ... 其他配置 ...

    # 异步训练配置
    off_policy_degree=10,        # 设置为10允许中等异步
    enable_async_eval=False,     # 同步eval
)
```

### 2. 运行训练

```bash
# 使用默认配置（同步模式）
python priorzero_entry.py --env_id zork1.z5

# 快速测试模式
python priorzero_entry.py --env_id zork1.z5 --quick_test
```

### 3. 查看异步统计

训练结束后会自动打印异步训练统计信息：

```
================================================================================
Async Training Statistics:
  Mode: ASYNCHRONOUS
  Collect iterations: 150
  Train iterations: 145
  Final lag: 5
  Avg collect time: 2.35s
  Avg train time: 1.82s
  Avg eval time: 8.67s
================================================================================
```

## 配置示例

### 示例1: 完全同步（原始行为）

```python
policy_config = dict(
    off_policy_degree=0,         # 同步模式
    enable_async_eval=False,     # 同步eval
)
```

### 示例2: 轻度异步（推荐用于初次尝试）

```python
policy_config = dict(
    off_policy_degree=5,         # 允许5个batch的lag
    enable_async_eval=False,     # 同步eval
)
```

### 示例3: 激进异步（最大吞吐量）

```python
policy_config = dict(
    off_policy_degree=50,        # 允许50个batch的lag
    enable_async_eval=True,      # 异步eval
)
```

### 示例4: 自动调优

```python
policy_config = dict(
    off_policy_degree=-1,        # 自动调优
    enable_async_eval=False,
)
```

## 工作原理

### 同步模式 (off_policy_degree=0)

```
时间线：
Iter 0: [Collect] -> [Train] -> [Eval]
Iter 1: [Collect] -> [Train]
Iter 2: [Collect] -> [Train] -> [Eval]
...
```

### 异步模式 (off_policy_degree=10)

```
时间线：
Iter 0: [Collect 0]
Iter 1:   [Train 0] | [Collect 1]
Iter 2:   [Train 1] | [Collect 2]
Iter 3:   [Train 2] | [Collect 3] | [Eval]
...

注：collect和train可以并发执行，只要lag不超过设定值
```

## AsyncTrainingCoordinator 类

核心协调器类，负责管理异步执行：

### 主要方法

- `can_collect()`: 检查是否可以开始collect（lag未超限）
- `can_train()`: 检查是否可以开始train（有数据可用）
- `run_collect(collect_fn)`: 运行collect并更新计数器
- `run_train(train_fn)`: 运行train并更新计数器
- `run_eval(eval_fn)`: 运行eval（同步或异步）
- `get_statistics()`: 获取性能统计

### 关键属性

- `collect_count`: collect完成次数
- `train_count`: train完成次数
- `collect_train_lag`: 当前lag = collect_count - train_count

## 测试

运行单元测试以验证异步训练功能：

```bash
# 运行所有测试
cd zoo/jericho/priorzero
python test_async_training.py

# 运行特定测试
python test_async_training.py --mode sync         # 测试同步模式
python test_async_training.py --mode async_low    # 测试低异步
python test_async_training.py --mode async_high   # 测试高异步
python test_async_training.py --mode async_eval   # 测试异步eval
python test_async_training.py --mode auto_tune    # 测试自动调优
```

## 性能建议

### 何时使用同步模式 (off_policy_degree=0)

- 首次运行新环境/任务
- 需要严格的on-policy训练
- 调试和验证算法正确性
- Collect和train时间相近

### 何时使用异步模式 (off_policy_degree>0)

- Collect明显慢于train
- 追求最大训练吞吐量
- 算法对off-policy数据不敏感
- 有充足的replay buffer容量

### 推荐配置

| 场景 | off_policy_degree | enable_async_eval |
|------|-------------------|-------------------|
| 调试/验证 | 0 | False |
| 生产训练（保守） | 5-10 | False |
| 生产训练（激进） | 20-50 | True |
| 最大吞吐量 | -1 (auto) | True |

## 注意事项

1. **Off-policy偏差**：异步度越高，off-policy偏差越大，可能影响训练稳定性
2. **Buffer大小**：确保replay_buffer_size足够大以容纳lag带来的额外数据
3. **内存使用**：异步模式可能增加内存使用（同时存在collect和train的临时数据）
4. **Eval结果**：异步eval模式下，eval结果可能不能及时用于early stopping

## 故障排除

### 问题：训练不稳定

**解决方案**：
- 降低 `off_policy_degree`
- 使用同步eval (`enable_async_eval=False`)
- 增加 `replay_buffer_size`

### 问题：吞吐量没有提升

**可能原因**：
- Collect和train时间相近（异步收益小）
- `off_policy_degree` 设置过小
- 单GPU上LLM推理成为瓶颈

**解决方案**：
- 增加 `off_policy_degree`
- 使用更多collector环境 (`collector_env_num`)
- 优化vLLM配置（tensor_parallel_size, gpu_memory_utilization）

### 问题：内存溢出

**解决方案**：
- 降低 `off_policy_degree`
- 减少 `collector_env_num`
- 降低 `batch_size`
- 减少 `replay_buffer_size`

## 实现细节

异步训练功能通过以下文件实现：

1. **async_training_coordinator.py**: 核心协调器类
   - `AsyncTrainingCoordinator`: 管理异步执行逻辑
   - 计数器和锁保证线程安全
   - 性能统计收集

2. **priorzero_config.py**: 配置参数
   - `off_policy_degree`: 异步程度控制
   - `enable_async_eval`: eval模式控制

3. **priorzero_entry.py**: 主训练循环
   - 集成AsyncTrainingCoordinator
   - 支持同步/异步模式切换
   - 统计信息打印

4. **test_async_training.py**: 单元测试
   - 验证同步/异步模式正确性
   - 性能基准测试

## 未来改进

- [ ] 支持多GPU异步训练（collect和train在不同GPU）
- [ ] 动态调整off_policy_degree（根据训练稳定性）
- [ ] 更细粒度的异步控制（per-environment async）
- [ ] 异步reanalyze buffer
- [ ] TensorBoard可视化异步统计
