# Worker 模块

本目录包含 LightZero 强化学习算法的工作组件，包括数据收集器和评估器。

## 文件概览

### 收集器文件
收集器负责在训练过程中通过环境交互收集经验数据。

| 文件 | 算法 | 收集模式 | 说明 |
|------|------|---------|------|
| `alphazero_collector.py` | AlphaZero | 基于回合 | 为 AlphaZero 算法收集完整的游戏回合。专为完全信息博弈设计（如棋类游戏）。 |
| `muzero_collector.py` | MuZero/EfficientZero/Gumbel MuZero | 基于回合 | 为 MCTS+RL 算法收集完整的游戏回合。支持完全和不完全信息环境。 |
| `muzero_segment_collector.py` | MuZero/EfficientZero/Gumbel MuZero | 基于片段 | 收集指定数量的游戏片段而非完整回合。提供更大的灵活性和可扩展性。 |

### 评估器文件
评估器在训练过程中评估已训练策略的性能。

| 文件 | 算法 | 说明 |
|------|------|------|
| `alphazero_evaluator.py` | AlphaZero | 在测试环境中评估 AlphaZero 策略性能。 |
| `muzero_evaluator.py` | MuZero/EfficientZero | 评估基于 MuZero 的策略性能，支持多任务场景。 |

## 主要差异

### AlphaZero vs MuZero
- **AlphaZero**：专为完全信息博弈设计，游戏状态完全可观察（如围棋、象棋）
- **MuZero**：通用算法，支持完全和不完全信息环境，具有学习的动力学模型

### Collector vs Evaluator
- **Collector（收集器）**：通过自我对弈或环境交互收集训练数据
- **Evaluator（评估器）**：在训练期间定期评估策略性能

### MuZeroCollector vs MuZeroSegmentCollector
- **MuZeroCollector**：收集完整游戏回合后返回数据
- **MuZeroSegmentCollector**：收集指定数量的游戏片段后返回数据，提供更细粒度的数据收集控制

## 共同特性

所有工作组件都支持：
- 分布式训练（多进程/多GPU）
- TensorBoard 日志记录
- 多任务学习场景（通过 `task_id` 参数）
- 可配置的收集/评估频率
- 环境和策略重置功能

## 使用示例

```python
from lzero.worker import MuZeroCollector, MuZeroEvaluator

# 初始化收集器
collector = MuZeroCollector(
    collect_print_freq=100,
    env=env_manager,
    policy=policy,
    tb_logger=tb_logger,
    exp_name='my_experiment',
    policy_config=policy_config
)

# 初始化评估器
evaluator = MuZeroEvaluator(
    eval_freq=1000,
    n_evaluator_episode=10,
    env=eval_env,
    policy=policy,
    tb_logger=tb_logger,
    exp_name='my_experiment'
)
```
