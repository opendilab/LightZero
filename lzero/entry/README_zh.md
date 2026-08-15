# LightZero 入口函数说明

[English](./README.md) | 中文

本目录包含了 LightZero 框架中各种算法的训练和评估入口函数。这些入口函数是启动不同类型强化学习实验的主要接口。

## 📁 目录结构

### 🎯 训练入口 (Training Entries)

#### AlphaZero 系列
- **`train_alphazero.py`** - AlphaZero 算法的训练入口
  - 适用于完美信息的棋类游戏（如五子棋、中国象棋等）
  - 不需要环境模型，直接通过自我对弈学习
  - 使用蒙特卡洛树搜索（MCTS）进行策略改进

#### MuZero 系列
- **`train_muzero.py`** - MuZero 算法的标准训练入口
  - 支持 MuZero、EfficientZero、Sampled EfficientZero、Gumbel MuZero 等变体
  - 学习环境的隐式模型（dynamics model）
  - 适用于单任务强化学习场景

- **`train_muzero_segment.py`** - MuZero 带分段收集器和缓冲区重分析技巧的训练入口
  - 使用 `MuZeroSegmentCollector` 进行数据收集
  - 支持缓冲区重分析（reanalyze）技巧提高样本效率
  - 支持的算法：MuZero, EfficientZero, Sampled MuZero, Sampled EfficientZero, Gumbel MuZero, StochasticMuZero

- **`train_muzero_with_gym_env.py`** - 适配 Gym 环境的 MuZero 训练入口
  - 专门为 OpenAI Gym 风格的环境设计
  - 简化了环境接口的适配过程

- **`train_muzero_with_reward_model.py`** - 带奖励模型的 MuZero 训练入口
  - 集成外部奖励模型（Reward Model）
  - 适用于需要学习复杂奖励函数的场景

- **`train_muzero_multitask_segment_ddp.py`** - MuZero 多任务分布式训练入口
  - 支持多任务学习（Multi-task Learning）
  - 使用 DDP (Distributed Data Parallel) 进行分布式训练
  - 使用分段收集器（Segment Collector）

#### UniZero 系列
- **`train_unizero.py`** - UniZero 算法的训练入口
  - 基于论文 "UniZero: Generalized and Efficient Planning with Scalable Latent World Models"
  - 增强的规划能力，能更好地捕获长期依赖
  - 使用可扩展的隐式世界模型
  - 论文链接：https://arxiv.org/abs/2406.10667

- **`train_unizero_segment.py`** - UniZero 带分段收集器的训练入口
  - 使用 `MuZeroSegmentCollector` 进行高效数据收集
  - 支持缓冲区重分析技巧

- **`train_unizero_multitask_segment_ddp.py`** - UniZero/ScaleZero 多任务分布式训练入口
  - 支持多任务学习和分布式训练
  - 包含基准测试分数定义（如 Atari 的人类归一化分数）
  - 支持课程学习（Curriculum Learning）策略
  - 使用 DDP 加速训练

- **`train_unizero_multitask_balance_segment_ddp.py`** - UniZero/ScaleZero 多任务均衡分布式训练入口
  - 在多任务训练中实现任务间的均衡采样
  - 动态调整不同任务的批次大小
  - 适用于任务难度差异较大的场景

- **`train_unizero_multitask_segment_eval.py`** - UniZero/ScaleZero 多任务评估训练入口
  - 专门用于多任务场景的训练和周期性评估
  - 包含详细的评估指标统计

- **`train_unizero_with_loss_landscape.py`** - UniZero 损失地形可视化训练入口
  - 用于训练的同时进行损失地形（Loss Landscape）可视化
  - 帮助理解模型的优化过程和泛化性能
  - 集成 `loss_landscapes` 库

#### ReZero 系列
- **`train_rezero.py`** - ReZero 算法的训练入口
  - 支持 ReZero-MuZero 和 ReZero-EfficientZero
  - 通过残差连接改进训练稳定性
  - 论文链接：https://arxiv.org/pdf/2404.16364

### 🎓 评估入口 (Evaluation Entries)

- **`eval_alphazero.py`** - AlphaZero 算法的评估入口
  - 加载训练好的 AlphaZero 模型进行评估
  - 可以与其他智能体对弈测试性能

- **`eval_muzero.py`** - MuZero 系列算法的评估入口
  - 支持所有 MuZero 变体的评估
  - 提供详细的性能统计

- **`eval_muzero_with_gym_env.py`** - Gym 环境下的 MuZero 评估入口  (最近没有维护此入口)
  - 专门用于评估在 Gym 环境中训练的模型


## 📖 使用指南

### 基本使用模式

所有训练入口函数遵循相似的调用模式：

```python
from lzero.entry import train_muzero

# 准备配置
cfg = dict(...)  # 用户配置
create_cfg = dict(...)  # 创建配置

# 开始训练
policy = train_muzero(
    input_cfg=(cfg, create_cfg),
    seed=0,
    model=None,  # 可选：预初始化模型
    model_path=None,  # 可选：预训练模型路径
    max_train_iter=int(1e10),  # 最大训练迭代次数
    max_env_step=int(1e10),  # 最大环境步数
)
```

### 选择合适的入口函数

1. **单任务学习**：
   - 棋类游戏 → `train_alphazero`
   - 一般 RL 任务 → `train_muzero` 或 `train_unizero`
   - Gym 环境 → `train_muzero_with_gym_env` (最近没有维护此入口)

2. **多任务学习**：
   - 标准多任务 → `train_unizero_multitask_segment_ddp`
   - 任务均衡采样 → `train_unizero_multitask_balance_segment_ddp`

3. **分布式训练**：
   - 所有带 `_ddp` 后缀的入口函数都支持数据并行分布式训练

4. **特殊需求**：
   - 损失地形可视化 → `train_unizero_with_loss_landscape`
   - 外部奖励模型 → `train_muzero_with_reward_model`
   - 改进训练稳定性 → `train_rezero`

## 🔗 相关资源

- **AlphaZero**: [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)
- **MuZero**: [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)
- **EfficientZero**: [Mastering Atari Games with Limited Data](https://arxiv.org/abs/2111.00210)
- **UniZero**: [Generalized and Efficient Planning with Scalable Latent World Models](https://arxiv.org/abs/2406.10667)
- **ReZero**: [Boosting MCTS-based Algorithms by Reconstructing the Terminal Reward](https://arxiv.org/abs/2404.16364)
- **ScaleZero**: [One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning](https://arxiv.org/abs/2509.07945)

## 💡 提示

- 建议从标准的 `train_muzero` 或 `train_unizero` 开始
- 对于大规模实验，考虑使用 DDP 版本以提高训练速度
- 使用 `_segment` 版本可以获得更好的样本效率
- 查看 `zoo/` 目录下的配置示例以了解如何设置各个算法

## 📝 注意事项

1. 所有路径参数建议使用**绝对路径**
2. 预训练模型路径通常格式为 `exp_name/ckpt/ckpt_best.pth.tar`
3. 使用分布式训练时，确保正确设置 `CUDA_VISIBLE_DEVICES` 环境变量
