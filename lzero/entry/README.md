# LightZero Entry Functions

English | [中文](./README_zh.md)

This directory contains the training and evaluation entry functions for various algorithms in the LightZero framework. These entry functions serve as the main interfaces for launching different types of reinforcement learning experiments.

## 📁 Directory Structure

### 🎯 Training Entries

#### AlphaZero Family
- **`train_alphazero.py`** - Training entry for AlphaZero algorithm
  - Suitable for perfect information board games (e.g., Go, Chess)
  - No environment model needed, learns through self-play
  - Uses Monte Carlo Tree Search (MCTS) for policy improvement

#### MuZero Family
- **`train_muzero.py`** - Standard training entry for MuZero algorithm
  - Supports MuZero, EfficientZero, Sampled EfficientZero, Gumbel MuZero variants
  - Learns an implicit model of the environment (dynamics model)
  - Suitable for single-task reinforcement learning scenarios

- **`train_muzero_segment.py`** - MuZero training with segment collector and buffer reanalyze
  - Uses `MuZeroSegmentCollector` for data collection
  - Supports buffer reanalyze trick for improved sample efficiency
  - Supported algorithms: MuZero, EfficientZero, Sampled MuZero, Sampled EfficientZero, Gumbel MuZero, StochasticMuZero

- **`train_muzero_with_gym_env.py`** - MuZero training adapted for Gym environments
  - Specifically designed for OpenAI Gym-style environments
  - Simplifies environment interface adaptation

- **`train_muzero_with_reward_model.py`** - MuZero training with reward model
  - Integrates external Reward Model
  - Suitable for scenarios requiring learning complex reward functions

- **`train_muzero_multitask_segment_ddp.py`** - MuZero multi-task distributed training
  - Supports multi-task learning
  - Uses DDP (Distributed Data Parallel) for distributed training
  - Uses Segment Collector

#### UniZero Family
- **`train_unizero.py`** - Training entry for UniZero algorithm
  - Based on paper "UniZero: Generalized and Efficient Planning with Scalable Latent World Models"
  - Enhanced planning capabilities for better long-term dependency capture
  - Uses scalable latent world models
  - Paper: https://arxiv.org/abs/2406.10667

- **`train_unizero_segment.py`** - UniZero training with segment collector
  - Uses `MuZeroSegmentCollector` for efficient data collection
  - Supports buffer reanalyze trick

- **`train_unizero_multitask_segment_ddp.py`** - UniZero/ScaleZero multi-task distributed training
  - Supports multi-task learning and distributed training
  - Includes benchmark score definitions (e.g., Atari human-normalized scores)
  - Supports curriculum learning strategies
  - Uses DDP for training acceleration

- **`train_unizero_multitask_balance_segment_ddp.py`** - UniZero/ScaleZero balanced multi-task distributed training
  - Implements balanced sampling across tasks in multi-task training
  - Dynamically adjusts batch sizes for different tasks
  - Suitable for scenarios with large task difficulty variations

- **`train_unizero_multitask_segment_eval.py`** - UniZero/ScaleZero multi-task evaluation training
  - Specialized for training and periodic evaluation in multi-task scenarios
  - Includes detailed evaluation metric statistics

- **`train_unizero_with_loss_landscape.py`** - UniZero training with loss landscape visualization
  - For training with loss landscape visualization
  - Helps understand model optimization process and generalization performance
  - Integrates `loss_landscapes` library

#### ReZero Family
- **`train_rezero.py`** - Training entry for ReZero algorithm
  - Supports ReZero-MuZero and ReZero-EfficientZero
  - Improves training stability through residual connections
  - Paper: https://arxiv.org/pdf/2404.16364

### 🎓 Evaluation Entries

- **`eval_alphazero.py`** - Evaluation entry for AlphaZero
  - Loads trained AlphaZero models for evaluation
  - Can play against other agents for performance testing

- **`eval_muzero.py`** - Evaluation entry for MuZero family
  - Supports evaluation of all MuZero variants
  - Provides detailed performance statistics

- **`eval_muzero_with_gym_env.py`** - MuZero evaluation for Gym environments (not recently maintained)
  - Specialized for evaluating models trained in Gym environments


## 📖 Usage Guide

### Basic Usage Pattern

All training entry functions follow a similar calling pattern:

```python
from lzero.entry import train_muzero

# Prepare configuration
cfg = dict(...)  # User configuration
create_cfg = dict(...)  # Creation configuration

# Start training
policy = train_muzero(
    input_cfg=(cfg, create_cfg),
    seed=0,
    model=None,  # Optional: pre-initialized model
    model_path=None,  # Optional: pretrained model path
    max_train_iter=int(1e10),  # Maximum training iterations
    max_env_step=int(1e10),  # Maximum environment steps
)
```

### Choosing the Right Entry Function

1. **Single-Task Learning**:
   - Board games → `train_alphazero`
   - General RL tasks → `train_muzero` or `train_unizero`
   - Gym environments → `train_muzero_with_gym_env` (not recently maintained)

2. **Multi-Task Learning**:
   - Standard multi-task → `train_unizero_multitask_segment_ddp`
   - Balanced task sampling → `train_unizero_multitask_balance_segment_ddp`

3. **Distributed Training**:
   - All entry functions with `_ddp` suffix support distributed training

4. **Special Requirements**:
   - Loss landscape visualization → `train_unizero_with_loss_landscape`
   - External reward model → `train_muzero_with_reward_model`
   - Improved training stability → `train_rezero`

## 🔗 Related Resources

- **AlphaZero**: [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)
- **MuZero**: [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)
- **EfficientZero**: [Mastering Atari Games with Limited Data](https://arxiv.org/abs/2111.00210)
- **UniZero**: [Generalized and Efficient Planning with Scalable Latent World Models](https://arxiv.org/abs/2406.10667)
- **ReZero**: [Boosting MCTS-based Algorithms by Reconstructing the Terminal Reward](https://arxiv.org/abs/2404.16364)
- **ScaleZero**: [One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning](https://arxiv.org/abs/2509.07945)

## 💡 Tips

- Recommended to start with standard `train_muzero` or `train_unizero`
- For large-scale experiments, consider using DDP versions for faster training
- Using `_segment` versions can achieve better sample efficiency (via reanalyze trick)
- Check configuration examples in `zoo/` directory to learn how to set up each algorithm

## 📝 Notes

1. All path parameters should use **absolute paths**
2. Pretrained model paths typically follow format: `exp_name/ckpt/ckpt_best.pth.tar`
3. When using distributed training, ensure `CUDA_VISIBLE_DEVICES` environment variable is set correctly
4. Some entry functions have specific algorithm type requirements - check function documentation
