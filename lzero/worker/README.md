# Worker Module

This directory contains the worker components for LightZero's reinforcement learning algorithms, including data collectors and evaluators.

## File Overview

### Collector Files
Collectors are responsible for gathering experience data during training through environment interaction.

| File | Algorithm | Collection Mode | Description |
|------|-----------|----------------|-------------|
| `alphazero_collector.py` | AlphaZero | Episode-based | Collects complete game episodes for AlphaZero algorithm. Designed for perfect information games (e.g., board games). |
| `muzero_collector.py` | MuZero/EfficientZero/Gumbel MuZero | Episode-based | Collects complete game episodes for MCTS+RL algorithms. Supports both perfect and imperfect information environments. |
| `muzero_segment_collector.py` | MuZero/EfficientZero/Gumbel MuZero | Segment-based | Collects a specified number of game segments rather than complete episodes. Provides greater flexibility and extensibility. |

### Evaluator Files
Evaluators assess the performance of trained policies during the training process.

| File | Algorithm | Description |
|------|-----------|-------------|
| `alphazero_evaluator.py` | AlphaZero | Evaluates AlphaZero policy performance on test environments. |
| `muzero_evaluator.py` | MuZero/EfficientZero | Evaluates MuZero-based policy performance with support for multi-task scenarios. |

## Key Differences

### AlphaZero vs MuZero
- **AlphaZero**: Specifically designed for perfect information games where the full game state is observable (e.g., Go, Chess)
- **MuZero**: General-purpose algorithm supporting both perfect and imperfect information environments, with learned dynamics models

### Collector vs Evaluator
- **Collector**: Gathers training data through self-play or environment interaction
- **Evaluator**: Assesses policy performance at regular intervals during training

### MuZeroCollector vs MuZeroSegmentCollector
- **MuZeroCollector**: Returns data after collecting complete game episodes
- **MuZeroSegmentCollector**: Returns data after collecting a specified number of game segments, offering more fine-grained control over data collection

## Common Features

All workers support:
- Distributed training (multi-process/multi-GPU)
- TensorBoard logging
- Multi-task learning scenarios (via `task_id` parameter)
- Configurable collection/evaluation frequencies
- Environment and policy reset capabilities

## Usage Example

```python
from lzero.worker import MuZeroCollector, MuZeroEvaluator

# Initialize collector
collector = MuZeroCollector(
    collect_print_freq=100,
    env=env_manager,
    policy=policy,
    tb_logger=tb_logger,
    exp_name='my_experiment',
    policy_config=policy_config
)

# Initialize evaluator
evaluator = MuZeroEvaluator(
    eval_freq=1000,
    n_evaluator_episode=10,
    env=eval_env,
    policy=policy,
    tb_logger=tb_logger,
    exp_name='my_experiment'
)
```
