# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LightZero is a lightweight, efficient MCTS (Monte Carlo Tree Search) + Deep RL framework. This fork implements **UniZero PPO** - replacing MCTS search with PPO (Proximal Policy Optimization) training while preserving UniZero's Transformer-based world model architecture.

## Quick Start

### Environment Setup
```bash
cd /mnt/shared-storage-user/tangjia/unizero_ppo/LightZero
source /mnt/shared-storage-user/tangjia/miniconda3/bin/activate ppo
```

### Running Experiments

**LunarLander PPO (current debugging):**
```bash
python3 -u zoo/box2d/lunarlander/config/lunarlander_disc_unizero_ppo_online_config.py
```

**Standard MuZero (CartPole):**
```bash
python3 -u zoo/classic_control/cartpole/config/cartpole_muzero_config.py
```

**Atari (Pong):**
```bash
python3 -u zoo/atari/config/atari_muzero_segment_config.py
```

### Testing
```bash
# Run specific test
python3 -u lzero/policy/tests/test_utils.py

# Run environment test
python3 -u zoo/box2d/lunarlander/envs/test_lunarlander_env.py
```

## Architecture Overview

### Core Components

**Three-Layer Architecture:**
1. **Model** (`lzero/model/`) - Network structures (representation, dynamics, prediction)
2. **Policy** (`lzero/policy/`) - Training/collection/evaluation logic
3. **MCTS** (`lzero/mcts/`) - Tree search (ptree=Python, ctree=C++)

**Training Pipeline:**
```
Config → train_unizero → Collector → Buffer → Policy.learn → WorldModel.loss → Backprop
```

### Key Directories

- `lzero/entry/` - Training entry points (train_unizero.py, train_muzero.py)
- `lzero/policy/` - Policy implementations (unizero.py, muzero.py, etc.)
- `lzero/model/` - Neural network models
- `lzero/worker/` - Data collectors (muzero_collector.py)
- `lzero/mcts/buffer/` - Replay buffers (game_buffer_unizero.py, game_segment.py)
- `zoo/` - Environment configs and wrappers

## UniZero PPO Modifications

### Core Concept
Replace MCTS planning with PPO policy gradient training while keeping UniZero's world model for representation learning.

### Modified Files

**Training Entry:**
- `lzero/entry/train_unizero.py` - Main loop with mixed sampling (new/old data separation)

**Policy:**
- `lzero/policy/unizero.py` - `_forward_learn` unpacks PPO data (advantage, old_log_prob, return); `_forward_collect` supports pure policy mode (skips MCTS)
- `lzero/policy/utils.py` - Added `ppo_error`, `ppo_policy_error`, `ppo_value_error`

**World Model:**
- `lzero/model/unizero_world_models/world_model.py` - New `compute_loss_ppo()` method:
  - PPO clipped policy loss
  - Value loss (cross-entropy on returns_categorical)
  - Entropy loss
  - Obs/reward reconstruction loss

**Data Collection:**
- `lzero/worker/muzero_collector.py` - Stores policy_logits as old_log_prob; computes GAE via `_batch_compute_gae_for_pool()` after episode ends; supports value normalization

**Buffer:**
- `lzero/mcts/buffer/game_buffer.py` - Added `latest_push_count`, `new_data_ratio`
- `lzero/mcts/buffer/game_buffer_unizero.py` - `_make_batch` extracts advantage/old_log_prob/return with padding; `sample()` supports new/old data separation
- `lzero/mcts/buffer/game_segment.py` - Added `episode_id`, `advantage_segment`, `old_log_prob_segment`, `return_segment`; `pad_over()` handles old_log_prob padding

**Configs:**
- `zoo/atari/config/atari_unizero_ppo_config.py`
- `zoo/box2d/lunarlander/config/lunarlander_disc_unizero_ppo_config.py`
- `zoo/box2d/lunarlander/config/lunarlander_disc_unizero_ppo_online_config.py` (current)

### PPO Data Flow

```
Collector
  ├─ policy.forward_collect() → pure policy mode (action + policy_logits + pred_value)
  ├─ Store policy_logits → game_segment.old_log_prob_segment (raw logits, not log_prob)
  ├─ Store pred_value → game_segment.root_value_segment
  ├─ Episode end → _batch_compute_gae_for_pool() computes GAE advantage and return
  └─ Output game_segments → replay_buffer

Buffer._make_batch()
  ├─ Extract obs, action, mask, timestep from game_segment
  ├─ Extract advantage_segment, old_log_prob_segment, return_segment
  └─ Assemble current_batch (11 elements)

Policy._forward_learn()
  ├─ Unpack current_batch: obs, action, ..., advantage, old_log_prob, return
  ├─ Convert return → categorical distribution
  └─ Call world_model.compute_loss_ppo()

WorldModel.compute_loss_ppo()
  ├─ Obs encoding + transformer forward
  ├─ Obs reconstruction loss + reward prediction loss
  ├─ PPO policy loss: ratio = exp(log_prob_new - log_prob_old), clipped surrogate
  ├─ PPO value loss: cross-entropy on returns_categorical
  ├─ entropy loss
  └─ total_loss = obs_loss + reward_loss + policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
```

### Important Notes

- **old_log_prob storage:** Collector stores raw logits, not log probabilities
- **GAE computation:** Happens in collector after episode completion via `_batch_compute_gae_for_pool()`
- **Value normalization:** Supported in collector for stable training
- **Batch structure:** current_batch has 11 elements (last 3 are advantage, old_log_prob, return for PPO)

## Known Issues

**Current debugging:** LunarLander PPO not converging on `lunarlander_disc_unizero_ppo_online_config`. Need to investigate PPO-related bugs.

**Backup:** `/mnt/shared-storage-user/tangjia/unizero_ppo/LightZero-bak/`

## Development Patterns

### Config Structure
Configs have two parts:
- `main_config` - Environment, policy, hyperparameters
- `create_config` - Types and import paths for env/policy

### Adding New Algorithms
1. Create model in `lzero/model/`
2. Create policy in `lzero/policy/` with `@POLICY_REGISTRY` decorator
3. Create config in `zoo/<env>/config/`
4. Register in `create_config` with proper import_names

### Debugging Tips
- Check `data_unizero_ppo/` for training logs and checkpoints
- Use `docs/lunarlander_ppo_online_flow.md` for detailed PPO flow
- GAE computation details in `GAE_COMPUTATION_PLAN.md`
- Value/reward transforms use scalar_transform + phi_transform → categorical

## Installation

```bash
git clone https://github.com/opendilab/LightZero.git
cd LightZero
pip3 install -e .
```

Supports Linux and macOS. Compiles Cython extensions for performance-critical MCTS code.

## Documentation

- Main docs: https://opendilab.github.io/LightZero/
- Customize environments: `docs/source/tutorials/envs/customize_envs.md`
- Customize algorithms: `docs/source/tutorials/algos/customize_algos.md`
- Config guide: `docs/source/tutorials/config/config.md`
