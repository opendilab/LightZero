# UniZero PPO Project

## 项目概述
本项目基于 LightZero 框架，将 UniZero（基于 Transformer 的 world model + MCTS）的训练方式改为 PPO（Proximal Policy Optimization）。核心思路是保留 UniZero 的 world model 架构，但用 PPO 的 policy gradient + GAE 替代 MCTS 搜索来训练策略。

## 启动命令
```bash
cd /mnt/shared-storage-user/tangjia/unizero_ppo/LightZero
source /mnt/shared-storage-user/tangjia/miniconda3/bin/activate ppo
python3 -u zoo/box2d/lunarlander/config/lunarlander_disc_unizero_ppo_online_config.py
```

## 修改的核心文件

### 训练入口
- `lzero/entry/train_unizero.py` — 训练主循环，支持混合采样（新旧数据分离）

### 策略
- `lzero/policy/unizero.py` — `_forward_learn` 增加 PPO 数据解包（advantage, old_log_prob, return），调用 `compute_loss_ppo`；`_forward_collect` 增加纯策略模式（跳过 MCTS）
- `lzero/policy/utils.py` — 新增 `ppo_error`, `ppo_policy_error`, `ppo_value_error` 函数

### World Model
- `lzero/model/unizero_world_models/world_model.py` — 新增 `compute_loss_ppo()` 方法，计算 PPO clipped policy loss + value loss + entropy loss + obs/reward reconstruction loss

### 数据收集
- `lzero/worker/muzero_collector.py` — 收集时存储 policy_logits（作为 old_log_prob）；episode 结束后批量计算 GAE（`_batch_compute_gae_for_pool`）；支持 value normalization

### Buffer
- `lzero/mcts/buffer/game_buffer.py` — 基类增加 `latest_push_count`, `new_data_ratio`
- `lzero/mcts/buffer/game_buffer_unizero.py` — `_make_batch` 增加 advantage/old_log_prob/return 的提取和 padding；`sample()` 支持新旧数据分离
- `lzero/mcts/buffer/game_segment.py` — 新增 `episode_id`, `advantage_segment`, `old_log_prob_segment`, `return_segment` 字段；`pad_over()` 支持 old_log_prob padding

### 配置
- `zoo/atari/config/atari_unizero_ppo_config.py` — Atari PPO 配置
- `zoo/box2d/lunarlander/config/lunarlander_disc_unizero_ppo_config.py` — LunarLander PPO 配置
- `zoo/box2d/lunarlander/config/lunarlander_disc_unizero_ppo_online_config.py` — LunarLander PPO online 配置（当前调试用）

## 数据流（PPO 模式）
```
Collector
  ├─ policy.forward_collect() → 纯策略模式，输出 action + policy_logits + pred_value
  ├─ 存储 policy_logits → game_segment.old_log_prob_segment（注意：存的是 raw logits，不是 log_prob）
  ├─ 存储 pred_value → game_segment.root_value_segment
  ├─ episode 结束后 → _batch_compute_gae_for_pool() 计算 GAE advantage 和 return
  └─ 输出 game_segments → replay_buffer

Buffer._make_batch()
  ├─ 从 game_segment 提取 obs, action, mask, timestep
  ├─ 提取 advantage_segment, old_log_prob_segment, return_segment
  └─ 组装 current_batch（11 个元素）

Policy._forward_learn()
  ├─ 解包 current_batch: obs, action, ..., advantage, old_log_prob, return
  ├─ 转换 return → categorical distribution
  └─ 调用 world_model.compute_loss_ppo()

WorldModel.compute_loss_ppo()
  ├─ obs encoding + transformer forward
  ├─ obs reconstruction loss + reward prediction loss
  ├─ PPO policy loss: ratio = exp(log_prob_new - log_prob_old), clipped surrogate
  ├─ PPO value loss: cross-entropy on returns_categorical
  ├─ entropy loss
  └─ total loss = obs_loss + reward_loss + policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
```

## 已知问题（不收敛）
当前在 lunarlander_disc_unizero_ppo_online_config 上不收敛，需要排查 PPO 相关的 bug。
备份版本在 `/mnt/shared-storage-user/tangjia/unizero_ppo/LightZero-bak/`。
