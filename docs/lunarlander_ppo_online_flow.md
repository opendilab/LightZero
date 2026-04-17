# LunarLander UniZero PPO Online 运行流程与 Loss 计算

## 1. 配置入口

**文件**: `zoo/box2d/lunarlander/config/lunarlander_disc_unizero_ppo_online_config.py`

- 定义 `main_config`（环境、policy、PPO 超参等）和 `create_config`（env/policy 的 type 与 import_names）。
- `if __name__ == "__main__"` 时执行：
  - `from lzero.entry import train_unizero`
  - `train_unizero([main_config, create_config], seed=0, max_env_step=max_env_step)`

---

## 2. LightZero 主 Pipeline

**文件**: `lzero/entry/train_unizero.py`

流程概览：

1. **编译配置**: `compile_config(cfg, ..., create_cfg=create_cfg)`，会根据 create_cfg 导入 env 与 policy 模块。
2. **创建环境**: `get_vec_env_setting` + `create_env_manager` → collector_env、evaluator_env。
3. **创建策略**: `create_policy(cfg.policy, ...)` → 实例化 `UniZero` policy（learn/collect/eval 三种模式）。
4. **创建核心组件**:
   - `GameBuffer` = `UniZeroGameBuffer(cfg.policy)`（从 `lzero.mcts` 按 type 取）
   - `Collector` = `MuZeroCollector(env=collector_env, policy=policy.collect_mode, ...)`
   - `Evaluator` = `MuZeroEvaluator(...)`
   - `BaseLearner` 包装 `policy.learn_mode`，负责 train/checkpoint 等。
5. **主循环**（每次迭代）:
   - 可选：evaluator.eval(...)
   - `new_data = collector.collect(...)` 收集 trajectory
   - `replay_buffer.push_game_segments(new_data)` 把 segment 推入 buffer
   - `replay_buffer.remove_oldest_data_to_fit()` 控制容量
   - 若数据足够，则 `update_per_collect` 次：
     - `train_data = replay_buffer.sample(batch_size, policy)` 得到 (current_batch, target_batch) 或混合采样元组
     - `learner.train(train_data, ...)` → 内部调用 policy 的 learn 接口，最终进入 **policy._forward_learn**，进而调用 **world_model.compute_loss_ppo**
   - 若开启 online_learning，可在迭代末清空 buffer（当前代码里该段被注释）。
   - 终止条件：`collector.envstep >= max_env_step` 或 `learner.train_iter >= max_train_iter`。

---

## 3. Policy：UniZero

**文件**: `lzero/policy/unizero.py`

- 注册为 `POLICY_REGISTRY('unizero')`，create_config 中 `import_names=['lzero.policy.unizero']` 会触发该类被加载。
- 提供 `learn_mode` / `collect_mode` / `eval_mode`，内部是同一模型不同前向（learn / collect / eval）。
- **训练入口**: `_forward_learn(data)`：
  - `data = (current_batch, target_batch, train_iter)`。
  - current_batch 共 11 项：obs, action, bootstrap_action, mask, indices, weights, make_time, **timestep**, **advantage**, **old_log_prob**, **return**（后三项为 PPO 用）。
  - target_batch：target_reward, target_value, target_policy（用于 reward/value/policy 的 target）。
  - 将 reward/value/return 做 scalar_transform + phi_transform 得到 categorical，拼成 `batch_for_gpt`（含 observations, actions, timestep, rewards, target_value, rewards_categorical, target_value_categorical, mask_padding, **advantages**, **old_log_prob**, **returns**, **returns_categorical** 等）。
  - **调用**: `losses = self._learn_model.world_model.compute_loss_ppo(batch_for_gpt, tokenizer, value_inverse_scalar_transform_handle, clip_ratio=..., value_coef=..., entropy_coef=...)`
  - 用返回的 total loss 反传、更新参数。

即：**真正算 PPO + 世界模型 loss 的是 world_model.compute_loss_ppo**。

---

## 4. Collector：采集 Segment

**文件**: `lzero/worker/muzero_collector.py`

- 类名: `MuZeroCollector`（`SERIAL_COLLECTOR_REGISTRY('episode_muzero')`）。
- **collect()**: 与 env 交互，用 policy.collect_mode 选动作，按 episode 或 step 切分为 **GameSegment**（见 `lzero.mcts.buffer.game_segment`）。
- 每个 GameSegment 里会包含：obs、action、reward、value、policy 等；若启用 PPO，还会在 collector 或后续流程里填 **advantage_segment**、**old_log_prob_segment**、**return_segment**（GAE 与 return 一般在 collector 或 buffer 侧算好）。
- 返回的 `new_data` 是 list of (game_segment_list, meta)，交给 `replay_buffer.push_game_segments(new_data)`。

---

## 5. Buffer：UniZeroGameBuffer

**文件**: `lzero/mcts/buffer/game_buffer_unizero.py`

- 继承 `MuZeroGameBuffer`，存的是 **GameSegment** 列表（`game_segment_buffer`）。
- **push_game_segments(data)**: 把 collector 产生的 segment（及 meta）写入 buffer，必要时做容量控制。
- **sample(batch_size, policy)**:
  - 内部会调 `_sample_original` 或混合采样（如 `_sample_from_segment_range` 等）。
  - `_make_batch(...)` 从 segment 里按位置截取 **num_unroll_steps** 的 obs、action、mask、timestep，以及 PPO 需要的 **advantage_segment**、**old_log_prob_segment**、**return_segment**，拼成 **current_batch**（11 项）。
  - 再算 target：`_compute_target_reward_value`、`_compute_target_policy_*` 得到 target_reward、target_value、target_policy，组成 **target_batch**。
  - 返回 `[current_batch, target_batch]`（或 (train_data_new, train_data_old) 的混合采样格式，由具体配置决定）。

也就是说，**你“找出来的 segment”会先 push 进这个 buffer，训练时再从 buffer 里 sample 成 batch，供 policy._forward_learn 使用**。

---

## 6. World Model Loss 计算（重点）

**文件**: `lzero/model/unizero_world_models/world_model.py`

LunarLander 用的是 **PPO 训练**，因此实际用的是 **compute_loss_ppo**，而不是 **compute_loss_unizero**（后者里 policy/value 部分被置 0，仅作结构参考）。

### 6.1 `compute_loss_ppo` 的输入

- **batch**：来自 policy 组好的 `batch_for_gpt`，包含：
  - `observations`, `actions`, `timestep`, `mask_padding`
  - `rewards`, `rewards_categorical`, `target_value`, `target_value_categorical`
  - `returns`, `returns_categorical`（PPO 的 return 与分类形式）
  - `advantages`, `old_log_prob`（PPO 用）
- **target_tokenizer**：用于算 target 的 obs embedding（如 compute_labels_world_model）。
- **inverse_scalar_transform_handle**：分类 value 转回标量（若需要）。
- **clip_ratio, value_coef, entropy_coef**：PPO 超参。

### 6.2 `compute_loss_ppo` 的步骤概览

1. **编码与前向**
   - `obs_embeddings = tokenizer.encode_to_obs_embeddings(batch['observations'])`
   - action 做成 token，和 obs_embeddings 一起输入 **forward(obs_embeddings_and_act_tokens, start_pos=timestep)**
   - 得到 `outputs`：`logits_observations`, `logits_rewards`, `logits_value`, `logits_policy` 等。

2. **Observation 与 Reward Loss（与 compute_loss_unizero 一致）**
   - **vector**（LunarLander）：perceptual_loss=0，latent_recon_loss 用默认（如 self.latent_recon_loss）。
   - 用 target_tokenizer 得到 `target_obs_embeddings`，再 `compute_labels_world_model` 得到 `labels_observations`, `labels_rewards`。
   - **loss_obs**：MSE 或 group_kl（predict_latent_loss_type）在 `logits_observations` 与 `labels_observations` 上算，再乘 `mask_padding`。
   - **loss_rewards**：`compute_cross_entropy_loss(outputs, labels_rewards, batch, element='rewards')`。

3. **PPO Policy Loss**
   - 从 batch 取 `advantages`, `old_log_prob`, `actions`。
   - 用当前 `outputs.logits_policy` 建 Categorical（离散）或 Normal（连续），得到 `log_prob`, `entropy`。
   - `ratio = exp(log_prob - old_log_prob)`，surrogate = min(ratio * adv, clip(ratio) * adv)，**policy_loss = -masked_mean(clipped_surrogate)**。
   - 可选：approx_kl、clipfrac 等日志。

4. **PPO Value Loss**
   - 使用 **returns_categorical** 作为 target，与 compute_loss_unizero 对齐：`loss_value_elem = compute_cross_entropy_loss(outputs, returns_categorical, batch, element='value')`。
   - 按 timestep 做 **gamma^t 折扣** 再求平均：`discounted_loss_value = (loss_value_elem * discounts).sum() / (mask_padding.sum() + 1e-8)`。

5. **Entropy Loss**
   - `entropy_loss = -policy_entropy`（鼓励探索）。

6. **总 Loss**
   - 对 obs、rewards 同样做按步 discount（discounted_loss_obs, discounted_loss_rewards）。
   - **loss_total** = latent_recon_weight * discounted_loss_obs + discounted_loss_rewards + **policy_loss** + **value_coef * discounted_loss_value** + **entropy_coef * entropy_loss**。
   - 返回 `LossWithIntermediateLosses(...)`，包含各项子 loss 供日志与监控。

### 6.3 `compute_loss_unizero` 在本配置下的角色

- 仍实现完整的 obs / reward / value / policy 的 target 与 loss 形式（包括 returns_categorical、discount 等）。
- 但 **policy 相关项被置为 0**（注释里说明改用 compute_loss_ppo），value 用 returns_categorical。
- 当前 LunarLander PPO online 训练 **只调用 compute_loss_ppo**，不直接调用 compute_loss_unizero；compute_loss_unizero 更多是保留“纯 UniZero”的 loss 结构，便于对比或非 PPO 设置。

---

## 7. 文件与数据流小结

| 步骤           | 文件 | 作用 |
|----------------|------|------|
| 启动           | `zoo/box2d/lunarlander/config/lunarlander_disc_unizero_ppo_online_config.py` | 读入配置并调用 train_unizero |
| 主循环         | `lzero/entry/train_unizero.py` | 环境/策略/buffer/collector/learner 的创建与训练循环 |
| 策略与 learn   | `lzero/policy/unizero.py` | 组 batch，调用 world_model.compute_loss_ppo |
| 采集           | `lzero/worker/muzero_collector.py` | 产生 GameSegment（含 PPO 的 advantage/old_log_prob/return） |
| 存 segment     | `lzero/mcts/buffer/game_buffer_unizero.py` | push_game_segments 存 segment，sample 出 current_batch + target_batch |
| Loss 计算      | `lzero/model/unizero_world_models/world_model.py` | compute_loss_ppo：obs + reward + PPO policy + value + entropy |

整体数据流：**Config → train_unizero → Collector 采集 Segment → Buffer 存储并 sample → Policy._forward_learn 组 batch → WorldModel.compute_loss_ppo 计算 loss → 反传更新**。
