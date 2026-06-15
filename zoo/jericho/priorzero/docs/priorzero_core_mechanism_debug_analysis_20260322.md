# PriorZero 核心机制与调试分析文档

> 生成时间: 2026-03-22 | 基于最新代码（含 PR #441 重构，VL 模型支持、rollout_logprob 重命名、样本去重、扩展训练指标）

---

## 1. PriorZero 整体架构与数据流

### 1.1 Actor-Critic 交互流程

PriorZero 在 Jericho 文本冒险环境下，采用 **WM-LLM (World Model) + Policy LLM** 双模型协同架构：

- **WM-LLM (World Model)**：基于 UniZero 的 transformer-based world model，负责环境建模、value 预测、policy logits 生成
- **Policy LLM**：基于 Qwen2.5 系列（含 VL 变体）的因果语言模型，通过 PPO/GSPO 进行策略优化，输出动作的 token-level log-probability。最新代码通过 `AutoConfig` 自动检测 VL 模型并使用 `AutoModelForVision2Seq`（`actor.py:92-113`）

两者通过 **交替训练 (alternating training)** 机制协调：先训练 WM 若干轮，再训练 LLM 若干轮，循环往复。

### 1.2 核心数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ROLLOUT (Rank 0)                             │
│                                                                     │
│  Jericho Env ──→ raw_obs_text, valid_actions, history               │
│       │                                                             │
│       ▼                                                             │
│  DataProcessor.get_llm_prior()                                      │
│    ├─ [可选] _build_cot_prefix_texts() → CoT reasoning prefix       │
│    ├─ _score_labels_with_prompt_logprobs() → per-action logprob     │
│    │   (vLLM prompt_logprobs=1, 拼接 context+label 后提取)           │
│    └─ 返回: llm_prior_per_seq, llm_prior_per_tok, cot_prefixes     │
│       │   (tok_dict 中 key 为 'rollout_action_logprob')              │
│       ▼                                                             │
│  Policy._forward_collect(llm_prior_logprob=...)                     │
│    ├─ WM initial_inference() → wm_policy_logits, wm_value          │
│    ├─ 融合 LLM + WM logits (fixed/adaptive 加权)                    │
│    └─ MCTS search → 选择动作                                        │
│       │                                                             │
│       ▼                                                             │
│  GameSegment.append(raw_obs, history, llm_prior_per_tok,            │
│                     cot_prefix, llm_action)                         │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BUFFER (Rank 0)                                   │
│                                                                     │
│  PriorZeroGameBufferOptimized                                       │
│    ├─ push_game_segments(new_data)                                  │
│    ├─ sample(batch_size) → WM 训练数据                               │
│    └─ fetch_latest_batch() → LLM 训练数据 (priorzero_batch)          │
│         返回: (raw_obs_list, history_obs_list,                       │
│                llm_prior_per_tok_list, target_value,                 │
│                pred_value, cot_prefix_list, llm_action_list)         │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│              TRAIN (All Ranks via broadcast)                        │
│                                                                     │
│  ── WM Phase ──                                                     │
│  learner.train(train_data) → WM losses (obs, reward, policy, value) │
│                                                                     │
│  ── LLM Phase ──                                                    │
│  1. bcast_obj(priorzero_batch) → 广播到所有 rank                     │
│  2. DataProcessor.make_llm_train_samples()                          │
│     ├─ build_llm_samples() → advantage = target_value - pred_value  │
│     ├─ unique_dicts_hash() 去重（datafactory.py:46-58）              │
│     ├─ advantage normalization (batch_norm / running_norm)           │
│     ├─ [可选] format_reward 融合                                     │
│     └─ tokenize + pad → 返回 (flag, (input_ids, attn_mask,          │
│           action_mask, advantage, rollout_logprob, log_status))      │
│  3. PriorZeroLLMTrainer.train_batch()                               │
│     ├─ PolicyModel.forward() → old_action_log_probs (当前策略)       │
│     ├─ [可选] ReferenceModel.forward() → ref_log_probs              │
│     ├─ PolicyModel.fit(batch_data, kl_ctl)                          │
│     │   └─ BatchPPOTrainer.train_batch()                            │
│     │       ├─ Actor.forward() → action_log_probs                   │
│     │       ├─ PolicyLoss(log_probs, old_log_probs, advantages,     │
│     │       │             action_mask, rollout_log_probs)            │
│     │       │   → actor_loss, clipfrac, approx_kl, vllm_kl          │
│     │       ├─ KL loss (vs reference model)                         │
│     │       ├─ Entropy loss                                         │
│     │       ├─ 扩展指标: ratio_mean/std, adv_mean/std,              │
│     │       │            log_prob_new/old_mean, kl_coef, total_loss  │
│     │       └─ backward + optimizer_step                            │
│     └─ broadcast_to_vllm() → 同步权重到 vLLM engine                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 WM-LLM 角色流转

| 阶段 | WM 角色 | Policy LLM 角色 |
|------|---------|-----------------|
| WM Warmup | 训练中（obs/reward/policy/value loss） | 冻结，仅用 vLLM 提供 prior |
| WM Phase | 训练中 | 冻结，仅用 vLLM 提供 prior |
| LLM Phase | 冻结（提供 target_value, pred_value） | 训练中（PPO/GSPO loss） |
| Collect | 推理（initial_inference） | 推理（vLLM 计算 prior） |

关键控制逻辑在 `priorzero_entry_sync.py:242-294`：
```python
# WM phase
if llm_cfg.enable_world_model and current_phase == "wm":
    for i in range(update_per_collect):
        train_data = replay_buffer.sample(batch_size, policy)
        learner.train(train_data)
    if learner.train_iter - last_wm_train_iter >= train_schedule["wm_update_iters"]:
        current_phase = "llm"

# LLM phase
if llm_cfg.enable_rft and current_phase == "llm":
    priorzero_batch = replay_buffer.fetch_latest_batch(batch_size=-1, policy=policy)
    torch.cuda.empty_cache()  # 清理 policy 的 cache，防止 OOM（entry_sync.py:264）
    flag, train_samples = data_processor.make_llm_train_samples(priorzero_batch, ...)
    if not flag:  # 样本不足时跳过（entry_sync.py:282）
        continue
    trainer.train_batch(train_samples)
    if trainer.global_step - last_llm_train_iter >= train_schedule["llm_update_iters"]:
        current_phase = "wm"
        data_processor.value_normalizer.clear()  # ← 关键：切换回 WM 时清空 normalizer
```

---

## 2. 核心机制深度解析

### 2.1 Value Normalization (`value_running_norm`)

**代码位置**：`models/stability_optimizer.py:AdaptiveValueNormalizer`，调用点在 `priorzero_datafactory.py:368-440`

#### 更新逻辑

1. **输入**：`advantage = target_value - pred_value`（由 WM 提供的 TD bootstrap value 差）
2. **裁剪**（可选）：
   - `soft`：`f(x) = sign(x) * log(1 + |x|)` → 压缩极端值但保留符号（`stability_optimizer.py:47-51`）
   - `hard`：分位数裁剪，保留 `[2.5%, 97.5%]` 区间（`stability_optimizer.py:53-66`）
3. **EMA 统计量更新**：
   ```python
   # stability_optimizer.py:102-108
   momentum = init_momentum + (final_momentum - init_momentum) * min(update_count / warmup_steps, 1.0)
   if update_count == 0:
       running_mean = batch_mean
       running_std = batch_std
   else:
       running_mean = momentum * running_mean + (1 - momentum) * batch_mean
       running_std  = momentum * running_std  + (1 - momentum) * batch_std
   ```
4. **归一化**：`y = (x - running_mean) / (running_std + 1e-6)`（`stability_optimizer.py:114`）

#### 清空机制及其影响

**清空时机**：`priorzero_entry_sync.py:293-294`
```python
if train_alternate and trainer.global_step - last_llm_train_iter >= train_schedule["llm_update_iters"]:
    current_phase = "wm"
    if data_processor.value_normalizer is not None:
        data_processor.value_normalizer.clear()  # reset running_mean=0, running_std=1, update_count=0
```

**`clear()` 实现**（`stability_optimizer.py:146-155`）：
```python
def clear(self):
    self.running_mean = 0.0
    self.running_std = 1.0
    self.update_count = 0
    self.value_history.clear()
```

**影响分析**：
- **正面**：每轮 WM 训练后 value 分布可能显著变化（WM 学到了新东西），清空 EMA 防止旧统计量产生 stale bias
- **负面风险**：清空后第一个 batch 的 `update_count=0`，直接用 batch 统计量初始化 running stats。若该 batch 恰好含极端值，会导致归一化后的 advantage 尺度不稳定
- **调试建议**：监控每次 `clear()` 后首个 batch 的 `norm_min/norm_max`，若出现极端值（>10 或 <-10），考虑在 clear 后保留 `running_std` 的下界

### 2.2 Advantage 计算与截断

**代码位置**：`priorzero_datafactory.py:345-442`

#### 计算方式

**非 GAE**，而是直接的 TD-error：
```python
# priorzero_datafactory.py:349
advantage = target_value - pred_value
```
其中：
- `target_value[t]`：从时刻 t 开始的 `td_step` 步真实奖励折扣和 + bootstrap `V(t + td_step)`
- `pred_value[t]`：WM 在时刻 t 的 value 预测 `V(t)`

#### 三种归一化模式

| 模式 | 代码位置 | 说明 |
|------|---------|------|
| `advantage` | `datafactory.py:351-356` | 原始值不变，最简单但尺度不可控 |
| `advantage_batch_norm` | `datafactory.py:359-366` | `(adv - mean) / (std + 1e-8)` 当前 batch 归一化 |
| `advantage_running_norm` | `datafactory.py:368-440` | `AdaptiveValueNormalizer`（EMA + soft/hard clip）或 fallback 手动 EMA |

#### 截断处理

- **soft clip**：`sign(x) * log(1 + |x|)`，阈值判定 `|x| > 10` 时计入 `clipped_count`
- **hard clip**：分位数 `[2.5%, 97.5%]`，前 `hard_clip_start_updates=10` 次不启用
- **注意**：截断在归一化 **之前** 应用，先压缩极端值再计算统计量

#### Format Reward 融合（可选）

```python
# priorzero_datafactory.py:354-356
if fmt_rewards is not None:
    advantage = (1 - fmt_weight) * advantage + fmt_weight * fmt_rewards
```
`fmt_rewards` 为 0/1 二值，检查输出是否符合 `Reasoning: ...\nAction: ...` 格式（`_format_reward()` at `datafactory.py:17-42`）。

#### 样本去重机制

最新代码在 `make_llm_train_samples()` 中增加了基于 hash 的去重：
```python
# datafactory.py:294-304
if len(samples) >= max_samples:
    unique_samples = unique_dicts_hash(samples)  # MD5 hash 去重
    if len(unique_samples) >= max_samples:
        samples = unique_samples[:max_samples]
    else:
        remain = max_samples - len(unique_samples)
        samples = unique_samples + samples[:remain]  # 不足时用重复样本补齐
else:
    return False, samples  # 样本不足，返回 flag=False
```

`unique_dicts_hash()`（`datafactory.py:46-58`）通过 `pickle.dumps` + `md5` 对每个样本 dict 做去重。

### 2.3 Importance Sampling (IS) 与 `clipfrac`

**代码位置**：`models/loss.py:PolicyLoss`

#### IS Ratio 计算

当前代码中存在 **三层 logprob**，理解其区别至关重要：

| 变量名 | 来源 | 含义 |
|--------|------|------|
| `rollout_action_logprob` | vLLM 在 collect 时计算 | `π_rollout(a|s)` — rollout 策略的 logprob |
| `old_action_log_probs` | `PolicyModel.forward()` 在 LLM 训练开始前计算 | `π_θ_old(a|s)` — 当前 epoch 开始时的策略 |
| `action_log_probs` | `Actor.forward()` 在每个 micro-batch 中计算 | `π_θ(a|s)` — 正在更新中的策略 |

PPO 标准 ratio（`loss.py:52-54`）：
```python
log_ratio = log_probs - old_log_probs      # π_θ / π_θ_old（同一 epoch 内的变化）
ratio = log_ratio.exp()
```

vLLM IS correction ratio（`loss.py:84-97`，仅 `enable_vllm_is_correction=True` 时）：
```python
vllm_is = exp(old_log_probs - rollout_log_probs)  # π_θ_old / π_rollout（跨 epoch 的偏移）
vllm_is = vllm_is.clamp(low_threshold, high_threshold)
loss = vllm_is * loss  # 修正 off-policy 偏差
```

#### PPO Clipped Surrogate Loss

```python
# loss.py:68-73
surr1 = ratio * advantages
surr2 = ratio.clamp(1 - eps_low, 1 + eps_high) * advantages  # 默认 [0.8, 1.2]
loss = -torch.min(surr1, surr2)
```

Dual-clip 变体（`loss.py:74-80`）：当 advantage < 0 时额外增加下界 `dual_clip * advantages`。

ICEPOP 变体（`loss.py:86-90`）：区间外的 IS 权重直接置零（而非 clamp）。

#### `clipfrac` 指标含义

```python
# loss.py:104-105
clipped = ratio.gt(1 + eps_high) | ratio.lt(1 - eps_low)
clipfrac = masked_mean(clipped, action_mask, dim=None)
```

- **含义**：token 级别的 IS ratio 落在 `[1-ε, 1+ε]` 区间 **之外** 的比例
- **健康值**：`clipfrac ∈ [0.05, 0.3]`
  - `< 0.05`：策略更新太保守，学习效率低
  - `> 0.5`：策略偏移严重，PPO clip 大量生效，可能导致训练不稳定
- **相关指标**：`clip_ratio = P(surr2 < surr1)` 表示 clip 实际约束了多少 loss

#### `approx_kl` 计算

```python
# loss.py:108
approx_kl = masked_mean(-log_ratio.detach(), action_mask, dim=None)
```
即 `E[-log(π_θ/π_old)] ≈ KL(π_old || π_θ)`，Schulman k1 近似。

#### 新增训练指标（`actor.py:324-365`）

最新代码在 `BatchPPOTrainer.train_batch()` 中新增了以下诊断指标：

| 指标 | 计算方式 | 诊断价值 |
|------|---------|---------|
| `ratio_mean` | `masked_mean(exp(log_probs - old_log_probs))` | IS ratio 均值，健康值 ≈ 1.0 |
| `ratio_std` | IS ratio 的标准差 | 偏移幅度，过大说明策略变化剧烈 |
| `advantage_mean/std` | 当前 micro-batch 的 advantage 统计 | 监控 advantage 分布 |
| `log_prob_new_mean` | 当前策略 log_prob 均值 | 策略信心度 |
| `log_prob_old_mean` | 旧策略 log_prob 均值 | 基线参考 |
| `total_loss` | `actor_loss + kl_loss * kl_coef - entropy * entropy_coef` | 含所有正则项的完整 loss |
| `kl_coef` | `float(kl_ctl.value)` | 当前 KL penalty 系数 |
| `vllm_kl` | `masked_mean(rollout_logprobs - old_logprobs)` | vLLM IS 校正时的 KL 散度 |

### 2.4 异步采样控制 (`max_rollout_staleness`)

**代码位置**：`priorzero_entry_sync.py:280`

#### 控制逻辑

```python
llm_need_sample_cnt = llm_cfg.train_batch_size * llm_cfg.max_rollout_staleness // 1
flag, train_samples = data_processor.make_llm_train_samples(priorzero_batch, max_samples=llm_need_sample_cnt)
```

- `max_rollout_staleness` 控制 LLM 训练时允许使用多少倍于 `train_batch_size` 的样本
- 默认值 `1`：只用最近一次 collect 的数据量（= `train_batch_size` 个样本）
- 值越大，允许使用越多"旧"数据，提升样本效率但增加 off-policy 程度

#### 返回值变化

最新代码中 `make_llm_train_samples()` 返回 `(flag, data)` 元组（`datafactory.py:304, 462`）：
- `flag=True`：样本充足，`data` 为训练数据元组
- `flag=False`：样本不足（`< max_samples`），`data` 为原始 samples 列表（非训练格式）

调用方通过 `if not flag: continue` 跳过本轮 LLM 训练（`entry_sync.py:282-284`）。

#### 过时轨迹处理

当前实现中，过时数据不是通过时间戳丢弃的，而是通过 **buffer 的 `mark_latest_transitions_consumed()` + `fetch_latest_batch()`** 机制：

```python
# priorzero_entry_sync.py:287
replay_buffer.mark_latest_transitions_consumed()  # 标记当前数据已消费
```

`fetch_latest_batch(batch_size=-1)` 只返回自上次 `mark` 以来新增的数据。因此 `max_rollout_staleness` 实际控制的是**每次 LLM 训练使用的样本上限**，而非数据的"年龄"。

---

## 3. 三大 Bug/痛点排查指南

### 3.1 痛点一：Policy Loss 出现 NaN

**现象**：LLM 接近最优时 KL 变大，固定 LR 下后期 Loss 变 NaN。

#### 潜在原因 1：Value Normalizer 清空后首 batch 极端值

**风险点**：`priorzero_entry_sync.py:293-294` 调用 `value_normalizer.clear()` 后：
- `update_count` 重置为 0
- 首 batch 直接赋值 `running_mean = batch_mean, running_std = batch_std`
- 若 WM 刚训练完 value 分布剧变，首 batch 可能包含极端 advantage
- `stability_optimizer.py:114`: `y = (x - running_mean) / (running_std + 1e-6)` — 若 `running_std` 极小（batch 中所有 advantage 接近），归一化后值可能爆炸

**修复建议**：
```python
# 在 clear() 中保留 std 下界
def clear(self):
    self.running_mean = 0.0
    self.running_std = max(1.0, self.running_std * 0.5)  # 不完全重置 std
    self.update_count = 0
    self.value_history.clear()
```

#### 潜在原因 2：KL 散度计算中的数值溢出

**风险点**：`utils.py:60-94` 中的 `compute_approx_kl()`

```python
# k3 estimator (utils.py:88-91)
log_ratio = log_probs - log_probs_base  # 当策略偏移很大时，可能是很大的正/负数
log_ratio = -log_ratio
log_ratio = log_ratio.exp() - 1 - log_ratio  # exp(大正数) → Inf → NaN
```

虽然有 `log_ratio.clamp(min=-10, max=10)`（line 93），但 clamp 在 **最后** 应用，此时 `exp()` 可能已经溢出。

**修复建议**：将 clamp 移到 `exp()` 之前：
```python
if kl_estimator == "k3":
    log_ratio = log_probs.float() - log_probs_base.float()
    log_ratio = (-log_ratio).clamp(min=-10, max=10)  # 先 clamp 再 exp
    log_ratio = log_ratio.exp() - 1 + (log_probs.float() - log_probs_base.float())
```

#### 潜在原因 3：log_probs 在 bfloat16 下精度不足

**风险点**：`actor.py:150`
```python
output["logits"] = output["logits"].to(torch.float32)
```
虽然 logits 转了 float32，但 `log_probs_from_logits()` 中的 `flash_attn cross_entropy_loss` 路径（`utils.py:121`）可能在内部回退到低精度。

**排查方法**：利用最新代码中的扩展指标，在 `BatchPPOTrainer.train_batch()` 的 `actor.py:324-339` 处已自动记录 `ratio_mean/std`、`log_prob_new/old_mean`。观察这些指标是否出现 NaN/Inf 前兆：
```python
# 已有的指标（无需额外添加代码）
# ratio_mean ≈ 1.0 是健康的；>> 1 或 << 1 说明策略偏移严重
# log_prob_new_mean 与 log_prob_old_mean 的差值 ≈ approx_kl
```

若需更细粒度排查，可添加：
```python
# 在 actor_loss 计算后添加（actor.py:286 之后）
if torch.isnan(actor_loss) or torch.isinf(actor_loss):
    print(f"[NaN DEBUG] action_log_probs: min={action_log_probs.min()}, max={action_log_probs.max()}")
    print(f"[NaN DEBUG] old_log_probs: min={micro_batch['old_action_log_probs'].min()}, max={micro_batch['old_action_log_probs'].max()}")
    print(f"[NaN DEBUG] advantages: min={micro_batch['advantages'].min()}, max={micro_batch['advantages'].max()}")
```

#### 潜在原因 4：Advantage 极端值未被充分抑制

当 `advantage_type="advantage"`（无归一化）时，raw advantage 可能非常大。PPO ratio * advantage 的乘积导致梯度爆炸。

**排查**：监控 `value_advantage_max/min` 和新增的 `advantage_mean/std` 指标，若 `|adv| > 100` 需要启用 `advantage_running_norm`。

### 3.2 痛点二：LLM 与 vLLM 的 `logprob` 差异

**现象**：相同输入输出下，原生 LLM（`Actor.forward()`）和 vLLM（`_score_labels_with_prompt_logprobs()`）给出的 logprob 差异很大。

#### 差异来源 1：Temperature 处理不一致

- **vLLM 侧**：`priorzero_datafactory.py:609-611`
  ```python
  sampling_params = SamplingParams(temperature=self.temperature, ...)
  ```
  vLLM 的 `prompt_logprobs` 返回的是 **经过 temperature 缩放后** 的 logprob（`logit / T` 后做 log_softmax）

- **Actor 侧**：`actor.py:157`
  ```python
  log_probs = log_probs_from_logits(output["logits"], rolled_sequences, temperature=self.temperature)
  ```
  `utils.py:112-113`：`logits.div_(temperature)` **原地修改**后做 log_softmax

- **风险**：如果两侧的 `temperature` 配置不一致（`llm_cfg.temperature` vs `strategy.args.temperature`），logprob 会系统性偏移。**特别注意**：`Actor.__init__` 的 `temperature` 来自 `strategy.args.temperature`（`actor.py:551`），`DataProcessor` 的 `self.temperature` 也来自 `strategy.args.temperature`（`datafactory.py:84`），理论上应一致，但需确认。

**排查方法**：
```python
# 在 train_batch 中对比
print(f"Actor temperature: {self.actor.temperature}")
print(f"vLLM SamplingParams temperature: {data_processor.temperature}")
```

#### 差异来源 2：Tokenization 对齐问题

- **vLLM 侧**：`priorzero_datafactory.py:618-636`
  ```python
  context_ids = tokenizer(all_context_texts, add_special_tokens=False, ...)["input_ids"]
  label_ids = tokenizer(label_texts, add_special_tokens=False, ...)["input_ids"]
  full_ids = [c + l for c, l in zip(context_ids, label_ids)]  # 手动拼接
  ```
  然后通过 `prompt_token_ids=full_ids` 传给 vLLM

- **Actor 侧**：`priorzero_datafactory.py:329`
  ```python
  inputs = self.tokenizer.pad({"input_ids": full_ids_list}, padding=True, return_tensors="pt")
  ```
  使用同一套 `full_ids`（从 sample 中取出的），通过 **左填充 (padding_side="left")** 对齐

- **关键风险**：**padding 引入的 attention_mask 差异**。vLLM 不做 padding，直接处理变长序列；Actor 做左 padding 但依赖 `attention_mask` 和 `position_ids` 正确排除 pad tokens。若 `position_ids` 计算有误（`actor.py:146-147`），会导致 logprob 偏移。

#### 差异来源 3：BOS Token 处理

- **vLLM**：`prompt_logprobs[0]` 是 `None`（第一个 token 无条件概率），从 `j=1` 开始提取（`datafactory.py:651`）
- **Actor**：`log_probs = log_probs[:, :-1]`（`actor.py:159`），即 logits 右移一位后取 log_softmax

两侧都跳过了第一个 token，理论一致。但如果 `apply_chat_template()` 在 vLLM 和 Actor 侧产生不同的 BOS/前缀 token，会导致 context 长度不同。

**排查方法**：利用新增的 `log_prob_new_mean` 和 `log_prob_old_mean` 指标，对比两者与 `rollout_action_logprob` 的差异：
```python
# 在 train_batch 开始时对比 token-level logprob
actor_lp = action_log_probs[0]  # [T_action]
vllm_lp = micro_batch['rollout_action_logprob'][0]  # [T_action]
mask = micro_batch['action_mask'][0]
print(f"Actor logprob (masked): {(actor_lp * mask).sum()}")
print(f"vLLM logprob (masked):  {(vllm_lp * mask).sum()}")
print(f"Diff per token: {((actor_lp - vllm_lp) * mask).abs().max()}")
```

#### 差异来源 4：vLLM model_impl 与 HF 实现差异

`vllm_engine.py` 配置 `model_impl="transformers"`，理论上使用与 HF 相同的模型实现。但 vLLM 的 attention kernel（即使用 eager 模式）、数值精度路径可能与 HF + flash_attention_2 存在微小差异。

**注意**：最新代码中 Actor 新增了 VL 模型支持（`actor.py:92-113`），若使用 VL 模型，vLLM 侧也需确保使用对应的 VL 推理路径。

### 3.3 痛点三：CoT (Chain of Thought) 融合优化

**需求**：在无 CoT 的最佳 config 基础上，加入 `weight=0.1` 的 CoT loss。

#### 当前 CoT 实现分析

**CoT 生成**：`priorzero_datafactory.py:464-516` (`_build_cot_prefix_texts()`)
- 使用 vLLM 生成 "Reasoning: ... \nAction:" 格式的推理前缀
- Stop condition: `"\n\n"`
- 生成后截取到 "Action:" 标记处

**CoT 融入训练**：`priorzero_datafactory.py:316-324`
```python
if self.use_cot:
    targets_only = [s["prefix_cot"] + " " + s["target"] + eos for s in real_samples]
    # 即训练 label = "Reasoning: <reasoning>\nAction: <action><eos>"
else:
    targets_only = ["Action: " + s["target"] + eos for s in real_samples]
```

**当前问题**：CoT 是一个全局开关 (`use_cot=True/False`)，没有支持 **部分权重** 融合。开启 CoT 后，**全部 label tokens 的 loss 权重相同**，CoT 推理部分和动作部分共享同一个 advantage。

#### 推荐的 CoT Loss 加权融合方案

**目标**：`total_loss = (1 - cot_weight) * action_loss + cot_weight * cot_loss`，其中 `cot_weight=0.1`。

**方案：在 `action_mask` 层面分离 CoT tokens 和 Action tokens**

代码修改点在 `priorzero_datafactory.py:make_llm_train_samples()`：

```python
# 在 line 336 处（action_mask 构建后），添加 CoT/Action 分离逻辑

if self.use_cot and hasattr(self.args, 'cot_loss_weight') and self.args.cot_loss_weight > 0:
    cot_weight = self.args.cot_loss_weight  # e.g., 0.1

    # 需要 label_ids_no_cots 信息，可在 build_llm_samples 中额外存储
    # 构建两套 mask
    cot_action_mask = action_mask.clone()  # 全部 label tokens
    pure_action_mask = torch.zeros_like(action_mask)

    for i, (tgt_ids, tgt_no_cot_ids) in enumerate(zip(tgt_ids_list, label_ids_no_cots_list)):
        no_cot_len = len(tgt_no_cot_ids)
        # pure_action_mask 只标记 Action 部分的 tokens
        pure_action_mask[i, -no_cot_len:] = action_mask[i, -no_cot_len:]

    # 加权 mask: CoT tokens 的 mask 值 = cot_weight, Action tokens 的 mask 值 = 1.0
    weighted_action_mask = pure_action_mask.float() + (cot_action_mask - pure_action_mask).float() * cot_weight
    action_mask = weighted_action_mask
```

这样在 `PolicyLoss.forward()` 的 `masked_mean(loss, action_mask)` 中，CoT tokens 的 loss 自然被降权到 0.1。

**更优雅的方案**：在 `BatchPPOTrainer.train_batch()` 中分开计算两个 loss 再加权，但需要传递额外的 `cot_mask`，改动量更大。

**配置添加**（在 `priorzero_config.py` 的 `PriorZeroLLMConfig` 中）：
```python
cot_loss_weight: float = 0.0  # 0 = 不使用 CoT loss, > 0 = CoT tokens 的 loss 权重
```

#### 需要同步修改的位置

1. `priorzero_config.py`: 添加 `cot_loss_weight` 字段
2. `priorzero_datafactory.py:make_llm_train_samples()`: 构建加权 `action_mask`
3. **注意**：需要保留 `label_ids_no_cots` 信息到 `make_llm_train_samples()` 阶段。当前代码在 `_score_labels_with_prompt_logprobs()` 中有 `l_no_cots_lens`（`datafactory.py:639`），但未传递到训练样本中。需要在 `build_llm_samples()` 中额外存储 `label_ids_no_cots`。

---

## 附录：关键变量速查表

| 变量/函数 | 文件:行号 | 说明 |
|-----------|----------|------|
| `AdaptiveValueNormalizer.clear()` | `stability_optimizer.py:146` | 重置所有 EMA 统计量 |
| `AdaptiveValueNormalizer.normalize()` | `stability_optimizer.py:83` | clip → batch_stats → EMA update → normalize |
| `PolicyLoss.forward()` | `loss.py:44` | PPO/GSPO loss + IS correction + clipfrac |
| `BatchPPOTrainer.__init__()` | `actor.py:221` | 初始化时传入 `enable_vllm_is_correction`, `vllm_is_truncated_threshold` |
| `BatchPPOTrainer.train_batch()` | `actor.py:251` | 微批次循环，累积梯度，含扩展指标 |
| `Actor.__init__()` | `actor.py:68` | VL 模型自动检测 (`AutoConfig` + `AutoModelForVision2Seq`) |
| `Actor.forward()` | `actor.py:135` | logits→float32→log_probs→action_log_probs |
| `PolicyModel.forward()` | `actor.py:615` | 分 chunk 推理，返回 `action_log_probs [B, T_action]` |
| `DataProcessor.make_llm_train_samples()` | `datafactory.py:272` | 返回 `(flag, data)` 元组；含去重逻辑 |
| `DataProcessor._score_labels_with_prompt_logprobs()` | `datafactory.py:606` | vLLM prompt_logprobs 提取，返回 `rollout_action_logprob` |
| `DataProcessor._build_cot_prefix_texts()` | `datafactory.py:464` | CoT reasoning prefix 生成 |
| `unique_dicts_hash()` | `datafactory.py:46` | 训练样本去重（pickle + MD5） |
| `compute_approx_kl()` | `utils.py:60` | KL 散度近似（k1/k2/k3） |
| `log_probs_from_logits()` | `utils.py:111` | logits → log_softmax（含 temperature） |
| `value_normalizer.clear()` 调用点 | `entry_sync.py:293-294` | LLM→WM 切换时清空 |
| `max_rollout_staleness` 使用点 | `entry_sync.py:280` | 控制 LLM 训练样本上限 |
| `_format_reward()` | `datafactory.py:17` | CoT 格式奖励（0/1） |
| `_normalize_vllm_weight_name()` | `actor.py:21` | vLLM 权重同步时的名称规范化 |
| `_should_skip_vllm_sync_param()` | `actor.py:28` | 跳过 LoRA adapter 参数不同步到 vLLM |
