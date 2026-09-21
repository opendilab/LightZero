# UniZero 世界模型

本目录实现 UniZero 使用的 latent Transformer 世界模型。模型将 observation 编码为
latent token，并根据交错的 observation/action 历史预测下一 latent、reward、policy
和 value。

## 模型结构

标准 Atari 配置中，一个 transition 使用两个 token：

```text
[observation latent, action, observation latent, action, ...]
```

主要组件包括：

- `Tokenizer`：将 observation 编码为 latent observation token；
- `Transformer`：对 observation/action token 执行 causal attention；
- prediction heads：预测下一 latent、reward、policy 和 value；
- KV cache：在采集、评估和 MCTS 中复用 Transformer 历史。

`world_model.py` 负责核心模型与推理生命周期，其他功能按职责拆分为：

- `cache_window.py`：精确重建 KV 窗口的公共原语；
- `reanalysis_context.py`：replay root 历史重建及 contextual policy/value 计算；
- `open_loop.py`：open-loop 诊断和辅助训练目标；
- `world_model_multitask.py`：多任务扩展。

## 训练与推理

训练使用 teacher-forced observation/action 序列。Replay reanalysis 的 target 路径
会计算全部 `H+1` 个 observation root，包括真实 bootstrap 状态 `s[t+H]`；普通
learner 训练仍保留原来的 `H` 步计算，因为最后的占位 target 会被丢弃。

在线推理只保留 `context_length` 个 token 的有界上下文。KV cache 按环境隔离，异步
episode reset 不会清空其他环境的历史。当同一 batch 内有效历史长度不同时，cache
使用左 padding，并通过 attention mask 排除 padding，避免其影响预测。

## 位置编码与 KV 窗口

目前支持两种位置编码：

- `rotary_emb=False`：learned absolute position embedding；
- `rotary_emb=True`：根据 episode position 对 attention query/key 应用 RoPE。

learned-absolute KV 窗口不能通过累加位置投影差来精确平移，因为缓存的 K/V 非线性
依赖完整隐状态上下文。因此目前提供三种窗口行为：

| 配置 | 行为 |
| --- | --- |
| `rebuild_kv_window_from_tokens=True` | 同步保存有界的原始 embedded token，并通过重放精确重建滚动窗口。 |
| `exact_kv_window_reset=True` | 只从最新 latent 重建；适合诊断，但会有意丢弃更早历史。 |
| 两者均为 `False` | 使用 legacy 位置差修正路径，以兼容旧实验。 |

两种重建模式互斥。Raw-token 重建只用于 learned absolute position，与 RoPE 同时启用
会直接报错。发生窗口溢出时，所有相关样本会通过一次 batched Transformer forward
完成重建，再复制回各自独立的 cache。

RoPE key 在裁剪旧 token 后仍保留原有旋转，因此不需要 learned-position rebasing。
当前 multitask world model 尚未为每个 root 传递 episode position，所以会明确拒绝
RoPE，避免静默使用错误位置。

## Replay Reanalysis

UniZero replay root 不是自包含状态：当前 latent 和 Transformer prefix 共同定义状态。
当前实现始终保证：

- value/policy target 与真实 `H+1` observation root 对齐；
- C++ replay search 按不超过在线环境容量的宽度分块，避免 recurrent cache ring
  覆写仍在使用的搜索树；
- 跨 chunk 保持 root 顺序和 episode position 不变。

以下两个历史条件功能可选，且默认关闭：

- `contextual_reanalysis=True`：重建 replay root 的有界 observation/action prefix，
  并由同一次 prefix forward 同时产生 root policy prior 和 recurrent KV cache。
- `bootstrap_value_context=True`：使用在线 planning 实际可获得的 rolling replay
  context 计算 TD bootstrap value。

多任务模式在不添加 task token 时，可选择对应 task 的 tokenizer 和 prediction head。
add/concat/register task token 的精确 raw-context 语义尚未实现，因此会明确报错。

## 可选 Open-loop 诊断与训练

所有 open-loop 功能默认关闭。

设置 `open_loop_diagnostic_freq > 0` 后，会在关闭 dropout 的情况下比较三条路径：

- full teacher forcing；
- rolling teacher forcing：使用在线窗口，但输入真实后续 latent；
- open-loop rolling：将预测 latent 继续反喂给模型。

主要比率用于区分滚窗误差与自回归暴露误差：

```text
rolling_context_ratio    = rolling_teacher_mse / full_teacher_mse
open_loop_exposure_ratio = open_loop_mse / rolling_teacher_mse
open_loop_total_ratio    = open_loop_mse / full_teacher_mse
```

可以二选一启用以下辅助目标：

- `open_loop_consistency_loss_weight > 0`：在短可微 rollout 上监督预测 latent；
- `open_loop_recurrent_loss_weight > 0`：进一步在每个 action 后监督 reward，并在预测
  下一状态上监督 policy/value。

`open_loop_consistency_batch_size`、`open_loop_consistency_horizon` 和
`open_loop_prefix_transitions` 用于控制 rollout 开销与上下文。目前这些路径只支持
single-task、离散动作、单 observation token、learned absolute position，并要求
`rebuild_kv_window_from_tokens=True`。

## 默认行为

UniZero policy 的默认值为：

```python
context_length = 8                 # 四个 observation/action block
rotary_emb = False
exact_kv_window_reset = False
rebuild_kv_window_from_tokens = False
contextual_reanalysis = False
bootstrap_value_context = False
open_loop_consistency_loss_weight = 0.0
open_loop_recurrent_loss_weight = 0.0
open_loop_prefix_transitions = 0
```

环境配置可以为受控实验覆盖这些值。复现默认基线时应保持算法实验项关闭；评估新机制
时建议一次只改变一个变量。
