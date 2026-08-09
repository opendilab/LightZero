## UniZero 世界模型中的位置编码

本节详细介绍了 UniZero 世界模型中所采用的位置编码策略，并就两种可配置选项进行了说明，其选择依据为配置参数 `self.config.rotary_emb` 的取值情况。

> **配置选项：**
> - 当 `self.config.rotary_emb = False` 时，采用 **绝对位置编码** （基于 `nn.Embedding`）。
> - 当 `self.config.rotary_emb = True` 时，采用 **旋转位置编码**（RoPE）。

---

### 1. 绝对位置编码（基于 `nn.Embedding`）

当配置参数 `self.config.rotary_emb` 为 **False** 时，模型使用 `nn.Embedding` 进行位置编码，其实现流程包括以下步骤：

#### 1.1 Embedding 层初始化

- **初始化：**  
  利用 `nn.Embedding` 初始化位置嵌入层，将序列中每个位置索引映射为固定尺寸的嵌入向量。

#### 1.2 上下文长度的限制

- **kv_cache 管理：**  
  由于受限于上下文长度（`context_length`），模型在缓存键值对（kv_cache）时只保留最近的 `<context_length>` 步，以保证计算效率与内存消耗处于可控范围内。

#### 1.3 滑动已满的 KV 窗口

不能通过给缓存的 Key/Value 加上“新旧位置投影之差”来精确平移 learned
absolute position 窗口。缓存的 Key/Value 来自经过归一化且依赖上下文的隐状态，
并不是位置嵌入的线性函数；反复使用这种代数修正只是一种近似，并会累积误差。

需要精确的 learned-absolute-position 推理时，可启用
`rebuild_kv_window_from_tokens`。UniZero 会为每个 KV 条目同步保留有界的原始
observation/action embedding；窗口前移时保留最近 token，从位置 0 开始重新编号，
并重新经过 Transformer 生成每一层的 Key/Value。raw-token 与 KV 存储使用相同的
淘汰和清理生命周期。

`exact_kv_window_reset` 是诊断用的替代方案：它只从最新 latent observation
重建。该方案避免错误的 K/V 代数修正，但会有意丢弃更早的上下文。旧的位置差
修正路径仅用于 checkpoint 对照，不应视为精确实现。

---

### 2. 旋转位置编码（RoPE）

当配置参数 `self.config.rotary_emb` 为 **True** 时，模型采用 ROPE（Rotary Position Embedding）进行位置编码。ROPE 的主要特点和实现流程如下：

#### 2.1 ROPE 初始化

- **预计算频率成分：**  
  使用提前计算出的频率成分，对查询（Query）和键（Key）的张量施加旋转位置嵌入，将位置信息直接融入自注意力计算中。

#### 2.2 基于剧集时间步的索引方式

- **索引方式：**  
  每个位置的索引基于剧集（episode）的时间步进行分配。  
  例如，在状态 (`s`) 和动作 (`a`) 交替出现的情况下，每个时间步占用两个位置索引。  
  假设一局游戏总共 50 步，其状态和动作依次为：  
  `(s₁, a₁, s₂, a₂, ..., s₅₀, a₅₀)`  
  
  则对应的位置索引为：  
  `1, 2, 3, 4, ..., 99, 100`

#### 2.3 ROPE 的原理

- **理论依据：**  
  ROPE 的设计灵感来源于论文 [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)。  
  该方法不仅通过旋转矩阵对绝对位置进行了编码，还在自注意力计算中直接融入了相对位置信息，从而实现了：
  - 更高的灵活性（序列长度可以灵活调整）；
  - 随着相对距离增加而逐渐衰减的 inter-token 依赖；
  - 能够兼容线性自注意力结构的相对位置编码。

### 3. 模式选择

learned absolute position 与 RoPE 的 checkpoint 参数不同，必须在训练前确定。
RoPE 的缓存 Key 在裁剪旧 token 后会保留原来的旋转，因此不需要 learned-absolute
K/V 的重新标定。不同任务应通过实验比较两种模式；仅凭依赖长度不足以预判性能。
当前 multitask world model 的 cache API 尚未传递每个 root 的 episode position，
因此会明确拒绝 RoPE，避免静默使用错误位置。

### 4. Replay Reanalysis 与 KV 上下文

UniZero 的 replay reanalysis 与 MuZero 并不等价。MuZero 的 recurrent state 是
自包含的，而 UniZero root 还依赖 Transformer KV prefix。现在 buffer 会恢复每个
采样 root 之前可用的 replay observation/action prefix，用对应 target tokenizer
编码，并执行与在线推理相同的有界 raw-token rolling。该次 prefix forward 同时产生
root KV 与 contextual policy prior，保证 root prior 和第一条 recurrent edge 描述
同一段历史。不同历史可能具有相同当前 latent，因此 root cache 写入相互隔离的
per-root init slot，而不是仅依赖 latent hash。

ordinary 与 sampled 两条 C++ replay 搜索路径都会按不超过在线环境 batch 的宽度
分块，从而不超过 recurrent-cache 容量，并恢复同一次搜索中 descendant 之间的 KV
命中。episode position 与 H+1 root 跨分块保持原顺序。当前不使用额外 task token
的多任务配置会选择对应 task tokenizer 与 prediction head；add/concat/register
task-token 的精确 raw-context 语义尚未实现，因此会明确拒绝而不是静默错算。
Atari 实验仍将 reanalysis 保持为可配置且默认关闭，因为实现正确并不等于已有性能收益。

### 5. 与在线上下文对齐的 TD Bootstrap Value

设置 `bootstrap_value_context`（CLI：`--bootstrap-value-context`）后，TD bootstrap
root 会使用在线 planning 真正可获得的 replay prefix 与 rolling window 估值。旧路径
中第一个 bootstrap root 没有此前历史，而后续 root 又可看到更长的 training-only
sequence，使 value target 依赖线上 planner 无法获得的状态信息。

contextual 路径直接用 target tokenizer 取得 root latent，只执行真正参与 target 的
context Transformer。首批及每 1000 批仍执行一次完整 legacy training-sequence
forward，用于记录两种 value 的均值、标准差、delta RMS/max 与 context 长度；其余
999 批跳过无用 forward。该优化不会改变 root latent 或 contextual target 数值。
多任务且不使用额外 task token 时，会选择对应 task tokenizer 与 value head。

### 6. Open-loop Latent 诊断

设置 `open_loop_diagnostic_freq > 0` 后，会按指定 learner 间隔执行无梯度诊断。
三条路径都在 eval mode 下运行，避免 dropout 污染比较：

- full teacher forcing 使用完整训练序列；
- rolling teacher forcing 使用线上 KV 滚窗规则，但持续输入真实后续 observation
  embedding，用于隔离窗口截断与 cache 语义误差；
- open-loop rolling 把每次预测的 latent 继续反喂，与 MCTS 的暴露分布一致。

`rolling_context_ratio` 是 rolling-teacher MSE / full-teacher MSE；
`open_loop_exposure_ratio` 是 open-loop MSE / rolling-teacher MSE；
`open_loop_total_ratio` 是 open-loop MSE / full-teacher MSE。以上指标只用于日志，
不会产生梯度，也不会改变训练 target。
当 `open_loop_prefix_transitions` 非零时，三条诊断路径使用相同的 teacher prefix
及 prefix 后 target 切片，因此日志中的 exposure 指标对应下方辅助损失实际训练的机制。

### 7. 可选 Open-loop Consistency Loss

设置 `open_loop_consistency_loss_weight > 0` 后，会增加一段可微的短 rollout：
把每一步预测 latent 继续反喂给 world model，并与 target encoder 的后续真实
observation embedding 对齐。这会直接训练 recurrent MCTS 实际使用的分布，而不再
只训练 teacher-forcing 路径。rollout 在保留梯度的同时使用 eval mode，并采用 raw
token 窗口重建；样本数和 horizon 均可配置。该损失是 batch 级辅助项，不再被 PER
importance weight 二次缩放。

`open_loop_prefix_transitions` 可在可微 rollout 前加入 replay 中的真实 transition。
例如 prefix=3 会构造 `[o0,a0,o1,a1,o2,a2,o3]`，它正好对应 10-token inference
cache 在下一 action 前保留的 7-token 稳态历史；监督 target 仍从 prefix 之后开始。
这样可以把“历史条件下的 exposure”与单纯增加 rollout horizon 区分开；默认值为 0。

默认权重为 0，因此不会改变现有训练。当前只支持 single-task、离散动作、learned
absolute position 且启用 raw-token KV 窗口重建的 world model。只有当诊断比率证明
主要误差来自自回归暴露、而不是窗口语义时，才应启用该项做受控验证。

### 8. 可选 MuZero-style Recurrent Loss

设置 `open_loop_recurrent_loss_weight > 0` 后，会在同一条 predicted-latent rollout
上补齐 MuZero recurrent learner 的监督语义：每个 action 后监督 latent 和 reward，
再把预测 observation 作为下一状态反喂，并在该预测状态上监督 policy 和 value。
各分量沿用 UniZero 主损失权重（latent 10、reward 1、value 0.5、policy 1），配置的
recurrent weight 再缩放合并后的辅助损失；batch size 与 horizon 复用 open-loop
consistency 的设置，可选 teacher prefix 也共用。

该选项已经包含 latent consistency，因此与 latent-only consistency 互斥。其支持边界
相同且默认权重为 0，不改变已有配置行为。
