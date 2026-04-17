# GAE 计算与 Segment 数据对齐方案

## 📋 问题分析

### 当前问题

1. **Segment 结构复杂**：
   - 原始部分长度：`game_segment_length` (如 100)
   - Padding 部分：`num_unroll_steps + td_steps` (如 5+3=8)
   - 导致 reward_segment 和 root_value_segment 长度不一致

2. **数据重复**：
   - 一个 episode 分成多个 segments
   - 每个 segment 都包含了下一个 segment 的开头（padding）
   - 直接拼接会导致重复

3. **维度不匹配**：
   - advantage_segment：只有原始部分（100）
   - reward_segment：有原始 + padding（107）
   - Buffer sample 时超出范围会补 0，造成不一致

## ✅ 解决方案

### 方案概述

```
三个阶段：
1️⃣ 收集阶段（Collector）：只取原始部分，拼接不重复
2️⃣ 计算阶段（Collector）：正常计算 GAE
3️⃣ 存储阶段（Collector）：存 num_unroll_steps + td_steps 个
```

---

## 📝 详细实现

### 第 1 步：Collector 中拼接数据（避免重复）

**位置**：`muzero_collector.py` 第 1034-1057 行

**原理**：
- 每个 segment 有两部分：`[原始部分] + [padding 部分]`
- Padding 部分 = 下一个 segment 的开头（会被下一个 segment 也包含）
- **只取原始部分，避免重复**

**数据结构**（假设 game_segment_length=100）：

```
Segment 0:
  原始：reward[0:100], value[0:100]
  padding：reward[100:107], value[100:108]（来自 Seg1 的开头）

Segment 1:
  原始：reward[0:100], value[0:100]
  padding：reward[100:107], value[100:108]（来自 Seg2 的开头）

Segment 2 (最后一个):
  原始：reward[0:100], value[0:100]
  （没有 padding，或者补 0）
```

**改动**：

```python
for pool_idx, segment, _, _ in segments_info:
    # 只取原始部分，不要 padding
    original_seg_len = segment.game_segment_length  # 如 100
    
    # Extract only original part
    values = segment.root_value_segment[:original_seg_len]      # [0:100]
    rewards = segment.reward_segment[:original_seg_len]         # [0:100]
    
    all_values.extend(values.tolist())
    all_rewards.extend(rewards.tolist())
```

**结果**：
```
all_values = [Seg0: v0...v99] + [Seg1: v100...v199] + [Seg2: v200...v299]
all_rewards = [Seg0: r0...r99] + [Seg1: r100...r199] + [Seg2: r200...r299]

长度都是 300（无重复）
```

---

### 第 2 步：GAE 计算（标准流程）

**位置**：`muzero_collector.py` 第 1059-1105 行

**逻辑**：

```python
T = len(all_rewards)  # 总长度，如 300

# 构建 value 和 next_value
value = torch.tensor(all_values[0:T])        # [v0, v1, ..., v299]
next_value[t] = all_values[t+1] if t+1<T else 0  # [v1, v2, ..., v299, 0]

reward = torch.tensor(all_rewards)           # [r0, r1, ..., r299]
done = [False, ..., False, True]             # 最后一个是 True（episode 结束）

# 调用 ding 库的 GAE
compute_adv_data = gae_data(value, next_value, reward, done, None)
advantages = gae(compute_adv_data, gamma, gae_lambda)  # [300]

# Advantage 归一化
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# 计算 return
returns = value + advantages  # [300]
```

**输出**：
- `advantages`：长度 300
- `returns`：长度 300

---

### 第 3 步：分配回 Segments（存 unroll_steps + td_steps）

**位置**：`muzero_collector.py` 第 1107-1129 行

**原理**：
- 我们计算了总长度为 300 的 advantage 和 return
- 需要分配回各个 segment
- 存储时存 `num_unroll_steps + td_steps` 个（对应 padding 长度）
- 这样 Buffer sample 时不会超出范围

**具体流程**：

```
num_unroll_steps = 5
td_steps = 3
unroll_plus_td = 8

从 advantages[0:300] 中分配：

Seg0:
  原始长度：100
  存的数量：8
  取：advantages[0:8]

Seg1:
  原始长度：100
  存的数量：8
  取：advantages[100:108]

Seg2:
  原始长度：100
  存的数量：8
  取：advantages[200:208]
```

**改动**：

```python
unroll_plus_td = self.policy_config.num_unroll_steps + self.policy_config.td_steps

offset = 0
for i, (pool_idx, segment, priorities, done_flag) in enumerate(segments_info):
    original_seg_len = segment.game_segment_length  # 如 100
    
    # 取 unroll_plus_td 个（8 个）
    num_to_take = unroll_plus_td
    available = len(advantages_np) - offset
    num_to_take = min(num_to_take, available)
    
    segment.advantage_segment = advantages_np[offset:offset + num_to_take].copy()
    segment.return_segment = returns_np[offset:offset + num_to_take].copy()
    
    offset += original_seg_len  # 下一个 offset = 100
    
    self.game_segment_pool[pool_idx] = (segment, priorities, done_flag)
```

---

## 📊 数据流完整示例

### 示例条件

```
game_segment_length = 100
num_unroll_steps = 5
td_steps = 3
一个 episode 分成 3 个 segments
```

### 数据流

#### 1️⃣ 收集阶段（Collector 中的 segments_info）

```
segments_info = [
    (pool_idx=0, Segment0, ...),
    (pool_idx=1, Segment1, ...),
    (pool_idx=2, Segment2, ...)
]

Segment0.reward_segment 长度 = 100（原始）+ 7（padding）= 107
Segment0.root_value_segment 长度 = 100（原始）+ 8（padding）= 108
...
```

#### 2️⃣ 拼接阶段（第 1 步）

```
for segment in segments_info:
    values = segment.root_value_segment[:100]    # 只取原始部分
    rewards = segment.reward_segment[:100]       # 只取原始部分

all_values = [100项] + [100项] + [100项] = 300项
all_rewards = [100项] + [100项] + [100项] = 300项
```

#### 3️⃣ GAE 计算阶段（第 2 步）

```
value = tensor([300 values])
next_value = tensor([299 values from 1:300 + 1 zero])
reward = tensor([300 rewards])

advantages, returns = gae(...)  # 输出都是 300 项
```

#### 4️⃣ 分配阶段（第 3 步）

```
unroll_plus_td = 8

Segment0:
  offset = 0
  advantages_segment = advantages_np[0:8]
  return_segment = returns_np[0:8]
  offset += 100 → 100

Segment1:
  offset = 100
  advantages_segment = advantages_np[100:108]
  return_segment = returns_np[100:108]
  offset += 100 → 200

Segment2:
  offset = 200
  advantages_segment = advantages_np[200:208]
  return_segment = returns_np[200:208]
  offset += 100 → 300
```

---

## 🔗 与 Buffer 的对接

### Buffer sample 流程

```
Buffer 中的 advantage_segment 长度 = 8

Sample 时从位置 pos 取 num_unroll_steps=5 个：
  pos ∈ [0, game_segment_length)

pos=0:   advantage[0:5]     ✅ 有效值
pos=50:  advantage[50:55]   ✅ 有效值（pos+5=55 < 8）
pos=95:  advantage[95:100]  ❌ 超出范围，但...

等等，这有问题！pos 可能是 95，而 advantage_segment 只有 8 个！
```

### 解决方案

**方案 A**：约束 sample 位置（推荐）
- 在 `game_buffer.py` 第 177-178 行
- 确保 `pos ≤ game_segment_length - num_unroll_steps`
- 这样总能取到足够的有效值

**改动**：
```python
if pos_in_game_segment >= self._cfg.game_segment_length - self._cfg.num_unroll_steps:
    pos_in_game_segment = np.random.choice(
        max(1, self._cfg.game_segment_length - self._cfg.num_unroll_steps),
        1
    ).item()
```

---

## 📋 改动清单

| 文件 | 行数 | 改动 | 优先级 |
|-----|-----|-----|--------|
| `muzero_collector.py` | 1034-1057 | 拼接时只取原始部分 | 🔴 高 |
| `muzero_collector.py` | 1107-1129 | 分配时存 `unroll_plus_td` 个 | 🔴 高 |
| `game_buffer.py` | 177-178 | 约束 sample 位置 | 🟡 中 |

---

## ✅ 验证清单

- [ ] 拼接数据无重复
- [ ] GAE 计算维度正确
- [ ] advantage_segment 长度 = `num_unroll_steps + td_steps`
- [ ] Buffer sample 不会超出范围
- [ ] 训练时 mask=0 的位置被正确忽略

---

## 🔍 当前 `muzero_collector.py` 中 GAE 实现思路概述

下面是基于当前 `LightZero/lzero/worker/muzero_collector.py` 实际代码逻辑，总结出的 GAE 计算与写回流程，便于对照实现与上面的设计方案。

### 1️⃣ 按 episode 分组所有 segments

- 每个 `GameSegment` 在创建或 reset 时都会被赋予一个 `episode_id`，同一条完整轨迹（一个 episode）里的所有 segments 共用同一个 `episode_id`。
- 在 `collect` 结束、`collected_episode >= n_episode` 时，调用 `self._batch_compute_gae_for_pool()`。
- `_batch_compute_gae_for_pool` 遍历 `self.game_segment_pool`，根据 `segment.episode_id` 分组，得到：
  - `episode_groups = { episode_id: [(pool_idx, segment, priorities, done_flag), ...], ... }`
- 同一个 episode 内的 segments 会按 `pool_idx` 排序，保证时间顺序正确。

### 2️⃣ 拼接整条 episode 的 reward / value 序列（只取原始部分）

- 对于一个 episode 内的每个 segment：
  - 使用 `segment.game_segment_length` 作为该段真实交互的长度（不包含 `pad_over` 添加的后缀）。
  - 只取原始部分：
    - `values = segment.root_value_segment[:original_seg_len]`
    - `rewards = segment.reward_segment[:original_seg_len]`
- 将这些 `values` / `rewards` 依时间顺序依次 `extend` 到：
  - `all_values`：该 episode 完整的 value 序列
  - `all_rewards`：该 episode 完整的 reward 序列
- 这样可以避免跨 segment 的重复（padding 部分来自下一段的开头，只在下一段的原始部分中出现一次）。

### 3️⃣ 使用 ding 的 `gae_data` / `gae` 在 episode 级别一次性计算 GAE

- 令 `T = len(all_rewards)`，构造以下张量：
  - `value[t]`：长度为 `T`，来自 `all_values`，多余部分不足则补 0。
  - `next_value[t]`：长度为 `T`，取 `all_values[t+1]`，最后一个位置补 0（终止状态 bootstrap 为 0）。
  - `reward[t]`：长度为 `T`，等于 `all_rewards`。
  - `done[t]`：长度为 `T`，全 0，只有最后一个时间步 `done[-1] = True`，表示整个 episode 在末尾终止。
- 使用 ding 内置 GAE 接口：
  - `compute_adv_data = gae_data(value, next_value, reward, done, None)`
  - `advantages = gae(compute_adv_data, gamma=self.ppo_gamma, gae_lambda=self.ppo_gae_lambda)`
- 对当前 episode 的 `advantages` 做一次标准化：
  - `advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)`
- 计算对应的 return（用于 PPO 的 value target）：
  - `unnormalized_returns = value + advantages`

### 4️⃣ 将 advantages / returns 切回各个 segment，并做“前看窗口”式 padding

- 维护一个 `offset` 指针，表示这一 episode 中已分配给前面 segments 的步数。
- 对 episode 内的每个 `(pool_idx, segment, priorities, done_flag)`：
  - 取 `original_seg_len = segment.game_segment_length`。
  - 先为当前段分配与自身原始长度对应的优势和回报：
    - `segment.advantage_segment = advantages_np[offset : offset + original_seg_len]`
    - `segment.return_segment = returns_np[offset : offset + original_seg_len]`
  - 然后根据 `unroll_plus_td_steps = num_unroll_steps + td_steps`，在末尾额外拼接一小段“展望窗口”：
    - 从 `next_start = offset + original_seg_len` 开始，再向后取最多 `unroll_plus_td_steps` 个：
      - `segment.advantage_segment = concat(自身原始部分, advantages_np[next_start : next_start + take_from_next])`
      - `segment.return_segment = concat(自身原始部分, returns_np[next_start : next_start + take_from_next])`
    - 这里的额外部分来自后续时间步，与 `GameSegment` 对 obs / reward / value 做 `pad_over` 的逻辑保持一致，只是对象换成了 advantage / return。
  - 更新 `offset += original_seg_len`，并把更新后的 `segment` 写回 `self.game_segment_pool[pool_idx]`。

### 5️⃣ 与旧版 `muzero_collector.py` 的差异点

- 旧版 `LightZero-bak/lzero/worker/muzero_collector.py` 中：
  - 没有引入 `from ding.rl_utils import gae_data, gae`。
  - 没有 `_batch_compute_gae_for_pool` 函数，也不会在收集结束时统一计算 GAE。
  - `GameSegment` 中不包含 `advantage_segment` / `return_segment`。
- 当前版本在 `collect` 结束前增加了整套：
  1. **按 episode 聚合 segments**  
  2. **拼接整条 episode 的 reward / value 序列**  
  3. **调用 ding 的 GAE 实现计算 advantages**  
  4. **再按 segment 切分并配合 `unroll_plus_td_steps` 做 padding 写回**  
- 这样每个 `GameSegment` 在被放入 buffer 之后，就已经自带了 PPO 所需的 `advantage_segment` 和 `return_segment`，训练侧可以直接使用。
