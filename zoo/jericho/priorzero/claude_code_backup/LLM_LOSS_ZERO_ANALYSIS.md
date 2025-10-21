# LLM 损失为零问题深度分析

**日期**: 2025-10-21
**状态**: ✅ 已识别根本原因

---

## 问题现象

TensorBoard 日志显示:
```
llm_sft_loss_avg: 0.0
llm_rft_loss_avg: 0.0
llm_total_loss_avg: 0.0
```

所有 LLM 相关的损失都是 0,说明 LLM 训练完全没有执行。

---

## 根本原因分析

### 原因 #1: `game_segments` 数据格式不匹配 ⚠️

**问题定位:**

1. **Buffer 返回格式** (`game_buffer_priorzero.py:122-123`):
   ```python
   train_data = [current_batch, target_batch, game_segments]  # 3 个元素
   return train_data
   ```

2. **Policy 期望格式** (`priorzero_policy.py:405-409`):
   ```python
   if len(data) == 4:
       current_batch, target_batch, train_iter, game_segments = data  # 4 个元素!
   elif len(data) == 3:
       current_batch, target_batch, train_iter = data
       game_segments = None  # ❌ 这里 game_segments 被设为 None!
   ```

**问题:**
- Buffer 返回 3 个元素: `[current_batch, target_batch, game_segments]`
- Policy 解包时认为第 3 个元素是 `train_iter`,把 `game_segments` 当作 `train_iter`
- 然后在 `len(data) == 3` 分支设置 `game_segments = None`
- 导致 line 590 的条件 `if game_segments is not None:` 为 False
- **所有 LLM 训练代码被跳过!**

### 原因 #2: `train_iter` 未从正确位置传入

查看 learner 调用 policy 的代码,`train_iter` 应该由 learner 传递,但 buffer 并不负责提供它。

**正确的数据流应该是:**
```
Buffer.sample() → [current_batch, target_batch, game_segments]
Learner.train() → 添加 train_iter → [current_batch, target_batch, train_iter, game_segments]
Policy._forward_learn() → 解包 4 个元素
```

但当前实现中,learner 可能没有正确添加 `train_iter`。

---

## 详细代码追踪

### 1. Buffer 采样 (✅ 正确)

**文件**: `lzero/mcts/buffer/game_buffer_priorzero.py:186-223`

```python
def sample(self, batch_size: int, policy) -> List[Any]:
    """Sample data with game_segments (optimized version)."""
    # ... 省略 ...

    target_batch = [batch_rewards, batch_target_values, batch_target_policies]

    return [current_batch, target_batch, game_segments]  # ✅ 返回 3 个元素
```

### 2. Policy 解包 (❌ 格式不匹配)

**文件**: `zoo/jericho/priorzero/priorzero_policy.py:405-417`

```python
def _forward_learn(self, data: Tuple[torch.Tensor]) -> Dict[str, Union[float, int]]:
    # Unpack data
    if len(data) == 4:
        current_batch, target_batch, train_iter, game_segments = data  # 期望 4 个
    elif len(data) == 3:
        current_batch, target_batch, train_iter = data  # ❌ 实际是这个分支
        game_segments = None  # ❌ game_segments 被设为 None!
```

### 3. LLM 训练检查 (❌ 被跳过)

**文件**: `zoo/jericho/priorzero/priorzero_policy.py:590`

```python
if game_segments is not None:  # ❌ 永远是 False
    # Collect training data from game segments
    sft_prompts = []
    # ... LLM 训练代码 ...
    # ❌ 这里的代码永远不会执行!
```

### 4. LLM 损失初始化 (✅ 但永远不更新)

**文件**: `zoo/jericho/priorzero/priorzero_policy.py:581-582`

```python
llm_sft_loss = torch.tensor(0.0, device=self._cfg.device)  # 初始化为 0
llm_rft_loss = torch.tensor(0.0, device=self._cfg.device)  # 初始化为 0

# ... game_segments is None,所以下面的代码不执行 ...

# 损失保持为 0!
```

---

## 解决方案

### 方案 A: 修改 Policy 解包逻辑 (推荐)

**文件**: `zoo/jericho/priorzero/priorzero_policy.py:405-417`

```python
# 修改前:
if len(data) == 4:
    current_batch, target_batch, train_iter, game_segments = data
elif len(data) == 3:
    current_batch, target_batch, train_iter = data  # ❌ 错误
    game_segments = None

# 修改后:
if len(data) == 4:
    # Learner 传入完整的 4 个元素 (理想情况)
    current_batch, target_batch, train_iter, game_segments = data
elif len(data) == 3:
    # Buffer 直接返回的 3 个元素
    current_batch, target_batch, game_segments = data  # ✅ 正确解包
    train_iter = None  # train_iter 可能不需要,或从 self._train_iteration 获取
```

**优点:**
- 最小改动
- 立即修复问题

**缺点:**
- 需要确认 `train_iter` 是否真的需要

### 方案 B: 修改 Learner 以添加 `train_iter`

**文件**: 需要查找 `BaseLearner` 或相关 learner 代码

在 learner 的 `train()` 方法中:

```python
# 修改前:
train_data = replay_buffer.sample(batch_size, policy)
# train_data = [current_batch, target_batch, game_segments]

log_dict = policy.forward(train_data)

# 修改后:
train_data = replay_buffer.sample(batch_size, policy)
# 添加 train_iter
train_data.append(self.train_iter)  # 或 policy._train_iteration
# 现在 train_data = [current_batch, target_batch, game_segments, train_iter]

log_dict = policy.forward(train_data)
```

**优点:**
- 保持 policy 的 4 元素解包逻辑
- 更清晰的职责分离

**缺点:**
- 需要修改 learner 代码 (可能影响其他地方)

### 方案 C: 完全移除 `train_iter` 依赖

检查 `train_iter` 在 `_forward_learn` 中是否真的被使用。如果没有使用,可以完全移除它:

```python
# 简化为统一的 3 元素格式
current_batch, target_batch, game_segments = data

# 如果需要 train_iter,从 self._train_iteration 获取
train_iter = self._train_iteration
```

---

## 验证步骤

### 1. 添加调试日志

在 `priorzero_policy.py:405` 之后添加:

```python
import logging
logger = logging.getLogger(__name__)
logger.info(f"[DEBUG_LLM_LOSS] data length: {len(data)}")
logger.info(f"[DEBUG_LLM_LOSS] game_segments type: {type(game_segments)}")
logger.info(f"[DEBUG_LLM_LOSS] game_segments is None: {game_segments is None}")
if game_segments is not None:
    logger.info(f"[DEBUG_LLM_LOSS] game_segments count: {len(game_segments)}")
```

### 2. 检查日志输出

**修复前** (预期):
```
[DEBUG_LLM_LOSS] data length: 3
[DEBUG_LLM_LOSS] game_segments type: <class 'NoneType'>
[DEBUG_LLM_LOSS] game_segments is None: True
```

**修复后** (预期):
```
[DEBUG_LLM_LOSS] data length: 3
[DEBUG_LLM_LOSS] game_segments type: <class 'list'>
[DEBUG_LLM_LOSS] game_segments is None: False
[DEBUG_LLM_LOSS] game_segments count: 32  (batch_size)
```

### 3. 监控 TensorBoard

修复后应该看到:
```
llm_sft_loss_avg: > 0.0  (如果有 MCTS 策略数据)
llm_rft_loss_avg: > 0.0  (如果启用 RFT)
num_sft_samples: > 0
num_rft_samples: > 0
```

---

## 立即行动

### 优先级 1: 快速修复 (推荐方案 A)

```python
# 文件: zoo/jericho/priorzero/priorzero_policy.py:405-417
# 修改解包逻辑:

if len(data) == 4:
    current_batch, target_batch, train_iter, game_segments = data
elif len(data) == 3:
    current_batch, target_batch, game_segments = data  # ✅ 修复
    train_iter = self._train_iteration  # 从实例变量获取
else:
    raise ValueError(f"Unexpected data format: expected 3 or 4 elements, got {len(data)}")
```

### 优先级 2: 验证修复

1. 运行训练
2. 检查日志中的 `[DEBUG_LLM_LOSS]` 输出
3. 查看 TensorBoard 确认 LLM 损失 > 0

### 优先级 3: 检查 `train_iter` 使用

搜索 `_forward_learn` 中是否有使用 `train_iter`:
```bash
grep -n "train_iter" zoo/jericho/priorzero/priorzero_policy.py
```

如果没有使用,可以完全移除对它的依赖。

---

## 总结

### 根本原因
数据格式不匹配:
- Buffer 返回 3 个元素
- Policy 期望 4 个元素
- 导致 `game_segments` 被误解为 `train_iter`
- LLM 训练代码被完全跳过

### 快速修复
修改 Policy 的解包逻辑,正确处理 3 元素格式

### 预期效果
- ✅ `game_segments` 正确传递给 LLM 训练
- ✅ SFT/RFT 损失开始计算
- ✅ LLM 策略开始训练

---

**下一步**: 应用修复并迁移 ORZ 的多卡训练和 RFT 代码
