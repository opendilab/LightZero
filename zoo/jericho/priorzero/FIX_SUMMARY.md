# PriorZero 修复总结

## 已修复的问题

### 1. EasyDict 整数键问题 ✅

**错误**:
```
TypeError: attribute name must be string, not 'int'
```

**原因**: `action_inv_map` 使用整数键 (例如 `{0: "go north", 1: "go south"}`), 但 EasyDict 要求字符串键。

**修复** ([priorzero_config.py:425-428](priorzero_config.py#L425-L428)):
```python
# 使用 object.__setattr__ 绕过 EasyDict 的自动转换
object.__setattr__(main_config.policy, 'action_map', _temp_action_map)
object.__setattr__(main_config.policy, 'action_inv_map', _temp_action_inv_map)
```

### 2. MockConfig 缺少属性 ✅

**错误**:
```
AttributeError: 'MockConfig' object has no attribute 'discount_factor'
AttributeError: 'MockConfig' object has no attribute 'gray_scale'
AttributeError: 'obj' object has no attribute 'action_space_size'
```

**原因**: GameSegment 父类需要完整的配置属性。

**修复** ([test_components.py:76-91](test_components.py#L76-L91), [game_segment_priorzero.py:357-373](game_segment_priorzero.py#L357-L373)):
```python
class MockConfig:
    def __init__(self):
        self.num_unroll_steps = 10
        self.td_steps = 5
        self.discount_factor = 0.99
        self.gray_scale = False
        self.transform2string = False
        self.sampled_algo = False
        self.gumbel_algo = False
        self.use_ture_chance_label_in_chance_encoder = False
        self.model = type('obj', (object,), {
            'frame_stack_num': 4,
            'action_space_size': 10,
            'observation_shape': (4, 84, 84),
            'image_channel': 4
        })()
```

### 3. GameSegment.append() 参数问题 ✅

**错误**:
```
TypeError: GameSegment.append() got an unexpected keyword argument 'raw_obs_text'
```

**原因**: 父类 GameSegment 不接受 PriorZero 特定的参数。

**修复** ([game_segment_priorzero.py:96-98](game_segment_priorzero.py#L96-L98)):
```python
# 在传递给父类之前提取 PriorZero 特定的 kwargs
raw_obs_text = kwargs.pop('raw_obs_text', None)
llm_prior_text = kwargs.pop('llm_prior_text', None)

# 使用剩余的 kwargs 调用父类
super().append(action, obs, reward, action_mask, to_play, **kwargs)
```

### 4. 错误的 Evaluator 导入路径 ✅

**错误**:
```
ModuleNotFoundError: No module named 'lzero.worker.evaluator'
```

**原因**: 导入路径不正确。

**修复** ([priorzero_evaluator.py:16](priorzero_evaluator.py#L16)):
```python
# 修改前
from lzero.worker.evaluator import MuZeroEvaluator as OriginalEvaluator

# 修改后
from lzero.worker.muzero_evaluator import MuZeroEvaluator as OriginalEvaluator
```

### 5. NumPy 版本冲突

**错误**:
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility
di-engine 0.5.3 requires numpy<2,>=1.18.0, but you have numpy 2.2.6
```

**解决方案**: 运行提供的修复脚本
```bash
bash fix_environment.sh
```

此脚本将：
- 降级 numpy 到 1.26.4
- 重新安装兼容的依赖包

---

## 测试结果

### ✅ 全部测试通过！

```bash
# 组件测试
python test_components.py
# 结果: 3/3 PASSED (Configuration, Game Segment, Policy Helpers)

# 配置生成测试
python priorzero_config.py
# 结果: ✓ All configurations generated successfully!

# GameSegment 测试
python game_segment_priorzero.py
# 结果: ✓ All tests passed!
```

---

## 关键技术要点

### 1. EasyDict 限制
- EasyDict 会递归地转换所有字典，包括整数键
- 解决方法：使用 `object.__setattr__()` 绕过 EasyDict 的 `__setattr__`
- 或者：在 EasyDict 转换前移除有问题的字段，转换后再添加

### 2. 继承与 kwargs 处理
- 子类添加新参数时，必须在调用 `super()` 前使用 `pop()` 提取
- 避免将未知参数传递给父类方法

### 3. MockConfig 完整性
- 测试用的 Mock 对象必须包含所有父类需要的属性
- 特别注意嵌套对象（如 `config.model.*`）的属性

---

## 下一步

现在所有组件测试都已通过，可以进行：

1. **环境修复** (如果还没做):
   ```bash
   bash fix_environment.sh
   ```

2. **完整流程测试** (需要 GPU):
   ```bash
   python priorzero_entry.py --quick_test --env_id zork1.z5 --seed 0
   ```

3. **正式实验运行**:
   ```bash
   python priorzero_entry.py --env_id zork1.z5 --seed 0 --max_iter 100000
   ```

---

## 文件修改总览

| 文件 | 修改内容 | 行数 |
|------|---------|------|
| [priorzero_config.py](priorzero_config.py) | 修复 EasyDict 整数键问题 | ~530 |
| [priorzero_evaluator.py](priorzero_evaluator.py) | 修复导入路径 | 56 |
| [game_segment_priorzero.py](game_segment_priorzero.py) | 修复 kwargs 处理和 MockConfig | ~445 |
| [test_components.py](test_components.py) | 修复 MockConfig | ~234 |

---

**修复完成日期**: 2025-10-20
**测试状态**: ✅ 全部通过
