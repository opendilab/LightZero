# PriorZero 快速启动指南

## 🚨 已知问题修复

### 问题 1: numpy 版本冲突

**错误**:
```
di-engine 0.5.3 requires numpy<2,>=1.18.0, but you have numpy 2.2.6
```

**解决方法**:
```bash
# 方法 1: 使用提供的修复脚本
bash fix_environment.sh

# 方法 2: 手动降级 numpy
pip install "numpy<2,>=1.24.1" --force-reinstall
```

### 问题 2: mujoco 导致的 import 错误

**错误**:
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```

**解决方法**:
这是由于 numpy 2.x 与 mujoco-py 不兼容。使用上述 numpy 降级方法即可解决。

### 问题 3: EasyDict 不支持整数键

**错误**:
```
TypeError: attribute name must be string, not 'int'
```

**解决方法**:
已修复！action_map 现在直接设置为属性，不经过 EasyDict 转换。

---

## 📋 测试步骤（按顺序）

### Step 1: 修复环境依赖

```bash
cd /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/jericho/priorzero

# 运行环境修复脚本
bash fix_environment.sh
```

### Step 2: 测试独立组件（推荐先做）

这个测试不需要完整的环境设置（避免 mujoco 等问题）：

```bash
# 运行轻量级组件测试
python test_components.py
```

预期输出：
```
================================================================================
PriorZero Component Tests
================================================================================

TEST 1: Configuration Generation
✅ Configuration test PASSED

TEST 2: Game Segment
✅ Game segment test PASSED

TEST 3: Policy Helper Functions
✅ Policy helpers test PASSED

================================================================================
TEST SUMMARY
================================================================================
Configuration        : ✅ PASSED
Game Segment        : ✅ PASSED
Policy Helpers      : ✅ PASSED

🎉 ALL TESTS PASSED!
```

### Step 3: 测试完整配置

```bash
# 测试配置生成
python priorzero_config.py
```

预期输出：
```
================================================================================
Testing PriorZero Configuration Generation
================================================================================

1. Standard PriorZero Config:
  Exp name: data_priorzero/priorzero_zork1.z5_seed0
  Action space size: 20
  LLM model: Qwen/Qwen2.5-0.5B-Instruct
  World model layers: 4
  Num action mappings: 23

✓ All configurations generated successfully!
```

### Step 4: 测试 GameSegment

```bash
python game_segment_priorzero.py
```

### Step 5: 完整训练测试（需要 GPU）

**⚠️ 注意**: 这一步需要：
- GPU（至少 12GB 显存）
- Jericho 环境安装
- vLLM 正常工作

```bash
# 快速测试（减少资源需求）
python priorzero_entry.py --quick_test --env_id zork1.z5 --seed 0
```

---

## 🔧 常见问题排查

### 问题: ImportError: No module named 'vllm'

**解决**:
```bash
pip install vllm==0.11.0
```

### 问题: CUDA out of memory

**解决方法 1**: 降低 GPU 内存使用
```python
# 在 priorzero_config.py 中修改：
gpu_memory_utilization=0.2  # 从 0.3 降到 0.2
```

**解决方法 2**: 使用更小的 LLM
```python
# 在 priorzero_config.py 中修改：
llm_model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # 最小的模型
```

**解决方法 3**: 启用 LoRA
```python
use_lora=True
lora_r=8
```

**解决方法 4**: 减少 batch size
```python
batch_size=8  # 从 32 降到 8
```

### 问题: Jericho 环境找不到游戏文件

**解决**:
```bash
# 下载 Jericho 游戏文件
git clone https://github.com/microsoft/z-machine-games-master.git

# 或修改 config 中的路径
game_path="./your-path-to/jericho-game-suite/zork1.z5"
```

---

## 🎯 推荐的开发工作流

### 1. 开发新功能

```bash
# 1. 测试组件
python test_components.py

# 2. 测试配置
python priorzero_config.py

# 3. 测试完整 pipeline（如果有 GPU）
python priorzero_entry.py --quick_test --no_save
```

### 2. 调试问题

使用 `test_components.py` 进行单独测试：
- 只测试配置: 修改 `main()` 只调用 `test_config()`
- 只测试某个函数: 直接运行该函数

### 3. 实验运行

```bash
# 小规模实验（快速验证）
python priorzero_entry.py --quick_test --seed 0

# 完整实验
python priorzero_entry.py --env_id zork1.z5 --seed 0 --max_iter 100000

# 使用纯 UniZero（消融实验）
# 修改 priorzero_config.py 使用 get_config_pure_unizero()
```

---

## 📊 监控训练

### TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir=./data_priorzero/ --port=6006

# 在浏览器打开
# http://localhost:6006
```

### 关键指标

- `train/total_loss`: 总损失
- `train/wm_total_loss`: World model 损失
- `train/llm_sft_loss`: LLM 监督微调损失
- `train/llm_rft_loss`: LLM 强化微调损失
- `evals/reward_mean`: 平均评估奖励

---

## 🐛 Debug 模式

如果遇到奇怪的错误，启用详细日志：

```python
# 在 priorzero_entry.py 开头添加
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## ✅ 验证清单

在提交代码或运行实验前，确保：

- [ ] `python test_components.py` 全部通过
- [ ] `python priorzero_config.py` 成功运行
- [ ] `python game_segment_priorzero.py` 成功运行
- [ ] numpy 版本正确 (`<2.0`)
- [ ] GPU 显存充足（至少 12GB）
- [ ] vLLM 正常工作

---

## 📞 获取帮助

如果以上步骤都无法解决问题：

1. 检查 numpy 版本: `python -c "import numpy; print(numpy.__version__)"`
2. 检查 torch 版本: `python -c "import torch; print(torch.__version__)"`
3. 检查 vllm 版本: `python -c "import vllm; print(vllm.__version__)"`
4. 查看完整错误堆栈
5. 在 GitHub issue 中报告问题

---

**祝实验顺利！🚀**
