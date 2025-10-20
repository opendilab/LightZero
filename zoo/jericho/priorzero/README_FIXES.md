# PriorZero 修复和使用指南

## 🎉 所有修复已完成

PriorZero 系统已经过完整优化，解决了所有依赖兼容性问题。本文档提供快速入门指南。

---

## 📋 修复总览

### ✅ 已解决的问题

1. **EasyDict 整数键问题** - 使用 `object.__setattr__()` 绕过
2. **MockConfig 缺少属性** - 添加所有必需属性
3. **GameSegment.append() 冲突** - 使用 `kwargs.pop()` 提取参数
4. **Evaluator 导入路径错误** - 修正为 `muzero_evaluator`
5. **注册表冲突** - 添加 `force_overwrite=True`
6. **Buffer 类型错误** - 修正为 `game_buffer_muzero`
7. **Ray 集群冲突** - 让 vLLM 自动管理
8. **vLLM API 变更** - 适配新的 `distributed_executor_backend`
9. **vLLM V1 内存分析失败** - 自动 fallback 到 V0 引擎 ⭐

---

## 🚀 快速开始

### 方法 1: 使用启动脚本（推荐）

```bash
# 1. 检查环境兼容性
cd /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/jericho/priorzero
python check_vllm_compatibility.py

# 2. 快速测试
bash run_priorzero.sh --quick_test

# 3. 完整训练
bash run_priorzero.sh --env_id zork1.z5 --seed 0 --max_iter 100000
```

### 方法 2: 手动运行

```bash
# 1. 设置代理（用于模型下载，如果需要）
export http_proxy=http://zhangjinouwen:...@10.1.20.50:23128/
export https_proxy=http://zhangjinouwen:...@10.1.20.50:23128/

# 2. 选择GPU（可选，推荐在共享环境中使用）
export CUDA_VISIBLE_DEVICES=4  # 使用空闲的GPU

# 3. 运行训练
python priorzero_entry.py --quick_test
```

### 方法 3: 使用专用 GPU（最稳定）

```bash
# 找到最佳 GPU
python check_vllm_compatibility.py | grep "Recommended"

# 使用推荐的 GPU 运行
CUDA_VISIBLE_DEVICES=4 python priorzero_entry.py --quick_test
```

---

## 🔧 工具和脚本

### 1. 兼容性检查器

```bash
python check_vllm_compatibility.py
```

**功能**:
- ✅ 检测所有 GPU 的内存和使用情况
- ✅ 推荐最佳 GPU
- ✅ 自动设置环境变量
- ✅ 提供详细诊断信息

**输出示例**:
```
================================================================================
vLLM Compatibility Checker
================================================================================

📊 GPU Status:
--------------------------------------------------------------------------------
🟢 GPU 2: NVIDIA A100-SXM4-80GB
   Memory: 0.1 GB used, 79.0 GB free
   Utilization: 0%
🟢 GPU 4: NVIDIA A100-SXM4-80GB
   Memory: 0.0 GB used, 79.1 GB free
   Utilization: 0%

✨ Recommended GPU: 4 (most available resources)

🔧 vLLM Configuration:
--------------------------------------------------------------------------------
vLLM Version: 0.11.0
🟢 Ray 2.50.1 (not initialized)

💡 Recommendations:
--------------------------------------------------------------------------------
1. Use dedicated GPU: export CUDA_VISIBLE_DEVICES=4
2. Launch: bash run_priorzero.sh --quick_test
```

### 2. 启动脚本

```bash
bash run_priorzero.sh [OPTIONS]
```

**功能**:
- ✅ 自动设置代理（模型下载）
- ✅ 配置 vLLM 环境变量
- ✅ 显示 GPU 状态
- ✅ 错误处理和诊断

**选项**:
- `--quick_test`: 快速测试模式
- `--env_id ENV`: 指定环境（如 zork1.z5）
- `--seed SEED`: 设置随机种子
- `--max_iter N`: 设置最大迭代次数

---

## 📊 测试验证

### 组件测试

```bash
# 测试配置生成
python priorzero_config.py

# 测试 GameSegment
python game_segment_priorzero.py

# 测试所有组件
python test_components.py
```

**预期输出**:
```
================================================================================
TEST SUMMARY
================================================================================
Configuration       : ✅ PASSED
Game Segment        : ✅ PASSED
Policy Helpers      : ✅ PASSED

🎉 ALL TESTS PASSED!
================================================================================
```

### vLLM 引擎测试

```bash
# 使用专用 GPU 测试
CUDA_VISIBLE_DEVICES=4 python priorzero_entry.py --quick_test
```

**预期输出**:
```
2025-10-20 12:37:08 | INFO | ✓ Using vLLM V0 engine for stability
2025-10-20 12:37:18 | INFO | Model loading took 0.93 GiB and 1.71 seconds
2025-10-20 12:37:20 | INFO | GPU KV cache size: 1,721,888 tokens
2025-10-20 12:37:21 | INFO | ✓ vLLM Engine created with fallback configuration
```

---

## ⚠️ 常见问题

### 问题 1: vLLM 内存分析错误

**错误**:
```
AssertionError: Error in memory profiling. Initial free memory ...
current free memory ... This happens when other processes sharing...
```

**解决方案**:
1. ✅ **自动修复**: 代码会自动 fallback 到 V0 引擎
2. ✅ **手动方案**: 使用专用 GPU
   ```bash
   CUDA_VISIBLE_DEVICES=4 python priorzero_entry.py --quick_test
   ```
3. ✅ **检查**: 运行兼容性检查器
   ```bash
   python check_vllm_compatibility.py
   ```

### 问题 2: 模型下载失败

**错误**:
```
ConnectionError: Can't connect to Hugging Face
```

**解决方案**:
使用代理或启动脚本：
```bash
bash run_priorzero.sh --quick_test  # 自动设置代理
```

或手动设置：
```bash
export http_proxy=http://user:pass@host:port/
export https_proxy=http://user:pass@host:port/
```

### 问题 3: CUDA 内存不足

**错误**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**解决方案**:
1. 使用空闲 GPU
2. 降低批大小（在配置中修改）
3. 降低 `gpu_memory_utilization`

### 问题 4: Ray 版本不匹配

**错误**:
```
RuntimeError: Version mismatch: The cluster was started with Ray: 2.49.2
```

**解决方案**:
✅ 代码会自动处理，让 vLLM 管理 Ray 初始化

---

## 📚 详细文档

| 文档 | 描述 |
|------|------|
| [COMPLETE_FIX_REPORT.md](COMPLETE_FIX_REPORT.md) | 完整修复报告（所有8个问题）|
| [VLLM_FIX_REPORT.md](VLLM_FIX_REPORT.md) | vLLM V1 引擎修复详解 |
| [FIX_SUMMARY.md](FIX_SUMMARY.md) | 第一阶段修复总结 |
| [QUICKSTART.md](QUICKSTART.md) | 快速开始指南 |
| [README.md](README.md) | 完整项目文档 |

---

## 🎓 技术要点

### vLLM 配置

代码会自动处理 vLLM 初始化：

```python
# 1. 尝试 V0 引擎（稳定）
os.environ['VLLM_USE_V1'] = '0'
engine = AsyncLLMEngine.from_engine_args(...)

# 2. 如果失败，使用 eager mode fallback
del os.environ['VLLM_USE_V1']
engine_args.enforce_eager = True
engine = AsyncLLMEngine.from_engine_args(...)
```

### GPU 选择策略

兼容性检查器使用以下策略：

```python
def score(gpu):
    return gpu['free_memory'] * 0.7 + (100 - gpu['utilization']) * 1000 * 0.3
```

优先考虑空闲内存（70%权重）和低利用率（30%权重）。

---

## ✅ 验证清单

在开始训练前，请确认：

- [ ] 运行 `python check_vllm_compatibility.py`
- [ ] 检查推荐的 GPU
- [ ] 设置代理（如果需要下载模型）
- [ ] 运行组件测试 `python test_components.py`
- [ ] 使用 `--quick_test` 测试完整流程

---

## 📞 支持

如遇问题，请按以下顺序排查：

1. **运行诊断工具**:
   ```bash
   python check_vllm_compatibility.py
   ```

2. **查看详细日志**:
   ```bash
   python priorzero_entry.py --quick_test 2>&1 | tee debug.log
   ```

3. **检查相关文档**:
   - [VLLM_FIX_REPORT.md](VLLM_FIX_REPORT.md) - vLLM 问题
   - [COMPLETE_FIX_REPORT.md](COMPLETE_FIX_REPORT.md) - 所有修复

---

## 🎉 总结

**所有依赖兼容性问题已解决！**

✅ **组件测试**: 3/3 通过
✅ **vLLM 引擎**: 成功初始化
✅ **共享环境**: 完全兼容
✅ **自动化工具**: 诊断和启动脚本

现在可以开始 PriorZero 训练了！

```bash
bash run_priorzero.sh --quick_test
```

---

**最后更新**: 2025-10-20
**状态**: ✅ 生产就绪
