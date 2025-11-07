# ORZ Evaluator - 独立评估模块

用于评估已训练的大型语言模型 (LLM) 在数学和推理任务上的性能。该模块支持多种数据集和灵活的配置选项。

## 📋 功能特性

- ✅ 支持多个评估数据集（Math500、AIME2024、GPQA Diamond、Jericho）
- ✅ 集成 vLLM 进行高效推理
- ✅ 支持分布式推理（多 GPU）
- ✅ 灵活的参数配置，支持简洁和详细用法
- ✅ 自动答案提取和正确性判断
- ✅ 详细的评估结果保存（JSONL 格式）
- ✅ 支持传入已加载的模型或预加载模型对象

## 🚀 快速开始

### 安装依赖

```bash
pip install transformers vllm ray loguru
```

### 基本用法

#### 方式 1：简洁用法（推荐）

```python
from eval_orz import Evaluator
import asyncio

# 只需传入模型路径和数据集路径
evaluator = Evaluator(
    model_path="path/to/model/checkpoint",
    eval_prompt_data=[
        "data/eval_data/math500.json",
        "data/eval_data/aime2024.json",
        "data/eval_data/gpqa_diamond.json",
    ]
)

# 运行评估
results = asyncio.run(evaluator.eval())
print(f"评估结果: {results}")

# 清理资源
evaluator.cleanup()
```

#### 方式 2：覆盖默认参数

```python
# 传入必要参数，同时覆盖部分配置
evaluator = Evaluator(
    model_path="path/to/model",
    eval_prompt_data=["data/eval_data/math500.json"],
    temperature=0.8,  # 覆盖默认值 1.0
    gpu_memory_utilization=0.5,  # 覆盖默认值 0.3
    vllm_num_engines=2,  # 多 GPU 时可增加
)

results = asyncio.run(evaluator.eval())
```

#### 方式 3：完整的 Config 对象（高级）

```python
from eval_orz import EvaluatorConfig, Evaluator

config = EvaluatorConfig(
    model_path="path/to/model",
    tokenizer_path="path/to/model",  # 可选，默认使用 model_path
    vllm_num_engines=1,
    vllm_tensor_parallel_size=1,
    enable_prefix_caching=True,
    gpu_memory_utilization=0.3,
    max_model_len=8192,
    temperature=1.0,
    top_p=1.0,
    top_k=-1,
    generate_max_len=8000,
    stop=["User:", "Human:", "Assistant:", "</answer>"],
    eval_prompt_data=[
        "data/eval_data/eval_jericho_dataset_his10_4games_1.8k_20251013_instruct.json",
        "data/eval_data/math500.json",
        "data/eval_data/aime2024.json",
        "data/eval_data/gpqa_diamond.json",
    ],
    prompt_max_len=2048,
    output_dir="eval_results",
    save_detailed_results=True,
)

evaluator = Evaluator(config)
results = asyncio.run(evaluator.eval())
```

#### 方式 4：传入已加载的模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 预加载模型和 tokenizer
model = AutoModelForCausalLM.from_pretrained("path/to/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/model")

# 传入预加载的对象
evaluator = Evaluator(
    model=model,
    tokenizer=tokenizer,
    eval_prompt_data=["data/eval_data/math500.json"]
)

results = asyncio.run(evaluator.eval())
```

## 📊 评估数据集

支持以下数据集（JSON 格式）：

| 数据集 | 文件名 | 描述 |
|-------|--------|------|
| Math500 | `math500.json` | 500 个数学问题 |
| AIME2024 | `aime2024.json` | 2024 年 AIME 竞赛题 |
| GPQA Diamond | `gpqa_diamond.json` | 高难度通用知识题 |
| Jericho | `eval_jericho_dataset_his10_4games_1.8k_20251013_instruct.json` | 文本冒险游戏数据 |

数据集格式示例：
```json
[
  {
    "prompt": [{"value": "问题内容"}],
    "final_answer": "期望答案",
    "file_name": "source_dataset"
  }
]
```

## ⚙️ 参数说明

### 模型和分词器配置

- `model_path` (str): 模型路径或 HuggingFace 模型名称
- `tokenizer_path` (str, 可选): 分词器路径，默认使用 `model_path`

### vLLM 推理配置

- `vllm_num_engines` (int): vLLM 引擎数量，默认 1（多 GPU 可增加）
  - **注意**：单 GPU 环境保持为 1；多节点训练可增加
- `vllm_tensor_parallel_size` (int): 张量并行大小，默认 1
- `enable_prefix_caching` (bool): 启用前缀缓存，默认 True
- `gpu_memory_utilization` (float): GPU 内存使用比例，范围 [0.0-1.0]，默认 0.3
- `max_model_len` (int): 最大模型长度，默认 8192

### 生成配置

- `temperature` (float): 采样温度，默认 1.0
- `top_p` (float): nucleus 采样参数，默认 1.0
- `top_k` (int): top-k 采样参数，默认 -1（禁用）
- `generate_max_len` (int): 生成的最大长度，默认 8000
- `stop` (List[str]): 停止词列表

### 数据和输出配置

- `eval_prompt_data` (List[str]): 评估数据集路径列表
- `prompt_max_len` (int): 提示词最大长度，默认 2048
- `output_dir` (str): 输出结果目录，默认 "eval_results"
- `save_detailed_results` (bool): 是否保存详细结果，默认 True

## 📈 输出结果

### 控制台输出
```
Evaluation completed: math500/accuracy: 0.7500, aime2024/accuracy: 0.5200, gpqa_diamond/accuracy: 0.4800, eval_accuracy: 0.5833
```

### 文件输出 (eval_results/*.jsonl)
```json
{
  "prompt": "完整的提示词文本",
  "output": "模型完整生成内容",
  "final_answer": "\\boxed{答案}",
  "answer": "期望答案",
  "iscorrect": true
}
```

## 🔧 对齐说明

本模块参数已对齐到 `Open-Reasoner-Zero/playground/orz_7b_ppo_jericho_1013.py`：

- ✅ vLLM 配置：完全对应
- ✅ 生成参数：完全对应
- ✅ 数据集：包含 Jericho 评估数据
- ⚠️ `vllm_num_engines`：
  - 参考值为 8（多节点环境）
  - 单 GPU 环境改为 1
  - 多 GPU 可根据需要增加

## 📝 完整示例脚本

```python
import asyncio
from eval_orz import Evaluator

async def main():
    # 创建评估器
    evaluator = Evaluator(
        model_path="checkpoints/orz_0p5b_ppo_jericho_1012_1gpu/iter12/policy",
        eval_prompt_data=[
            "data/eval_data/eval_jericho_dataset_his10_4games_1.8k_20251013_instruct.json",
            "data/eval_data/math500.json",
            "data/eval_data/aime2024.json",
            "data/eval_data/gpqa_diamond.json",
        ]
    )

    try:
        # 运行评估
        results = await evaluator.eval()

        # 处理结果
        print("=" * 50)
        print("评估结果汇总:")
        for dataset, accuracy in results.items():
            if "accuracy" in dataset:
                print(f"  {dataset}: {accuracy:.4f}")

    finally:
        # 清理资源
        evaluator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

## 🐛 常见问题

### Q: 如何在多 GPU 上加速评估？

A: 增加 `vllm_num_engines` 参数：
```python
evaluator = Evaluator(
    model_path="...",
    vllm_num_engines=2,  # 使用 2 个 vLLM 引擎
)
```

### Q: 如何调整 GPU 内存使用？

A: 修改 `gpu_memory_utilization` 参数：
```python
evaluator = Evaluator(
    model_path="...",
    gpu_memory_utilization=0.5,  # 使用 50% GPU 内存
)
```

### Q: 答案提取不正确？

A: 检查数据集格式，确保答案用 `\boxed{}` 标记：
```json
{
  "final_answer": "\\boxed{42}",
  ...
}
```

## 📚 相关文件

- `eval_orz.py` - 主评估模块
- `dataset/eval_dataset.py` - 评估数据集处理
- `orz/ppo/tools/math_utils.py` - 数学答案验证工具

## 📄 许可证

MIT License
