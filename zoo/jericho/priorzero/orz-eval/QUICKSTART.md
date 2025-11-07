# ORZ Evaluator 快速开始指南

## 项目位置

```
/mnt/shared-storage-user/tangjia/orz/orz-eval
```

## 快速开始（3分钟）

### 1. 设置环境

```bash
cd /mnt/shared-storage-user/tangjia/orz/orz-eval

# 创建虚拟环境（可选但推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据和模型已准备好！

✓ 评估数据集已在 `data/eval_data/` 中：
  - `math500.json` (数学问题)
  - `aime2024.json` (AIME竞赛题)
  - `gpqa_diamond.json` (常识问题)

✓ 模型checkpoint已链接到 `checkpoints/` 中：
  - `checkpoints/orz_ckpt_1gpu/orz_0p5b_ppo_jericho_1012_1gpu/iter12/policy/`

### 3. 直接运行（无需修改配置）！

```bash
python eval_orz.py
```

如需更改模型或数据集，编辑 `eval_orz.py` 中的 `checkpoint_path` 和 `eval_prompt_data`：

```python
if __name__ == "__main__":
    # 选择不同的checkpoint
    checkpoint_path = "checkpoints/orz_ckpt_1gpu/orz_0p5b_ppo_jericho_1012_1gpu/iter12/policy"

    # 选择要评估的数据集
    eval_prompt_data=[
        "data/eval_data/math500.json",
        "data/eval_data/aime2024.json",
    ]
```

## 常见问题

**Q: 我应该从哪里获得数据集？**
A: 你可以从以下地方获得：
- Math500: HuggingFace datasets library
- AIME2024: 竞赛官方网站
- GPQA Diamond: HuggingFace datasets library

**Q: 运行需要多少GPU显存？**
A:
- 最少: 8GB (设置 `gpu_memory_utilization=0.2`)
- 推荐: 24GB+ (设置 `gpu_memory_utilization=0.3-0.5`)

**Q: 如何使用多个GPU？**
A: 修改 `vllm_num_engines` 参数：
```python
eval_config = EvaluatorConfig(
    vllm_num_engines=2,  # 使用2个GPU
    # ...
)
```

**Q: 如何使用HuggingFace上的模型？**
A: 直接在 `model_path` 中使用模型ID：
```python
eval_config = EvaluatorConfig(
    model_path="Qwen/Qwen2.5-7B",
    tokenizer_path="Qwen/Qwen2.5-7B",
    # ...
)
```

## 文件结构说明

```
orz-eval/
├── eval_orz.py                    # 主评估脚本 ✓ 直接运行
├── requirements.txt               # Python依赖列表
├── README.md                      # 详细文档
├── QUICKSTART.md                 # 本文件
├── data/
│   └── eval_data/                # 放置JSON数据集的位置
│       ├── math500.json          # (你的数据)
│       ├── aime2024.json         # (你的数据)
│       └── gpqa_diamond.json     # (你的数据)
├── eval_results/                 # 评估结果输出目录（自动创建）
└── orz/                           # 核心库 (无需修改)
    ├── ppo/
    │   ├── dataset.py            # 数据集基类
    │   ├── deepspeed_strategy.py # 策略类
    │   ├── vllm_utils.py         # vLLM工具
    │   └── tools/
    │       └── math_utils.py     # 数学验证
    └── exp_engine/
        ├── accelerators/
        │   └── inference/
        │       ├── vllm_engine.py      # vLLM引擎
        │       └── vllm_worker_wrap.py # vLLM包装
        └── parallels/
            └── orz_distributed_c10d.py # 分布式通信
```

## 关键参数说明

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `model_path` | checkpoint或HF模型ID | 你的模型路径 |
| `vllm_num_engines` | 1-4 | vLLM引擎数量（受GPU限制） |
| `gpu_memory_utilization` | 0.2-0.5 | GPU内存使用率 |
| `generate_max_len` | 4000-8000 | 最大生成长度 |

## 完整示例

```python
from eval_orz import Evaluator, EvaluatorConfig
import asyncio

# 配置
config = EvaluatorConfig(
    model_path="/path/to/your/model",
    tokenizer_path="/path/to/your/model",
    vllm_num_engines=1,
    gpu_memory_utilization=0.3,
    eval_prompt_data=[
        "data/eval_data/math500.json",
    ],
    output_dir="my_eval_results",
)

# 运行
evaluator = Evaluator(config)
try:
    results = asyncio.run(evaluator.eval())
    print("✓ 评估完成！")
    print(f"准确率: {results['eval_accuracy']:.2%}")
finally:
    evaluator.cleanup()
```

## 获取帮助

1. 查看详细文档: `README.md`
2. 检查日志输出中的错误信息
3. 确保所有依赖正确安装: `pip list | grep -E 'ray|torch|vllm'`

---

祝你使用愉快！如有问题，请参考 README.md 中的故障排除部分。
