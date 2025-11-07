"""
独立评估模块 - 用于评估已训练的模型
"""

import asyncio
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional

import ray
from loguru import logger

from orz.ppo.tools.math_utils import is_equal, solution2answer
from orz.ppo.deepspeed_strategy import DeepspeedStrategy
from orz.ppo.vllm_utils import create_vllm_engines
from dataset.eval_dataset import EvalCustomDataset

# Global executor for async operations
executor = ThreadPoolExecutor(max_workers=64)


@dataclass
class EvaluatorConfig:
    """独立评估配置类"""
    # Model and tokenizer
    model_path: str  # checkpoint path or HF model name
    tokenizer_path: Optional[str] = None  # if None, use model_path

    # vLLM engine settings
    # 注意：参数对齐参考 Open-Reasoner-Zero/playground/orz_7b_ppo_jericho_1013.py
    vllm_num_engines: int = 1  # [对齐] 原始值为 8（multi-node），单 GPU 环境改为 1
    vllm_tensor_parallel_size: int = 1  # [对齐] 完全对应
    enable_prefix_caching: bool = True  # [对齐] 完全对应
    gpu_memory_utilization: float = 0.3  # [对齐] 完全对应
    max_model_len: int = 8192  # [对齐] 对应原文件的 max_len=8192

    # Generation settings
    temperature: float = 1.0  # [对齐] 完全对应
    top_p: float = 1.0  # [对齐] 完全对应
    top_k: int = -1  # [对齐] 完全对应
    generate_max_len: int = 8000  # [对齐] 完全对应
    stop: List[str] = field(default_factory=lambda: ["User:", "Human:", "Assistant:", "</answer>"])  # [对齐] 完全对应

    # Data settings
    eval_prompt_data: List[str] = field(default_factory=lambda: [
        "data/eval_data/eval_jericho_dataset_his10_4games_1.8k_20251013_instruct.json",  # [对齐] Jericho 评估数据
        "data/eval_data/math500.json",  # [对齐] 完全对应
        "data/eval_data/aime2024.json",  # [对齐] 完全对应
        "data/eval_data/gpqa_diamond.json",  # [对齐] 完全对应
    ])
    prompt_max_len: int = 2048  # [对齐] 完全对应

    # Output settings
    output_dir: str = "eval_results"
    save_detailed_results: bool = True

    def __post_init__(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path


class Evaluator:
    """独立评估类，支持从checkpoint加载模型进行评估"""

    def __init__(
        self,
        config: Optional[EvaluatorConfig] = None,
        model_path: Optional[str] = None,
        eval_prompt_data: Optional[List[str]] = None,
        model=None,
        tokenizer=None,
        vllm_engines: Optional[List] = None,
        **kwargs
    ):
        """
        初始化评估器

        支持多种初始化方式：
        1. 传入 EvaluatorConfig 对象（保留原有方式）
           Evaluator(config=EvaluatorConfig(...))

        2. 传入必要参数，其他使用默认值
           Evaluator(model_path="...", eval_prompt_data=[...])

        3. 传入已加载的模型对象
           Evaluator(model=my_model, tokenizer=my_tokenizer, eval_prompt_data=[...])
           或
           Evaluator(vllm_engines=[engine], eval_prompt_data=[...])

        Args:
            config: EvaluatorConfig 配置对象（可选）
            model_path: 模型路径（可选，当不传 config 时使用）
            eval_prompt_data: 评估数据路径列表（可选）
            model: 已加载的 transformers 模型对象（可选）
            tokenizer: 已加载的 tokenizer 对象（可选）
            vllm_engines: 已创建的 vLLM 引擎列表（可选）
            **kwargs: 其他 EvaluatorConfig 参数
        """
        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init()

        # 处理配置对象
        if config is not None:
            # 使用传入的 config 对象
            self.cfg = config
        else:
            # 从参数构建 config 对象
            if model_path is None and model is None and vllm_engines is None:
                raise ValueError("必须指定 model_path、model 或 vllm_engines")

            config_kwargs = {
                "model_path": model_path or "dummy_path",  # 当使用预加载模型时，可以是占位符
                "eval_prompt_data": eval_prompt_data or [
                    "data/eval_data/math500.json",
                    "data/eval_data/aime2024.json",
                    "data/eval_data/gpqa_diamond.json",
                ],
            }
            # 合并其他 kwargs
            config_kwargs.update(kwargs)
            self.cfg = EvaluatorConfig(**config_kwargs)

        self.tokenizer = tokenizer
        self.vllm_engines = vllm_engines or []
        self.eval_dataset = None
        self.executor = executor
        self._user_provided_model = model
        self._user_provided_tokenizer = tokenizer
        self._user_provided_vllm_engines = vllm_engines is not None

        logger.info(f"Initializing Evaluator with config: {self.cfg}")

        # Load components
        if not self._user_provided_tokenizer:
            self._load_tokenizer()
        if not self._user_provided_vllm_engines:
            self._create_vllm_engines(model)
        self._load_eval_datasets()

        logger.info("Evaluator initialization completed")

    def _load_tokenizer(self):
        """Load tokenizer from pretrained model"""
        from transformers import AutoTokenizer
        logger.info(f"Loading tokenizer from {self.cfg.tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.tokenizer_path,
            trust_remote_code=True,
        )

    def _create_vllm_engines(self, model=None):
        """Create vLLM inference engines or use transformers model"""
        if self._user_provided_vllm_engines:
            logger.info(f"Using user-provided {len(self.vllm_engines)} vLLM engines")
            return

        if model is not None:
            # 使用传入的 transformers 模型，通过 vLLM 包装
            logger.info(f"Creating vLLM engines from provided transformers model")
            # 这里需要从 transformers 模型创建 vLLM 引擎
            # 对于简单起见，我们使用模型的配置来创建引擎
            try:
                from vllm import LLM
                # 注意：这需要一个有效的模型路径或模型对象
                # 简单实现：如果提供了模型对象，使用其配置中的 model_name_or_path
                if hasattr(model, 'config') and hasattr(model.config, '_name_or_path'):
                    model_path = model.config._name_or_path
                else:
                    raise ValueError("Cannot determine model path from transformers model")
                self.vllm_engines = [LLM(model=model_path, **self._get_vllm_kwargs())]
                logger.info(f"Successfully created 1 vLLM engine from transformers model")
            except Exception as e:
                logger.warning(f"Failed to create vLLM from transformers model: {e}")
                raise
        else:
            # 从模型路径创建 vLLM 引擎
            logger.info(f"Creating {self.cfg.vllm_num_engines} vLLM engines from {self.cfg.model_path}")
            self.vllm_engines = create_vllm_engines(
                num_engines=self.cfg.vllm_num_engines,
                tensor_parallel_size=self.cfg.vllm_tensor_parallel_size,
                pretrain=self.cfg.model_path,
                seed=42,
                enable_prefix_caching=self.cfg.enable_prefix_caching,
                enforce_eager=False,
                max_model_len=self.cfg.max_model_len,
                colocate_with_actor=False,
                gpu_memory_utilization=self.cfg.gpu_memory_utilization,
            )
            logger.info(f"Successfully created {len(self.vllm_engines)} vLLM engines")

    def _get_vllm_kwargs(self) -> dict:
        """获取 vLLM 初始化参数"""
        return {
            "tensor_parallel_size": self.cfg.vllm_tensor_parallel_size,
            "enable_prefix_caching": self.cfg.enable_prefix_caching,
            "enforce_eager": False,
            "max_model_len": self.cfg.max_model_len,
            "gpu_memory_utilization": self.cfg.gpu_memory_utilization,
        }

    def _load_eval_datasets(self):
        """Load evaluation datasets"""
        logger.info(f"Loading evaluation datasets from {self.cfg.eval_prompt_data}")
        dialogues = []
        for file_path in self.cfg.eval_prompt_data:
            logger.info(f"Loading dataset from {file_path}")
            with open(file_path, "r") as f:
                loaded_data = json.load(f)
                for item in loaded_data:
                    # Add file name as metadata
                    item["file_name"] = os.path.splitext(os.path.basename(file_path))[0]
                dialogues.extend(loaded_data)

        logger.info(f"Loaded {len(dialogues)} samples from evaluation datasets")

        # Create strategy object for dataset processing
        strategy = DeepspeedStrategy()

        self.eval_dataset = EvalCustomDataset(
            dialogues,
            self.tokenizer,
            self.cfg.prompt_max_len,
            strategy,
            pretrain_mode=False,
            num_processors=1,
        )
        logger.info(f"Processed {len(self.eval_dataset)} evaluation samples")

    async def eval(self) -> dict:
        """
        执行评估

        Returns:
            dict: 包含各数据集准确率的字典
        """
        logger.info("Starting evaluation on datasets")
        from vllm import SamplingParams
        from torch.utils.data import DataLoader

        sampling_params = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            max_tokens=self.cfg.generate_max_len,
            stop=self.cfg.stop,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
        )

        # Create dataloader
        dataloader = DataLoader(
            self.eval_dataset,
            batch_size=len(self.eval_dataset),
            shuffle=False,
            drop_last=False,
        )

        output_for_save = []
        log_dict = defaultdict(float)

        for batch in dataloader:
            prompts = list(batch[0])
            answers = list(batch[1]["answer"])
            file_names = list(batch[1]["file_name"])

            # Distribute prompts to vLLM engines
            prompt_per_engine = (len(prompts) + len(self.vllm_engines) - 1) // len(self.vllm_engines)
            outputs = []
            for i, llm in enumerate(self.vllm_engines):
                start_idx = i * prompt_per_engine
                end_idx = min((i + 1) * prompt_per_engine, len(prompts))
                if start_idx < len(prompts):
                    outputs.append(
                        llm.generate.remote(
                            prompts=prompts[start_idx:end_idx],
                            sampling_params=sampling_params,
                        )
                    )

            # Gather outputs from all engines
            outputs = await asyncio.gather(*outputs)
            outputs = sum(outputs, [])

            # Extract final answers
            final_answers = []
            pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
            for output in outputs:
                matches = re.findall(pattern, output.outputs[0].text)
                if len(matches) > 0:
                    final_answers.append(matches[-1])
                else:
                    final_answers.append("")

            # Check correctness for each sample
            for prompt, output, final_answer, answer, file_name in zip(
                prompts, outputs, final_answers, answers, file_names
            ):
                label = solution2answer(answer)
                prefix_response = solution2answer(final_answer)
                iscorrect = await is_equal(label, prefix_response, self.executor)

                output_for_save.append(
                    dict(
                        prompt=prompt,
                        output=output.outputs[0].text,
                        final_answer=final_answer,
                        answer=answer,
                        iscorrect=iscorrect,
                    )
                )

                # Log metrics
                log_dict[f"{file_name}/total_response_len_in_char"] += len(output.outputs[0].text)
                log_dict[f"{file_name}/correct"] += iscorrect
                log_dict[f"{file_name}/total"] += 1

        # Calculate metrics per dataset
        all_file_names = [
            os.path.splitext(os.path.basename(file_path))[0]
            for file_path in self.cfg.eval_prompt_data
        ]

        for file_name in all_file_names:
            if log_dict[f"{file_name}/total"] > 0:
                log_dict[f"{file_name}/response_len_in_char"] = (
                    log_dict[f"{file_name}/total_response_len_in_char"]
                    / log_dict[f"{file_name}/total"]
                )
                log_dict[f"{file_name}/accuracy"] = (
                    log_dict[f"{file_name}/correct"] / log_dict[f"{file_name}/total"]
                )
                log_dict.pop(f"{file_name}/total_response_len_in_char")
                log_dict.pop(f"{file_name}/correct")
                log_dict.pop(f"{file_name}/total")

        # Calculate average accuracy
        accuracies = [log_dict[f"{fn}/accuracy"] for fn in all_file_names if f"{fn}/accuracy" in log_dict]
        if accuracies:
            log_dict["eval_accuracy"] = sum(accuracies) / len(accuracies)

        # Save results if requested
        if self.cfg.save_detailed_results:
            os.makedirs(self.cfg.output_dir, exist_ok=True)

            # Generate result filename
            dump_file_name = "eval_results"
            for file_name in all_file_names:
                if f"{file_name}/accuracy" in log_dict:
                    dump_file_name += f"_{file_name}_{log_dict[f'{file_name}/accuracy']:.4f}"
            dump_file_name += ".jsonl"

            result_path = os.path.join(self.cfg.output_dir, dump_file_name)
            logger.info(f"Saving evaluation results to {result_path}")
            with open(result_path, "w") as f:
                for item in output_for_save:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Log results
        logging_str = ",".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])
        logger.info(f"Evaluation completed: {logging_str}")

        return dict(log_dict)

    def cleanup(self):
        """清理资源"""
        logger.info("Cleaning up evaluator resources")
        self.vllm_engines.clear()
        if ray.is_initialized():
            ray.shutdown()
        logger.info("Cleanup completed")


if __name__ == "__main__":
    # Evaluation mode
    logger.info("Running in evaluation mode")

    # Model checkpoint path (relative to this script)
    # Available models in checkpoints/:
    #   - checkpoints/orz_ckpt_1gpu/orz_0p5b_ppo_jericho_1012_1gpu/iter12/policy
    #   - Or any other model from Open-Reasoner-Zero
    checkpoint_path = "/mnt/shared-storage-user/tangjia/orz_7b_ppo_jericho_1013/iter45/policy"

    # 简洁用法 - 示例 1：只传必要参数
    # evaluator = Evaluator(
    #     model_path=checkpoint_path,
    #     eval_prompt_data=[
    #         "data/eval_data/eval_jericho_dataset_his10_4games_1.8k_20251013_instruct.json",
    #         "data/eval_data/math500.json",
    #         "data/eval_data/aime2024.json",
    #         "data/eval_data/gpqa_diamond.json",
    #     ]
    # )

    # 简洁用法 - 示例 2：传必要参数并覆盖部分默认配置
    # evaluator = Evaluator(
    #     model_path=checkpoint_path,
    #     eval_prompt_data=[
    #         "data/eval_data/math500.json",
    #         "data/eval_data/aime2024.json",
    #     ],
    #     temperature=0.8,  # 覆盖默认值 1.0
    #     gpu_memory_utilization=0.5,  # 覆盖默认值 0.3
    # )

    # 原有用法 - 传入完整的 EvaluatorConfig 对象（参数已对齐至 orz_7b_ppo_jericho_1013.py）
    eval_config = EvaluatorConfig(
        model_path=checkpoint_path,
        tokenizer_path=checkpoint_path,
        vllm_num_engines=1,  # [对齐] 原始值为 8（multi-node），单 GPU 环境改为 1
        vllm_tensor_parallel_size=1,  # [对齐] 完全对应
        enable_prefix_caching=True,  # [对齐] 完全对应
        gpu_memory_utilization=0.3,  # [对齐] 完全对应
        max_model_len=8192,  # [对齐] 对应原文件的 max_len
        temperature=1.0,  # [对齐] 完全对应
        top_p=1.0,  # [对齐] 完全对应
        top_k=-1,  # [对齐] 完全对应
        generate_max_len=8000,  # [对齐] 完全对应
        stop=["User:", "Human:", "Assistant:", "</answer>"],  # [对齐] 完全对应
        eval_prompt_data=[  # [对齐] 默认包含 Jericho + math500 + aime2024 + gpqa_diamond
            "data/eval_data/eval_jericho_dataset_his10_4games_1.8k_20251013_instruct.json",  # [对齐] 参考文件的 Jericho 数据
            "data/eval_data/math500.json",  # [对齐] 完全对应
            "data/eval_data/aime2024.json",  # [对齐] 完全对应
            "data/eval_data/gpqa_diamond.json",  # [对齐] 完全对应
        ],
        prompt_max_len=2048,  # [对齐] 完全对应
        output_dir="eval_results",
        save_detailed_results=True,
    )
    evaluator = Evaluator(eval_config)

    # 简洁用法 - 示例 3：传已加载的模型对象
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    # tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    # evaluator = Evaluator(
    #     model=model,
    #     tokenizer=tokenizer,
    #     eval_prompt_data=[
    #         "data/eval_data/math500.json",
    #         "data/eval_data/aime2024.json",
    #     ]
    # )

    try:
        results = asyncio.run(evaluator.eval())
        logger.info(f"Evaluation results: {results}")
    finally:
        evaluator.cleanup()
