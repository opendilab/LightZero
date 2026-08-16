"""
vLLM-based VL Engine for multimodal inference.

This module provides a vLLM wrapper for Vision-Language (VL) models,
similar to the text-only vLLM engine but with multimodal support.
"""
import vllm
from typing import List, Union, Optional, Dict, Any
from PIL import Image
import numpy as np
from loguru import logger
from transformers import AutoProcessor


class VLActor:
    """
    vLLM Actor for Vision-Language (VL) models.

    Similar to LLMActor but with multimodal support.
    Applies ChatML formatting required by Instruct-tuned models.
    """

    def __init__(
        self,
        model: str = None,
        limit_mm_per_prompt: Optional[Dict[str, int]] = None,
        **kwargs
    ):
        """
        Args:
            model: Path to VL model
            limit_mm_per_prompt: Multimodal limits (e.g., {"image": 1})
            **kwargs: Additional vLLM arguments
        """
        self.kwargs = kwargs
        self.limit_mm_per_prompt = limit_mm_per_prompt or {"image": 1}
        self.model_path = model

        logger.info(f"Initializing VLActor with model: {model}")
        logger.info(f"  Multimodal limits: {self.limit_mm_per_prompt}")

        self.llm = vllm.LLM(
            model=model,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            **self.kwargs
        )

        # Load processor/tokenizer for chat template
        try:
            self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
            logger.info(f"  ✓ Loaded processor for chat template")
        except Exception as e:
            logger.warning(f"  Failed to load processor: {e}. Will use raw prompts (may cause garbled output).")
            self.processor = None

    def _apply_chat_template(self, prompt: str, system_prompt: Optional[str] = None, num_images: int = 1) -> str:
        """
        Apply ChatML template to convert raw user prompt into model-expected format.

        For Qwen2.5-VL / Qwen3-VL Instruct models, the expected format is:
            <|im_start|>system\nYou are a helpful assistant.<|im_end|>
            <|im_start|>user\n<image>\n<prompt><|im_end|>
            <|im_start|>assistant\n

        Supports multiple images by inserting multiple {"type": "image"} entries.
        """
        if self.processor is None:
            return prompt

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content = []
        for _ in range(num_images):
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})

        try:
            formatted = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return formatted
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}. Using raw prompt.")
            return prompt

    def sleep(self, level=1):
        """Put the engine to sleep to free GPU memory."""
        if hasattr(self.llm, 'sleep'):
            self.llm.sleep(level=level)

    def wake_up(self):
        """Wake up the engine from sleep mode."""
        if hasattr(self.llm, 'wake_up'):
            self.llm.wake_up()

    def update_weight(self, name, dtype, shape, weight, empty_cache=False):
        """Sync a single parameter from the DeepSpeed policy model to the vLLM engine."""
        return self.llm.collective_rpc("update_weight", args=(name, dtype, shape, weight, empty_cache))

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return self.llm.collective_rpc("update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache))

    def reset_prefix_cache(self):
        """Reset prefix cache after weight update."""
        self.llm.llm_engine.reset_prefix_cache()

    def generate(
        self,
        images: List[Union[Image.Image, np.ndarray, List[Image.Image]]],
        prompts: Optional[List[str]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        sampling_params: Any = None,
        system_prompt: Optional[str] = None,
    ) -> List[Any]:
        """
        Generate responses for multimodal inputs.

        Args:
            images: List of images or image lists
            prompts: List of text prompts (mutually exclusive with prompt_token_ids)
            prompt_token_ids: List of token ID lists (mutually exclusive with prompts)
            sampling_params: vLLM SamplingParams
            system_prompt: Optional system prompt

        Returns:
            List of vLLM RequestOutput objects
        """
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be provided")
        if prompts is not None and prompt_token_ids is not None:
            raise ValueError("Cannot provide both prompts and prompt_token_ids")

        # Prepare multimodal inputs
        inputs = []

        if prompts is not None:
            # Text prompt mode (original)
            for image, prompt in zip(images, prompts):
                img_list = self._normalize_images(image)
                formatted_prompt = self._apply_chat_template(prompt, system_prompt=system_prompt, num_images=len(img_list))
                img_data = img_list if len(img_list) > 1 else img_list[0]

                inputs.append({
                    "prompt": formatted_prompt,
                    "multi_modal_data": {"image": img_data},
                })
        else:
            # Token IDs mode (for logprob extraction)
            for image, token_ids in zip(images, prompt_token_ids):
                img_list = self._normalize_images(image)
                img_data = img_list if len(img_list) > 1 else img_list[0]

                inputs.append({
                    "prompt_token_ids": token_ids,
                    "multi_modal_data": {"image": img_data},
                })

        # Generate
        responses = self.llm.generate(
            inputs,
            sampling_params=sampling_params,
            use_tqdm=False
        )

        return responses

    def _normalize_images(self, image: Union[Image.Image, np.ndarray, List]) -> List[Image.Image]:
        """Normalize image input to list of PIL Images."""
        if isinstance(image, list):
            img_list = []
            for img in image:
                if isinstance(img, np.ndarray):
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8)
                    if len(img.shape) == 3 and img.shape[0] == 3:
                        img = np.transpose(img, (1, 2, 0))
                    img = Image.fromarray(img)
                img_list.append(img)
            return img_list
        else:
            # Single image
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                if len(image.shape) == 3 and image.shape[0] == 3:
                    image = np.transpose(image, (1, 2, 0))
                image = Image.fromarray(image)
            return [image]


def create_vllm_vl_engine(
    tensor_parallel_size: int,
    pretrain: str,
    max_model_len: int,
    gpu_memory_utilization: float = 0.3,
    vllm_enable_sleep: bool = False,
    limit_mm_per_prompt: Optional[Dict[str, int]] = None,
    standalone: bool = False,
):
    """
    Create a vLLM engine for Vision-Language (VL) models.

    Args:
        tensor_parallel_size: Number of GPUs for tensor parallelism
        pretrain: Path to pretrained VL model
        max_model_len: Maximum sequence length
        gpu_memory_utilization: GPU memory utilization ratio
        vllm_enable_sleep: Whether to enable sleep mode
        limit_mm_per_prompt: Multimodal limits per prompt
        standalone: If True, skip DDP-specific args (external_launcher, worker_extension_cls).
                    Use this for single-process evaluation scripts.

    Returns:
        VLActor instance
    """
    if limit_mm_per_prompt is None:
        limit_mm_per_prompt = {"image": 1}

    logger.info("Creating vLLM VL engine:")
    logger.info(f"  Model: {pretrain}")
    logger.info(f"  Tensor Parallel Size: {tensor_parallel_size}")
    logger.info(f"  Max Model Length: {max_model_len}")
    logger.info(f"  GPU Memory Utilization: {gpu_memory_utilization}")
    logger.info(f"  Enable Sleep: {vllm_enable_sleep}")
    logger.info(f"  Multimodal Limits: {limit_mm_per_prompt}")

    # DDP-specific args are only needed when running under torchrun
    extra_kwargs = {}
    if not standalone:
        extra_kwargs["worker_extension_cls"] = "vllm_utils.worker.WorkerWrap"
        extra_kwargs["distributed_executor_backend"] = "external_launcher"

    vllm_engine = VLActor(
        model=pretrain,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        enable_sleep_mode=vllm_enable_sleep,
        limit_mm_per_prompt=limit_mm_per_prompt,
        trust_remote_code=True,
        **extra_kwargs,
    )

    if vllm_enable_sleep:
        vllm_engine.sleep()

    logger.info("✓ vLLM VL engine created successfully")

    return vllm_engine
