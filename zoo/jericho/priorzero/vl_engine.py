"""
Vision-Language (VL) Engine

This module provides a unified interface for various VL models
to generate action priors from image observations.

Supported models:
- Qwen-VL / Qwen2-VL / Qwen2.5-VL / Qwen3-VL (via vLLM)
- LLaVA-1.5 / LLaVA-1.6
- InternVL
"""
import os
from typing import List, Union, Optional, Dict, Any
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from loguru import logger

try:
    from vllm import LLM
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available. VL engine will use transformers backend.")


class VLEngine:
    """
    Base VL Engine class.

    Provides a unified interface for different VL implementations.
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: str = "cuda",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.3,
        **kwargs
    ):
        """
        Args:
            model_name: Model identifier (e.g., 'qwen-vl', 'llava-1.5')
            model_path: Path to model weights
            device: Device to run on
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization

        self.model = None
        self.tokenizer = None
        self.processor = None

        logger.info(f"Initializing VL Engine: {model_name}")
        self._load_model()

    def _load_model(self):
        """Load the VL model. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _load_model()")

    def generate(
        self,
        image: Union[Image.Image, np.ndarray],
        prompt: str,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        **kwargs
    ) -> str:
        """
        Generate text response from image and prompt.

        Args:
            image: Input image (PIL Image or numpy array)
            prompt: Text prompt
            temperature: Sampling temperature
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Generated text response
        """
        raise NotImplementedError("Subclasses must implement generate()")

    def batch_generate(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        prompts: List[str],
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        **kwargs
    ) -> List[str]:
        """
        Batch generate text responses.

        Args:
            images: List of input images
            prompts: List of text prompts
            temperature: Sampling temperature
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            List of generated text responses
        """
        # Default implementation: sequential generation
        results = []
        for image, prompt in zip(images, prompts):
            result = self.generate(image, prompt, temperature, max_new_tokens, **kwargs)
            results.append(result)
        return results

    def wake_up(self):
        """
        Wake up the engine (for vLLM sleep mode compatibility).
        Subclasses using vLLM should override this.
        """
        if hasattr(self, 'model') and hasattr(self.model, 'wake_up'):
            self.model.wake_up()
        # For non-vLLM engines, this is a no-op

    def sleep(self, level: int = 1):
        """
        Put the engine to sleep (for vLLM sleep mode compatibility).
        Subclasses using vLLM should override this.

        Args:
            level: Sleep level (1 = light sleep, 2 = deep sleep)
        """
        if hasattr(self, 'model') and hasattr(self.model, 'sleep'):
            self.model.sleep(level=level)
        # For non-vLLM engines, this is a no-op


class VLLMVLEngine(VLEngine):
    """
    vLLM-based VL Engine for multimodal models.

    This engine uses vLLM's native multimodal support for efficient inference
    with sleep/wake_up functionality for memory management.

    Supports:
    - Qwen2.5-VL-2B-Instruct / Qwen2.5-VL-7B-Instruct
    - Qwen3-VL-2B-Instruct
    - Any vLLM-supported multimodal model
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: str = "cuda",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.3,
        max_model_len: int = 8192,
        enable_sleep: bool = True,
        limit_mm_per_prompt: Optional[Dict[str, int]] = None,
        standalone: bool = False,
        **kwargs
    ):
        """
        Args:
            model_name: Model identifier
            model_path: Path to model weights
            device: Device to run on
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
            max_model_len: Maximum sequence length
            enable_sleep: Whether to enable sleep mode
            limit_mm_per_prompt: Multimodal limits per prompt (e.g., {"image": 5})
        """
        self.max_model_len = max_model_len
        self.enable_sleep = enable_sleep
        self.limit_mm_per_prompt = limit_mm_per_prompt or {"image": 1}
        self.standalone = standalone

        # Call parent init which will call _load_model
        super().__init__(
            model_name=model_name,
            model_path=model_path,
            device=device,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            **kwargs
        )

    def _load_model(self):
        """Load VL model using vLLM."""
        try:
            from vllm_utils.vl_engine import create_vllm_vl_engine

            logger.info(f"Loading VL model with vLLM from {self.model_path}")

            self.model = create_vllm_vl_engine(
                tensor_parallel_size=self.tensor_parallel_size,
                pretrain=self.model_path,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                vllm_enable_sleep=self.enable_sleep,
                limit_mm_per_prompt=self.limit_mm_per_prompt,
                standalone=self.standalone,
            )

            logger.info("✓ vLLM VL engine loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load vLLM VL engine: {e}")
            raise

    def generate(
        self,
        image: Union[Image.Image, np.ndarray, List[Image.Image]],
        prompt: str,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        system_prompt: Optional[str] = None,
        return_logprobs: bool = False,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """Generate response using vLLM. Supports single image or image list."""
        from vllm import SamplingParams

        # Sampling parameters with sane defaults to prevent garbled output
        sampling_params = SamplingParams(
            temperature=max(temperature, 0.1),
            max_tokens=max_new_tokens,
            top_p=kwargs.pop('top_p', 0.95),
            top_k=kwargs.pop('top_k', 50),
            repetition_penalty=kwargs.pop('repetition_penalty', 1.1),
            logprobs=None,
            prompt_logprobs=1 if return_logprobs else None,
            **kwargs
        )

        # Generate (VLActor expects lists)
        outputs = self.model.generate(
            images=[image],
            prompts=[prompt],
            sampling_params=sampling_params,
            system_prompt=system_prompt,
        )

        # Extract text and logprobs from output
        if outputs and len(outputs) > 0:
            output = outputs[0].outputs[0]
            text = output.text
            if return_logprobs:
                return {
                    'text': text,
                    'prompt_logprobs': outputs[0].prompt_logprobs if hasattr(outputs[0], 'prompt_logprobs') else None
                }
            return text
        return "" if not return_logprobs else {'text': "", 'prompt_logprobs': None}

    def batch_generate_with_token_ids(
        self,
        images: List[Union[Image.Image, np.ndarray, List[Image.Image]]],
        prompt_token_ids: List[List[int]],
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        return_logprobs: bool = False,
        **kwargs
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """
        Batch generate with token IDs input (aligned with LLM).
        This allows extracting logprobs for the full sequence including appended actions.
        """
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=max(temperature, 0.1),
            max_tokens=max_new_tokens,
            top_p=kwargs.pop('top_p', 0.95),
            top_k=kwargs.pop('top_k', 50),
            repetition_penalty=kwargs.pop('repetition_penalty', 1.1),
            logprobs=None,
            prompt_logprobs=1 if return_logprobs else None,
            **kwargs
        )

        # Generate with token IDs
        outputs = self.model.generate(
            images=images,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )

        # Extract results
        results = []
        for output in outputs:
            if output.outputs:
                out = output.outputs[0]
                text = out.text
                if return_logprobs:
                    results.append({
                        'text': text,
                        'prompt_logprobs': output.prompt_logprobs if hasattr(output, 'prompt_logprobs') else None
                    })
                else:
                    results.append(text)
            else:
                results.append("" if not return_logprobs else {'text': "", 'prompt_logprobs': None})

        return results

    def batch_generate(
        self,
        images: List[Union[Image.Image, np.ndarray, List[Image.Image]]],
        prompts: List[str],
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        system_prompt: Optional[str] = None,
        return_logprobs: bool = False,
        **kwargs
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """Batch generate responses using vLLM. Supports single images or image lists per prompt."""
        from vllm import SamplingParams

        # Sampling parameters with sane defaults to prevent garbled output
        sampling_params = SamplingParams(
            temperature=max(temperature, 0.1),
            max_tokens=max_new_tokens,
            top_p=kwargs.pop('top_p', 0.95),
            top_k=kwargs.pop('top_k', 50),
            repetition_penalty=kwargs.pop('repetition_penalty', 1.1),
            logprobs=None,
            prompt_logprobs=1 if return_logprobs else None,
            **kwargs
        )

        # Batch generate
        outputs = self.model.generate(
            images=images,
            prompts=prompts,
            sampling_params=sampling_params,
            system_prompt=system_prompt,
        )

        # Extract texts and logprobs
        results = []
        for output in outputs:
            if output.outputs:
                out = output.outputs[0]
                text = out.text
                if return_logprobs:
                    results.append({
                        'text': text,
                        'prompt_logprobs': output.prompt_logprobs if hasattr(output, 'prompt_logprobs') else None
                    })
                else:
                    results.append(text)
            else:
                results.append("" if not return_logprobs else {'text': "", 'prompt_logprobs': None})

        return results

    def wake_up(self):
        """Wake up the vLLM engine."""
        if hasattr(self.model, 'wake_up'):
            self.model.wake_up()

    def sleep(self, level: int = 1):
        """Put the vLLM engine to sleep."""
        if hasattr(self.model, 'sleep'):
            self.model.sleep(level=level)

    def update_weight(self, name, dtype, shape, weight, empty_cache=False):
        """Sync a single parameter from DeepSpeed policy model to the vLLM VL engine."""
        return self.model.update_weight(name, dtype, shape, weight, empty_cache)

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return self.model.update_weight_cuda_ipc(name, dtype, shape, ipc_handles, empty_cache)

    def reset_prefix_cache(self):
        """Reset prefix cache after weight update."""
        self.model.reset_prefix_cache()


class QwenVLEngine(VLEngine):
    """
    Qwen-VL / Qwen2-VL / Qwen2.5-VL / Qwen3-VL Engine

    Supports:
    - Qwen-VL-Chat
    - Qwen2-VL-2B-Instruct / Qwen2-VL-7B-Instruct
    - Qwen2.5-VL-2B-Instruct / Qwen2.5-VL-7B-Instruct
    - Qwen3-VL-2B-Instruct
    """

    def _load_model(self):
        """Load Qwen-VL model."""
        try:
            from transformers import AutoModelForVision2Seq, AutoTokenizer
            from transformers.generation import GenerationConfig

            logger.info(f"Loading Qwen-VL from {self.model_path}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            # Load model - Use AutoModelForVision2Seq for VL models
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                device_map="auto" if self.tensor_parallel_size > 1 else self.device,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            ).eval()

            # Set generation config
            self.model.generation_config = GenerationConfig.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            logger.info("✓ Qwen-VL model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Qwen-VL: {e}")
            raise

    def generate(
        self,
        image: Union[Image.Image, np.ndarray],
        prompt: str,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        **kwargs
    ) -> str:
        """Generate response using Qwen-VL."""
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

        # Save image temporarily (Qwen-VL requires image path)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            image.save(f.name)
            image_path = f.name

        try:
            # Build query with image
            query = self.tokenizer.from_list_format([
                {'image': image_path},
                {'text': prompt},
            ])

            # Generate
            response, history = self.model.chat(
                self.tokenizer,
                query=query,
                history=None,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

            return response

        finally:
            # Clean up temp file
            os.unlink(image_path)


class LLaVAEngine(VLEngine):
    """
    LLaVA Engine

    Supports:
    - LLaVA-1.5-7B
    - LLaVA-1.5-13B
    - LLaVA-1.6-7B
    """

    def _load_model(self):
        """Load LLaVA model."""
        try:
            from transformers import AutoProcessor, LlavaForConditionalGeneration

            logger.info(f"Loading LLaVA from {self.model_path}")

            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map="auto" if self.tensor_parallel_size > 1 else self.device,
                torch_dtype=torch.float16,
            ).eval()

            logger.info("✓ LLaVA model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load LLaVA: {e}")
            raise

    def generate(
        self,
        image: Union[Image.Image, np.ndarray],
        prompt: str,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        **kwargs
    ) -> str:
        """Generate response using LLaVA."""
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

        # Prepare inputs
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        prompt_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs = self.processor(
            images=image,
            text=prompt_text,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        # Decode
        response = self.processor.decode(
            output_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response


class InternVLEngine(VLEngine):
    """
    InternVL Engine

    Supports:
    - InternVL-Chat-V1.5
    - InternVL2-2B
    - InternVL2-8B
    """

    def _load_model(self):
        """Load InternVL model."""
        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading InternVL from {self.model_path}")

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            self.model = AutoModel.from_pretrained(
                self.model_path,
                device_map="auto" if self.tensor_parallel_size > 1 else self.device,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            ).eval()

            logger.info("✓ InternVL model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load InternVL: {e}")
            raise

    def generate(
        self,
        image: Union[Image.Image, np.ndarray],
        prompt: str,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        **kwargs
    ) -> str:
        """Generate response using InternVL."""
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

        # Generate
        response = self.model.chat(
            self.tokenizer,
            pixel_values=None,
            question=prompt,
            generation_config={
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'do_sample': temperature > 0,
            },
            image=image,
        )

        return response


# VL Model Registry
VL_MODEL_REGISTRY = {
    'qwen-vl': QwenVLEngine,
    'qwen2-vl': QwenVLEngine,
    'qwen2.5-vl': VLLMVLEngine,  # Use vLLM for Qwen2.5-VL
    'qwen3-vl': VLLMVLEngine,     # Use vLLM for Qwen3-VL
    'llava': LLaVAEngine,
    'llava-1.5': LLaVAEngine,
    'llava-1.6': LLaVAEngine,
    'internvl': InternVLEngine,
    'internvl2': InternVLEngine,
}


def create_vl_engine(
    model_name: str,
    model_path: str,
    device: str = "cuda",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.3,
    **kwargs
) -> VLEngine:
    """
    Factory function to create VL engine.

    Args:
        model_name: Model identifier (e.g., 'qwen-vl', 'llava-1.5')
        model_path: Path to model weights
        device: Device to run on
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization ratio

    Returns:
        VLEngine instance
    """
    # Normalize model name
    model_name_lower = model_name.lower()

    # Find matching engine class
    engine_class = None
    for key, cls in VL_MODEL_REGISTRY.items():
        if key in model_name_lower:
            engine_class = cls
            break

    if engine_class is None:
        raise ValueError(
            f"Unknown VL model: {model_name}. "
            f"Supported models: {list(VL_MODEL_REGISTRY.keys())}"
        )

    # Create engine
    engine = engine_class(
        model_name=model_name,
        model_path=model_path,
        device=device,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        **kwargs
    )

    return engine


if __name__ == "__main__":
    # Example usage
    print("VL Engine Module")
    print("=" * 80)
    print("\nSupported VL models:")
    for model_name in VL_MODEL_REGISTRY.keys():
        print(f"  - {model_name}")

    print("\nUsage:")
    print("  engine = create_vl_engine('qwen-vl', '/path/to/model')")
    print("  response = engine.generate(image, prompt)")
