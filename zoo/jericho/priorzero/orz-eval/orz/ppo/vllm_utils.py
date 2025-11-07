"""
vLLM utility functions for evaluation.
This module contains functions to create and manage vLLM inference engines.
"""

import ray
from loguru import logger
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from typing import Optional

from orz.exp_engine.accelerators.inference.vllm_engine import LLMActor

# Create a Ray remote version of LLMActor
LLMRayActor = ray.remote(LLMActor)


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    colocate_with_actor: bool,
    enable_chunked_prefill: bool = False,
    max_num_batched_tokens: int = 2048,
    gpu_memory_utilization: float = 0.85,
    max_num_seqs: int = 256,
    colocate_pg: Optional[PlacementGroup] = None,
):
    """
    Create vLLM inference engines for evaluation.

    Args:
        num_engines: Number of vLLM engines to create
        tensor_parallel_size: Tensor parallel size for each engine
        pretrain: Model path or HuggingFace model name
        seed: Random seed
        enable_prefix_caching: Whether to enable prefix caching
        enforce_eager: Whether to enforce eager execution
        max_model_len: Maximum model length
        colocate_with_actor: Whether to colocate with actor
        enable_chunked_prefill: Whether to enable chunked prefill
        max_num_batched_tokens: Maximum number of batched tokens
        gpu_memory_utilization: GPU memory utilization fraction
        max_num_seqs: Maximum number of sequences
        colocate_pg: Ray placement group for colocation

    Returns:
        List of Ray remote vLLM actors
    """
    vllm_engines = []
    if tensor_parallel_size > 1:
        assert not colocate_with_actor, "colocate_with_actor is not supported when tensor_parallel_size > 1"
        num_gpus = 0
        for i in range(num_engines):
            bundles = [{"GPU": 1, "CPU": 1}] * tensor_parallel_size
            pg = placement_group(bundles, strategy="PACK")
            ray.get(pg.ready())

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0
            )
            vllm_engines.append(
                LLMRayActor.options(num_cpus=1, num_gpus=num_gpus, scheduling_strategy=scheduling_strategy,).remote(
                    pretrain,
                    trust_remote_code=True,
                    tensor_parallel_size=tensor_parallel_size,
                    dtype="bfloat16",
                    seed=seed + i,
                    enable_prefix_caching=enable_prefix_caching,
                    enforce_eager=enforce_eager,
                    max_model_len=max_model_len,
                    enable_chunked_prefill=enable_chunked_prefill,
                    max_num_batched_tokens=max_num_batched_tokens if enable_chunked_prefill else None,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_num_seqs=max_num_seqs,
                    block_size=256,
                )
            )
    else:
        if not colocate_with_actor:
            num_gpus = 1
            num_cpus = 1
            bundles = [{"GPU": 1, "CPU": 1}] * num_engines
            pg = placement_group(bundles, strategy="PACK")
            ray.get(pg.ready())
        else:
            num_gpus = 0.2
            num_cpus = 0.2
            assert colocate_pg is not None, "colocate_pg must be provided when colocate_with_actor is True"

        for i in range(num_engines):
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=colocate_pg if colocate_with_actor else pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=i,
            )
            vllm_engines.append(
                LLMRayActor.options(
                    num_cpus=num_cpus,
                    num_gpus=num_gpus,
                    scheduling_strategy=scheduling_strategy,
                ).remote(
                    pretrain,
                    trust_remote_code=True,
                    tensor_parallel_size=tensor_parallel_size,
                    dtype="bfloat16",
                    seed=seed + i,
                    enable_prefix_caching=enable_prefix_caching,
                    enforce_eager=enforce_eager,
                    max_model_len=max_model_len,
                    enable_chunked_prefill=enable_chunked_prefill,
                    max_num_batched_tokens=max_num_batched_tokens if enable_chunked_prefill else None,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_num_seqs=max_num_seqs,
                    block_size=256,
                )
            )
        if colocate_with_actor:
            offload_refs = []
            for llm in vllm_engines:
                offload_refs.append(llm.offload_to_cpu.remote())
            ray.get(offload_refs)
            logger.info("Offloaded all vLLM engines to CPU")

    return vllm_engines
