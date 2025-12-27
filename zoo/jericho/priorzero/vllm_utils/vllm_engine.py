import os
import queue
from typing import Any, List
import vllm

class LLMActor:
    def __init__(self, model: str = None, **kwargs):
        self.requests = {}
        self.kwargs = kwargs
        self.llm = vllm.LLM(model=model, **self.kwargs)

    # def update_weight(self, name, dtype, shape, empty_cache=False):
    #     return self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))
    
    def update_weight(self, name, dtype, shape, weight, empty_cache=False):
        return self.llm.collective_rpc("update_weight", args=(name, dtype, shape, weight, empty_cache))

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return self.llm.collective_rpc("update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache))

    def reset_prefix_cache(self):
        self.llm.llm_engine.reset_prefix_cache()

    def sleep(self, level=1):
        self.llm.sleep(level=level)

    def wake_up(self):
        self.llm.wake_up()

    def add_requests(self, sampling_params, prompt_token_ids):
        """
        Process requests from rank0 and generate responses.
        Since only rank0 will send requests, we don't need to track actor ranks.
        """
        from vllm.inputs import TokensPrompt
        self.sampling_params = sampling_params
        self.requests = [TokensPrompt(prompt_token_ids=r) for r in prompt_token_ids]

    def get_responses(self):
        """
        Return the responses for the actor with the given rank
        """
        responses = self.llm.generate(prompts=self.requests, sampling_params=self.sampling_params)
        self.requests = {}
        return responses


def create_vllm_engine(
    tensor_parallel_size: int,
    pretrain: str,
    enable_prefix_caching: bool,
    max_model_len: int,
    gpu_memory_utilization=None,
    vllm_enable_sleep=False,
):
    from packaging import version
    assert version.parse(vllm.__version__) > version.parse("0.8.2"), "OpenRLHF only supports vllm > 0.8.2"

    distributed_executor_backend = "external_launcher"

    vllm_engine = LLMActor(
        model=pretrain,
        worker_extension_cls="vllm_utils.worker.WorkerWrap",
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend=distributed_executor_backend,
        max_model_len=max_model_len,
        enable_prefix_caching=enable_prefix_caching,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        enable_sleep_mode=vllm_enable_sleep,
    )
    if vllm_enable_sleep:
        vllm_engine.sleep()
    return vllm_engine


def get_physical_gpu_id():
    import torch

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)
