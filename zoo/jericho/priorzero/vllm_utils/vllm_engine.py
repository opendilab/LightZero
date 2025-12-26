import os
import queue
from typing import Any, List

class LLMActor:
    def __init__(self, model: str = None, **kwargs):
        kwargs.pop("distributed_executor_backend", None)
        kwargs.pop("agent_func_path", None)
        if kwargs.get("gpu_memory_utilization") is None:
            kwargs.pop("gpu_memory_utilization", None)
            
        self.requests = {}
        import vllm
        from packaging import version
        if version.parse(vllm.__version__) >= version.parse("0.9.0"):
            os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        
        tensor_parallel_size = kwargs.get("tensor_parallel_size", 1)
        dist_backend = "mp" if tensor_parallel_size > 1 else "uni"
        
        print(f"Initializing vLLM Engine (Local) | TP: {tensor_parallel_size} | Backend: {dist_backend}")
        
        self.kwargs = kwargs
        self.model_path = model

        self.llm = vllm.LLM(model=model, distributed_executor_backend=dist_backend, **self.kwargs)

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray=False):
        return self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

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


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    enable_prefix_caching: bool,
    max_model_len: int,
    gpu_memory_utilization=None,
    vllm_enable_sleep=False,
    logprobs_mode=None,
):
    import vllm
    from packaging import version

    assert version.parse(vllm.__version__) > version.parse("0.8.2"), "OpenRLHF only supports vllm > 0.8.2"

    vllm_engines = []
    distributed_executor_backend = "uni"

    for i in range(num_engines):
        additional_kwargs = {}
        if logprobs_mode:
            additional_kwargs["logprobs_mode"] = logprobs_mode
            additional_kwargs["max_logprobs"] = 1
            assert version.parse(vllm.__version__) > version.parse(
                "0.10.0"
            ), "vLLM > 0.10.0 is required for logprobs_mode"

        vllm_engines.append(
            LLMActor(
                model=pretrain,
                enforce_eager=False,
                worker_extension_cls="vllm_utils.worker.WorkerWrap",
                tensor_parallel_size=tensor_parallel_size,
                seed=seed + i,
                distributed_executor_backend=distributed_executor_backend,
                max_model_len=max_model_len,
                enable_prefix_caching=enable_prefix_caching,
                dtype="bfloat16",
                trust_remote_code=True,
                gpu_memory_utilization=gpu_memory_utilization,
                enable_sleep_mode=vllm_enable_sleep,
            )
        )
    if vllm_enable_sleep:
        batch_vllm_engine_call(vllm_engines, "sleep")
    return vllm_engines


def batch_vllm_engine_call(engines: List[Any], method_name: str, *args, rank_0_only: bool = True, **kwargs):
    import torch

    if torch.distributed.is_initialized():
        if rank_0_only and torch.distributed.get_rank() != 0:
            return None

    for engine in engines:
        method = getattr(engine, method_name)
        method(*args, **kwargs)


def get_physical_gpu_id():
    import torch

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)
