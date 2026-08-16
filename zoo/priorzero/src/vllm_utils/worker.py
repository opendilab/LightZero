class WorkerWrap:
    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles=None, empty_cache=False):
        import torch
        from vllm_utils.vllm_engine import get_physical_gpu_id

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"

        handle = ipc_handles[get_physical_gpu_id()]
        device_id = self.device.index
        func, args = handle
        list_args = list(args)
        list_args[6] = device_id
        weight = func(*list_args)
        self.model_runner.model.load_weights(weights=[(name, weight)])
        torch.cuda.synchronize()

    def update_weight(self, name, dtype, shape, weight, empty_cache=False):  # pylint: disable=R0917, W0613
        import torch
        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight
