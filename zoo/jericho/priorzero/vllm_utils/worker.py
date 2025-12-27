class WorkerWrap:
    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles=None, empty_cache=False):
        import torch
        from vllm_utils.vllm_engine import get_physical_gpu_id

        if torch.distributed.get_rank() == 0:
            print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"

        handle = ipc_handles[get_physical_gpu_id()]
        device_id = self.device.index
        func, args = handle
        list_args = list(args)
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
        weight = func(*list_args)
        self.model_runner.model.load_weights(weights=[(name, weight)])
        torch.cuda.synchronize()
    
    # def update_weight(self, name, dtype, shape, empty_cache=False):
    #     import torch

    #     """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
    #     if torch.distributed.get_rank() == 0:
    #         print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

    #     assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
    #     weight = torch.empty(shape, dtype=dtype, device="cuda")

    #     self._model_update_group.broadcast(weight, src=0, stream=torch.cuda.current_stream())
    #     self.model_runner.model.load_weights(weights=[(name, weight)])

    #     del weight
    
    def update_weight(self, name, dtype, shape, weight, empty_cache=False):  # pylint: disable=R0917, W0613
        import torch
        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        if torch.distributed.get_rank() == 0:
            print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")
            
        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight
