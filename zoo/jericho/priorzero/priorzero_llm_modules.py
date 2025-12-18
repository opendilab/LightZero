from __future__ import annotations
import os
import copy
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import deepspeed
import ray
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from ding.utils import build_logger
from utils.vllm_engine import create_vllm_engines, batch_vllm_engine_call  
from utils.generator import SamplesGenerator
from priorzero_policy import build_llm_prompt
from openrlhf.utils import get_strategy
from openrlhf.trainer.ray.utils import get_physical_gpu_id
from priorzero_utils import compute_approx_kl


def torch_dist_barrier_and_cuda_sync():
    """Synchronize distributed training and CUDA operations.
    This function ensures that:
    1. All distributed processes reach this point (barrier)
    2. All CUDA operations are completed (synchronize)
    """
    import torch
    torch.distributed.barrier()
    torch.cuda.synchronize()

@dataclass
class PriorZeroOpenRLHFLLMConfig:
    model_name_or_path: str
    bf16: bool = True

    prompt_max_len: int = 8192
    generate_max_len: int = 128
    use_cot: bool = True

    rft_loss_type: str = "reinforce++"   # "reinforce" | "reinforce++"
    rft_clip_epsilon: float = 0.2
    rft_kl_coef: float = 0.0

    # DeepSpeed
    zero_stage: int = 0  # 只提供 zero_optimization
    lr: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    micro_train_batch_size: int = 1
    train_batch_size: int=128
    gradient_accumulation_steps: int = 1
    ds_tensor_parallel_size: int = 1

    # vLLM engines (OpenRLHF)
    enable_vllm: bool = True
    enable_prefix_caching: bool = True
    vllm_num_engines: int = 1
    vllm_tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 0

class PriorZeroOpenRLHFLLMTrainer:
    """
    目标：
      - 复用 OpenRLHF 的 vLLM RayActor 引擎与 weight update RPC 
      - RFT 训练走 DeepSpeed（支持单进程/多进程）
      - 权重同步走 update_weight_cuda_ipc（同机同卡多进程最直接）
    """

    def __init__(self, cfg: PriorZeroOpenRLHFLLMConfig, tb_logger, exp_name, instance_name='rft_llm'):
        self.cfg = cfg
        self.lr = cfg.lr
        self.weight_decay = cfg.weight_decay
        self.cfg.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if tb_logger is not None:
            self._logger, _ = build_logger(
                path=f'./{exp_name}/log/{instance_name}', name=instance_name, need_tb=False
            )
            self._tb_logger = tb_logger
        else:
            pass
        self.rft_log = {}
        self.train_samples_cnt = 0
        
        if not ray.is_initialized():
            ray.init()

        self.use_cuda_ipc = True   

        self.strategy = get_strategy(self.cfg)
        self.strategy.setup_distributed()   # 分布式初始化 + tokenizer + model + optimizer + deepspeed.initialize
        
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
            device_map=None,
        )

        optim = self.strategy.create_optimizer(
            model,
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=100000,  
            eta_min=self.lr * 0.1
        )
        self.model_engine, self.optim, self.scheduler = self.strategy.prepare(
            (model, optim, scheduler),
            is_rlhf=False,
        )

        self.ref_model = None
        if cfg.rft_kl_coef > 0.0:
            self.ref_model = copy.deepcopy(model).eval().to(self.model_engine.device)
            for p in self.ref_model.parameters():
                p.requires_grad_(False)

        self.vllm_engines = None
        if cfg.enable_vllm:
            # [OOM-FIX] Enable vLLM sleep mode to release GPU memory when not in use
            # This is critical to avoid OOM during LLM training phase
            self.vllm_engines = create_vllm_engines(
                num_engines=cfg.vllm_num_engines,
                tensor_parallel_size=cfg.vllm_tensor_parallel_size,
                pretrain=cfg.model_name_or_path,
                seed=cfg.seed,
                full_determinism=False,
                enable_prefix_caching=cfg.enable_prefix_caching,
                enforce_eager=False,
                gpu_memory_utilization=cfg.gpu_memory_utilization,
                max_model_len=cfg.prompt_max_len + cfg.generate_max_len,
                vllm_enable_sleep=True,  # [OOM-FIX] Enable sleep mode (saves ~4.6GB GPU memory)
            )
            self.llm_prior_generator = SamplesGenerator(vllm_engines=self.vllm_engines,
                                                        strategy=self.strategy,
                                                        tokenizer=self.tokenizer,
                                                        prompt_max_len=cfg.prompt_max_len,
                                                        temperature=cfg.temperature,
                                                        top_p=cfg.top_p)
        
        self._logger.info(f"✓ Load LLM Model in {cfg.model_name_or_path}")
    
    def build_samples(
        self,
        raw_obs_list: List[List[str]],
        history_obs_list: List[List[List[Tuple[str, str, float]]]],
        action_logprob_list: Optional[List[List[Any]]] = None,
        target_values: Optional[torch.Tensor] = None,   # [B, T-1] 的 G_t
    ) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        B = len(raw_obs_list)
        if B == 0:
            return samples
        T = len(raw_obs_list[0])

        for b in range(B):
            for t in range(T - 1):
                current_obs = raw_obs_list[b][t]
                current_hist = history_obs_list[b][t]
                next_hist = history_obs_list[b][t + 1]

                _, true_action, reward_value = next_hist[-1]
                if not true_action:
                    continue

                instruction = build_llm_prompt(
                    current_obs=current_obs,
                    history=current_hist,
                    use_cot=self.cfg.use_cot,
                )
                prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": instruction}],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                old_logprob = None
                if action_logprob_list is not None:
                    old_logprob = action_logprob_list[b][t + 1][true_action]

                target_value = None
                if target_values is not None:
                    target_value = float(target_values[b][t].item())

                samples.append(
                    {
                        "instruction": instruction,
                        "prompt": prompt,
                        "target": true_action,
                        "reward": float(reward_value) if reward_value is not None else 0.0,
                        "target_value": target_value,           
                        "old_logprob": old_logprob,  # Reinforce++ ratio 需要
                    }
                )
        return samples

    def log_state_to_tb(self):
        if self._tb_logger is not None:
            for k, v in self.rft_log.items():
                self._tb_logger.add_scalar(f'learner_llm_iter/{k}', np.mean(v) if v is not None else 0.0, self.train_samples_cnt)
                
        self.rft_log = {}
        
    def _log_state(self, x, name='none'):
        if name in self.rft_log:
            self.rft_log[name].append(x)
        else:
            self.rft_log[name] = [x] 
    
    def train_rft_from_priorzero_batch(
        self,
        data: Tuple[torch.Tensor]
    ) -> Dict[str, float]:

        current_batch, target_batch = data
        obs_batch_ori, action_batch, target_action_batch, mask_batch, batch_index_tensor, weights, make_time, timestep_batch, raw_obs_list, history_obs_list, action_logprob_list = current_batch
        target_reward, target_value, target_policy = target_batch
        
        samples = self.build_samples(raw_obs_list, history_obs_list, action_logprob_list, target_value)
        if len(samples) == 0:
            return {"rft_loss": 0.0}

        if self.cfg.use_cot:
            all_instructions = [s["instruction"] for s in samples]
            all_prefix_cot = self.llm_prior_generator._build_cot_prefix_texts(all_instructions)
            for i, s in enumerate(samples):
                s["prefix_cot"] = all_prefix_cot[i]
        
        micro_train_batch_size = self.strategy.micro_train_batch_size
        gradient_accumulation_steps = self.strategy.accumulated_gradient
        clip_eps = self.cfg.rft_clip_epsilon
        kl_coef = self.cfg.rft_kl_coef
        loss_type = self.cfg.rft_loss_type.lower()

        self.model_engine.train()
        total_loss = 0.0

        for i in range(0, len(samples), micro_train_batch_size):
            chunk = samples[i:i + micro_train_batch_size]
            if self.cfg.use_cot:
                prompts_only = [s["prompt"] + s["prefix_cot"] + " " for s in chunk]
            else:
                prompts_only = [s["prompt"] for s in chunk]

            targets_only = [s["target"] + self.tokenizer.eos_token for s in chunk]
            
            prompts_ids_list = self.tokenizer(
                prompts_only,
                add_special_tokens=False,
                truncation=True,
                max_length=self.cfg.prompt_max_len - 20,
            )["input_ids"]
            
            tgt_ids_list = self.tokenizer(
                targets_only,
                add_special_tokens=False,
                truncation=True,
            )["input_ids"]
            
            full_ids_list = [c + t for c, t in zip(prompts_ids_list, tgt_ids_list)]
            inputs = self.tokenizer.pad({"input_ids": full_ids_list}, padding=True, return_tensors="pt").to(self.model_engine.device)

            labels = inputs.input_ids.clone()
            labels[inputs.attention_mask == 0] = -100

            for row, prompts_ids in enumerate(prompts_ids_list):
                pad_len = int((inputs.attention_mask[row] == 0).sum().item())
                p_len = len(prompts_ids)
                real_prompt_len = pad_len + p_len
                labels[row, :real_prompt_len] = -100

            outputs = self.model_engine(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            logits = outputs.logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()

            token_logp = -F.cross_entropy(logits.transpose(1, 2), shifted_labels, reduction="none")
            mask = (shifted_labels != -100).float()
            token_logp = token_logp * mask
            seq_logp = token_logp.sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)  # 与你现在的实现一致：mean logp
            self._log_state(x=seq_logp.mean().item(), name='rft_logprob')
            
            gt = torch.tensor([s["target_value"] if s["target_value"] is not None else s["reward"] for s in chunk],
                              device=self.model_engine.device, dtype=torch.float32)

            if loss_type == "reinforce":
                adv = gt
                self._log_state(x=adv.mean().item(), name='rft_advantage')
                
                loss = -(adv * seq_logp).mean()
            else:
                adv = (gt - gt.mean()) / (gt.std() + 1e-8)
                self._log_state(x=adv.mean().item(), name='rft_advantage')
                
                old_lp = torch.tensor([s["old_logprob"] for s in chunk],
                                      device=self.model_engine.device, dtype=torch.float32)
                ratio = torch.exp(seq_logp - old_lp)
                clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                surrogate1 = ratio * adv
                surrogate2 = clipped * adv
                
                used_ratio = torch.where(surrogate1 <= surrogate2, ratio, clipped)
                self._log_state(x=used_ratio.mean().item(), name='rft_ratio_used')
                
                loss = -(torch.min(surrogate1, surrogate2)).mean()
                
                # optional KL(pi || ref)
                if kl_coef > 0.0 and self.ref_model is not None:
                    with torch.no_grad():
                        ref_out = self.ref_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
                        ref_logits = ref_out.logits[:, :-1, :].contiguous()
                        ref_token_logp = -F.cross_entropy(ref_logits.transpose(1, 2), shifted_labels, reduction="none")
                        ref_token_logp = (ref_token_logp * mask)
                        ref_seq_logp = ref_token_logp.sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
                    kl_per_seq = compute_approx_kl(seq_logp, ref_seq_logp, kl_estimator='k2')
                    kl_loss = kl_per_seq.mean()
                    
                    self._log_state(x=kl_loss.item(), name='rft_kl')
                    
                    loss = loss + kl_coef * kl_loss
                
            total_loss += loss.item()
            self.strategy.backward(loss, self.model_engine, self.optim)
            self.strategy.optimizer_step(self.optim, self.model_engine, self.scheduler)

        self._log_state(x=total_loss/gradient_accumulation_steps, name='rft_loss')
        self.train_samples_cnt += len(samples)
        
        if self.vllm_engines is not None:
            self._broadcast_to_vllm()
        self.log_state_to_tb()
    
    def _broadcast_to_vllm(self):
        use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
        cache_reset_refs = []
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            for engine in self.vllm_engines:
                cache_reset_refs.append(engine.reset_prefix_cache.remote())

        torch.cuda.empty_cache()
        model = self.model_engine.module
        count, num_params = 0, len(list(model.named_parameters()))

        def _broadcast_param(param, count, num_params):
            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            # Fire all vllm engines for broadcast
            if torch.distributed.get_rank() == 0:
                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight.remote(name, dtype=param.dtype, shape=shape, empty_cache=count == num_params)
                    for engine in self.vllm_engines
                ]

                if use_ray:
                    import ray.util.collective as collective

                    collective.broadcast(param.data, 0, group_name=self._model_update_group)
                else:
                    self._model_update_group.broadcast(param.data, src=0, stream=torch.cuda.current_stream())
                ray.get(refs)

        def _handle_cuda_ipc(param, count, num_params):
            from torch.multiprocessing.reductions import reduce_tensor

            weight = param.data.clone()
            ipc_handle = reduce_tensor(weight)

            ipc_handle = {get_physical_gpu_id(): ipc_handle}
            ipc_handle_list = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

            if torch.distributed.get_rank() == 0:
                ipc_handles = {}
                for d in ipc_handle_list:
                    ipc_handles.update(d)

                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight_cuda_ipc.remote(
                        name,
                        dtype=param.dtype,
                        shape=shape,
                        ipc_handles=ipc_handles,
                        empty_cache=count == num_params,
                    )
                    for engine in self.vllm_engines
                ]
                ray.get(refs)
            torch_dist_barrier_and_cuda_sync()

        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param

            # broadcast
            if not self.use_cuda_ipc:
                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                if self.strategy.args.ds_tensor_parallel_size > 1:
                    with deepspeed.module_inject.layers.GatherReplacedLayerParams([param], model, enabled=True):
                        _broadcast_param(param, count, num_params)
                else:
                    with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                        _broadcast_param(param, count, num_params)
            # CUDA IPC
            else:
                if self.strategy.args.ds_tensor_parallel_size > 1:
                    with deepspeed.module_inject.layers.GatherReplacedLayerParams([param], model, enabled=True):
                        _handle_cuda_ipc(param, count, num_params)
                else:
                    with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                        _handle_cuda_ipc(param, count, num_params)

        if cache_reset_refs:
            ray.get(cache_reset_refs)
        torch.cuda.empty_cache()
        torch_dist_barrier_and_cuda_sync()
        
    