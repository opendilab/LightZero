from contextlib import contextmanager
from typing import Optional, Union, List, Dict
from collections import defaultdict
import os
import math
from tqdm import tqdm
import numpy as np
import deepspeed
from torch.optim import Optimizer
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoConfig, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers.trainer import get_scheduler

from utils import compute_approx_kl, compute_entropy, masked_mean, torch_dist_barrier_and_cuda_sync, log_probs_from_logits

def _normalize_vllm_weight_name(name: str) -> str:
    if name.startswith("base_model.model."):
        name = name[len("base_model.model."):]
    name = name.replace(".base_layer.", ".")
    return name


def _should_skip_vllm_sync_param(name: str) -> bool:
    return any(marker in name for marker in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"))


def _validate_vllm_sync_config(args, train_mode: str, vllm_engine) -> None:
    if vllm_engine is None:
        return

    ds_tensor_parallel_size = getattr(args, "ds_tensor_parallel_size", 1)
    zero_stage = getattr(args, "zero_stage", 2)

    if ds_tensor_parallel_size != 1:
        raise NotImplementedError(
            "PolicyModel._deepspeed_broadcast currently supports only ds_tensor_parallel_size == 1. "
            f"Got ds_tensor_parallel_size={ds_tensor_parallel_size}. "
            "The active vLLM sync path does not safely handle DeepSpeed tensor parallel shards yet."
        )

    if zero_stage == 3 and train_mode == "lora":
        raise NotImplementedError(
            "PolicyModel._deepspeed_broadcast does not support train_mode='lora' with zero_stage=3. "
            "This path needs adapter merge/unmerge together with ZeRO-3 sharded parameters, which is not "
            "validated in the current implementation."
        )

class Actor(nn.Module):
    """
    Base class for Actor models in reinforcement learning.

    This class serves as a foundation for implementing various actor models, which are responsible for selecting actions based on the policy learned from the environment.

    Args:
        pretrain_or_model (nn.Module): A pretrained model or a new model instance to be used as the actor.
        attn_implementation (str, optional): Attention mechanism implementation to use. Defaults to "flash_attention_2".
        bf16 (bool, optional): Enable bfloat16 precision for model computations. Defaults to True.
        ds_config (dict, optional): Configuration for DeepSpeed, enabling model partitioning across multiple GPUs. Defaults to None.
        device_map (dict, optional): Device mapping for loading the model onto specific devices. Defaults to None.
        temperature (float, optional): Temperature for action selection. Defaults to 1.0.
    """

    def __init__(
        self,
        pretrain_or_model: str,
        attn_implementation="flash_attention_2",
        bf16=True,
        ds_config=None,
        device_map=None,
        temperature=1.0,
        train_mode_cfg=None,
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.temperature = temperature
        self.pretrain_or_model = pretrain_or_model
        self.train_mode_cfg = train_mode_cfg if train_mode_cfg is not None else {"mode": "full"}
        self.train_mode = self.train_mode_cfg.get("mode", "full")
        attn_impl = attn_implementation

        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            _ = HfDeepSpeedConfig(ds_config)
        else:
            _ = None

        # Detect if model is VL (Vision-Language) or LLM (Language Model)
        config = AutoConfig.from_pretrained(pretrain_or_model, trust_remote_code=True)
        is_vl = hasattr(config, 'vision_config') or 'VL' in config.__class__.__name__

        if is_vl:
            # Use AutoModelForVision2Seq for VL models (e.g., Qwen2.5-VL, Qwen3-VL)
            self.model = AutoModelForVision2Seq.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_impl,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )
        else:
            # Use AutoModelForCausalLM for text-only LLM models
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_impl,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )
        self.model.config.use_cache = False

        if self.train_mode == "lora":
            self.model.enable_input_require_grads()
            target_modules = self.train_mode_cfg.get("lora_target_modules")
            target_modules = list(target_modules) if target_modules else None
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.train_mode_cfg.get("lora_r", 16),
                lora_alpha=self.train_mode_cfg.get("lora_alpha", 32),
                lora_dropout=self.train_mode_cfg.get("lora_dropout", 0.05),
                bias=self.train_mode_cfg.get("lora_bias", "none"),
                target_modules=target_modules,
            )
            self.model = get_peft_model(self.model, lora_config)
        elif self.train_mode != "full":
            raise ValueError(f"Unsupported train_mode: {self.train_mode}")
        
        self.model.config.use_cache = False

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        return_entropy=False,
    ) -> torch.Tensor:

        foward_attention_mask = attention_mask
        rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        output = self.model(sequences, attention_mask=foward_attention_mask, position_ids=position_ids)

        if return_entropy:
            # Training path (micro-batch size 4): cast to fp32 for entropy + flash cross-entropy
            assert return_output
            output["logits"] = output["logits"].to(torch.float32)
            entropy = compute_entropy(output["logits"])
            setattr(output, "entropy", entropy[:, :-1])

        log_probs = log_probs_from_logits(output["logits"], rolled_sequences, temperature=self.temperature)

        log_probs = log_probs[:, :-1]

        action_log_probs = log_probs[:, -action_mask.shape[1] :] * action_mask.float()
        return (action_log_probs, output) if return_output else action_log_probs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        if hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()

class ReferenceModel:
    def __init__(self, strategy, pretrain):
        self.strategy = strategy
        model = Actor(
            pretrain,
            attn_implementation=strategy.args.attn_implementation,
            bf16=strategy.args.bf16,
            ds_config=strategy.get_ds_eval_config(
                offload=False
            ),
            temperature=strategy.args.temperature,
        )
        self.model = strategy.prepare(model, is_rlhf=True)
        self.model.eval()
        self.micro_train_batch_size = self.strategy.args.micro_train_batch_size
    
    @torch.no_grad()
    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return: action_log_probs [B, T_action]
        """
        device = torch.cuda.current_device()
        B = sequences.size(0)
        outs = []
        chunk_size = self.micro_train_batch_size

        for i in range(0, B, chunk_size):
            s = sequences[i : i + chunk_size].to(device)
            am = action_mask[i : i + chunk_size].to(device)
            attn = attention_mask[i : i + chunk_size].to(device)

            out = self.model(
                s,
                action_mask=am,
                attention_mask=attn,
            )
            outs.append(out)
        return torch.cat(outs, dim=0)

class BatchPPOTrainer:
    def __init__(
        self,
        strategy,
        actor,
        actor_optim,
        actor_scheduler,               
        micro_train_batch_size: int = 8,
        vllm_engine = None
    ):
        self.strategy = strategy
        self.args = strategy.args

        self.actor = actor
        self.actor_optim = actor_optim
        self.actor_scheduler = actor_scheduler
        self.vllm_engine = vllm_engine
        self.use_cuda_ipc = self.args.use_cuda_ipc

        self.micro_train_batch_size = micro_train_batch_size
        from models.loss import PolicyLoss
        self.policy_loss = PolicyLoss(
            clip_eps_low=self.args.eps_clip_low_high[0],
            clip_eps_high=self.args.eps_clip_low_high[1],
            policy_loss_type=self.args.policy_loss_type,
            enable_vllm_is_correction=self.args.enable_vllm_is_correction,
            vllm_is_truncated_threshold=self.args.vllm_is_truncated_threshold,
            use_cot=self.args.use_cot,
            cot_weight=self.args.cot_weight,
            use_mispo=self.args.use_mispo,
            mispo_token_truncated_threshold=self.args.mispo_token_truncated_threshold,
            mispo_traj_truncated_threshold=self.args.mispo_traj_truncated_threshold
        )
        self.train_iter = 0
        
    def train_batch(self, batch_data: Dict[str, torch.Tensor], kl_ctl: float, step_idx: int = 0) -> Dict[str, float]:
        device = torch.cuda.current_device()
        for k, v in batch_data.items():
            if torch.is_tensor(v):
                batch_data[k] = v.to(device)

        all_samples_size = batch_data["input_ids"].size(0)
        status_list = []
        pbar = tqdm(
            range(0, all_samples_size, self.micro_train_batch_size),
            desc=f"PPO batch step={step_idx}",
            disable=not self.strategy.is_rank_0(),
        )
        acc_grad_steps = self.strategy.accumulated_gradient
        metrics_buffer = defaultdict(list) # 用于累积 micro_step 指标的缓冲区
        kl_early_stop_threshold = getattr(self.args, 'kl_early_stop_threshold', None)
        kl_early_stopped = False
        for micro_step, start_idx in enumerate(pbar):
            end_idx = min(start_idx + self.micro_train_batch_size, all_samples_size)
            micro_batch = {
                'input_ids': batch_data['input_ids'][start_idx:end_idx],
                "attention_mask": batch_data['attention_mask'][start_idx:end_idx],
                "action_mask": batch_data['action_mask'][start_idx:end_idx],
                "advantages": batch_data['advantages'][start_idx:end_idx],
                "old_action_log_probs": batch_data['old_action_log_probs'][start_idx:end_idx],
                "log_status": batch_data['log_status'][start_idx:end_idx],
                "rollout_action_logprob": batch_data['rollout_action_logprob'][start_idx:end_idx],
            }
            micro_batch['ref_action_log_probs'] = batch_data['ref_action_log_probs'][start_idx:end_idx] if batch_data['ref_action_log_probs'] is not None else None
            action_log_probs, output = self.actor(
                micro_batch['input_ids'],
                micro_batch['action_mask'],
                attention_mask=micro_batch['attention_mask'],
                return_output=True,
                return_entropy=True,
            )
            actor_loss, clipfrac, clip_ratio, approx_kl, vllm_kl, mispo_token_mask, mispo_traj_mask = self.policy_loss(
                input_ids=micro_batch['input_ids'],
                log_probs=action_log_probs,
                old_log_probs=micro_batch['old_action_log_probs'],
                advantages=micro_batch['advantages'],
                action_mask=micro_batch['action_mask'],
                rollout_log_probs=micro_batch['rollout_action_logprob']
            )
            
            if self.args.rft_kl_coef > 0 and micro_batch['ref_action_log_probs'] is not None:
                kl = compute_approx_kl(
                    action_log_probs,
                    micro_batch['ref_action_log_probs'],
                    kl_estimator=self.args.kl_estimator
                )
                kl_loss = masked_mean(kl, micro_batch["action_mask"])
            else:
                kl_loss = torch.tensor(0.0, device=device)

            # KL early stopping: skip remaining micro-batches if ref_kl exceeds threshold
            kl_loss_item_for_check = kl_loss.detach().float().item()
            if kl_early_stop_threshold is not None and kl_early_stop_threshold > 0:
                if kl_loss_item_for_check > kl_early_stop_threshold:
                    if self.strategy.is_rank_0() and not kl_early_stopped:
                        import logging
                        logging.getLogger("priorzero.train").warning(
                            f"[KL Early Stop] ref_kl={kl_loss_item_for_check:.4f} > threshold={kl_early_stop_threshold}, "
                            f"skipping gradient updates for remaining micro-batches at micro_step={micro_step}"
                        )
                    kl_early_stopped = True
                    # Skip backward pass but still collect metrics for logging
                    entropy_loss = masked_mean(output.entropy[:, -micro_batch["action_mask"].shape[1] :], micro_batch["action_mask"])
                    policy_loss_item = actor_loss.detach().float().item()
                    clipfrac_item = clipfrac.detach().float().item()
                    clip_ratio_item = clip_ratio.detach().float().item()
                    approx_kl_item = approx_kl.detach().float().item()
                    kl_loss_item = kl_loss_item_for_check
                    entropy_loss_item = entropy_loss.detach().float().item()
                    input_response_length_item = micro_batch["attention_mask"].sum().detach().float().item() / micro_batch["attention_mask"].shape[0]
                    response_length_item = micro_batch["action_mask"].sum().detach().float().item() / micro_batch["action_mask"].shape[0]
                    input_length_item = input_response_length_item - response_length_item
                    metrics_buffer["policy_loss"].append(policy_loss_item)
                    metrics_buffer["clipfrac"].append(clipfrac_item)
                    metrics_buffer["clip_ratio"].append(clip_ratio_item)
                    metrics_buffer["approx_kl"].append(approx_kl_item)
                    metrics_buffer["ref_kl"].append(kl_loss_item)
                    metrics_buffer["input_length"].append(input_length_item)
                    metrics_buffer["response_length"].append(response_length_item)
                    metrics_buffer['entropy'].append(entropy_loss_item)
                    log_status = micro_batch["log_status"]
                    other_status = {k: [item[k] for item in log_status] for k in log_status[0].keys()}
                    for k, v in other_status.items():
                        metrics_buffer[k] = v
                    if ((micro_step + 1) % acc_grad_steps == 0) or ((micro_step + 1) == pbar.total):
                        self.train_iter += 1
                    continue

            loss = actor_loss + kl_loss * float(kl_ctl.value)
            
            entropy_loss = masked_mean(output.entropy[:, -micro_batch["action_mask"].shape[1] :], micro_batch["action_mask"])
            if self.args.entropy_loss_coef != 0:
                loss -= entropy_loss * self.args.entropy_loss_coef  
            
            self.strategy.backward(loss, self.actor, self.actor_optim)
            self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
            
            policy_loss_item = actor_loss.detach().float().item()
            clipfrac_item = clipfrac.detach().float().item()
            clip_ratio_item = clip_ratio.detach().float().item()
            approx_kl_item = approx_kl.detach().float().item()
            kl_loss_item = kl_loss.detach().float().item()
            input_response_length_item = micro_batch["attention_mask"].sum().detach().float().item() / micro_batch["attention_mask"].shape[0]
            response_length_item = micro_batch["action_mask"].sum().detach().float().item() / micro_batch["action_mask"].shape[0]
            input_length_item = input_response_length_item - response_length_item
            entropy_loss_item = entropy_loss.detach().float().item()
            total_loss_item = loss.detach().float().item()

            # PPO importance sampling ratio stats
            with torch.no_grad():
                ratio = torch.exp(action_log_probs - micro_batch['old_action_log_probs'])
                ratio_masked = (ratio * micro_batch['action_mask'].float())
                mask_sum = micro_batch['action_mask'].float().sum()
                ratio_mean_item = (ratio_masked.sum() / mask_sum).item() if mask_sum > 0 else 1.0
                ratio_std_item = ((((ratio - ratio_mean_item) ** 2) * micro_batch['action_mask'].float()).sum() / mask_sum).sqrt().item() if mask_sum > 0 else 0.0

                # Advantage stats for this micro-batch
                adv = micro_batch['advantages']
                adv_mean_item = adv.mean().item()
                adv_std_item = adv.std().item() if adv.numel() > 1 else 0.0

                # Log prob means
                log_prob_new_mean_item = masked_mean(action_log_probs, micro_batch['action_mask']).item()
                log_prob_old_mean_item = masked_mean(micro_batch['old_action_log_probs'], micro_batch['action_mask']).item()

            kl_coef_item = float(kl_ctl.value)

            pbar.set_postfix({
                "policy_loss": policy_loss_item,
                "clipfrac": clipfrac_item,
                "approx_kl": approx_kl_item,
                "iter": self.train_iter,
            })
            
            metrics_buffer["policy_loss"].append(policy_loss_item)
            metrics_buffer["clipfrac"].append(clipfrac_item)
            metrics_buffer["clip_ratio"].append(clip_ratio_item)
            metrics_buffer["approx_kl"].append(approx_kl_item)
            metrics_buffer["ref_kl"].append(kl_loss_item)
            metrics_buffer["input_length"].append(input_length_item)
            metrics_buffer["response_length"].append(response_length_item)
            metrics_buffer['entropy'].append(entropy_loss_item)
            metrics_buffer['total_loss'].append(total_loss_item)
            metrics_buffer['ratio_mean'].append(ratio_mean_item)
            metrics_buffer['ratio_std'].append(ratio_std_item)
            metrics_buffer['advantage_mean'].append(adv_mean_item)
            metrics_buffer['advantage_std'].append(adv_std_item)
            metrics_buffer['log_prob_new_mean'].append(log_prob_new_mean_item)
            metrics_buffer['log_prob_old_mean'].append(log_prob_old_mean_item)
            metrics_buffer['kl_coef'].append(kl_coef_item)
            if vllm_kl is not None:
                metrics_buffer['vllm_kl'].append(vllm_kl.item())
            if mispo_token_mask is not None:
                mispo_token_mask = mispo_token_mask * micro_batch["action_mask"]
                metrics_buffer['mispo_token_ratio'].append((mispo_token_mask.sum() / micro_batch["action_mask"].sum()).item())
            if mispo_traj_mask is not None:
                metrics_buffer['mispo_traj_ratio'].append((mispo_traj_mask.sum() / mispo_traj_mask.shape[0]).item())

            log_status = micro_batch["log_status"]
            other_status = {k: [item[k] for item in log_status] for k in log_status[0].keys()}
            for k, v in other_status.items():
                metrics_buffer[k] = v
        
            if ((micro_step + 1) % acc_grad_steps == 0) or ((micro_step + 1) == pbar.total):
                self.train_iter += 1
                status = {
                    "policy_loss": np.mean(metrics_buffer['policy_loss']),
                    "clipfrac": np.mean(metrics_buffer['clipfrac']),
                    "clip_ratio": np.mean(metrics_buffer['clip_ratio']),
                    "approx_kl": np.mean(metrics_buffer['approx_kl']),
                    "ref_kl": np.mean(metrics_buffer['ref_kl']),
                    "entropy": np.mean(metrics_buffer['entropy']),
                    
                    "iter": self.train_iter,
                    "lr": self.actor_scheduler.get_last_lr()[0],
                    "global_grad_norm": self.actor_optim._global_grad_norm,

                    "input_length_max": np.max(metrics_buffer['input_length']),
                    "input_length_mean": np.mean(metrics_buffer['input_length']),
                    "input_length_min": np.min(metrics_buffer['input_length']),

                    "response_length_max": np.max(metrics_buffer['response_length']),
                    "response_length_mean": np.mean(metrics_buffer['response_length']),
                    "response_length_min": np.min(metrics_buffer['response_length']),

                    "value_advantage_max": np.max(metrics_buffer['value_advantage']),
                    "value_advantage_mean": np.mean(metrics_buffer['value_advantage']),
                    "value_advantage_min": np.min(metrics_buffer['value_advantage']),

                    "total_loss": np.mean(metrics_buffer['total_loss']),
                    "ratio_mean": np.mean(metrics_buffer['ratio_mean']),
                    "ratio_std": np.mean(metrics_buffer['ratio_std']),
                    "advantage_mean": np.mean(metrics_buffer['advantage_mean']),
                    "advantage_std": np.mean(metrics_buffer['advantage_std']),
                    "log_prob_new_mean": np.mean(metrics_buffer['log_prob_new_mean']),
                    "log_prob_old_mean": np.mean(metrics_buffer['log_prob_old_mean']),
                    "kl_coef": np.mean(metrics_buffer['kl_coef']),
                }
                if "final_advantage" in metrics_buffer:
                    status["final_advantage_max"] = np.max(metrics_buffer['final_advantage'])
                    status["final_advantage_mean"] = np.mean(metrics_buffer['final_advantage'])
                    status["final_advantage_min"] = np.min(metrics_buffer['final_advantage'])
                if "fmt_rewards" in metrics_buffer:
                    status["fmt_rewards"] = np.mean(metrics_buffer['fmt_rewards'])
                if "vllm_kl" in metrics_buffer:
                    status["vllm_kl"] = np.mean(metrics_buffer['vllm_kl'])
                
                if "mispo_token_ratio" in metrics_buffer:
                    status["mispo_token_ratio"] = np.mean(metrics_buffer['mispo_token_ratio'])
                if "mispo_traj_ratio" in metrics_buffer:
                    status["mispo_traj_ratio"] = np.mean(metrics_buffer['mispo_traj_ratio'])
                if kl_early_stopped:
                    status["kl_early_stopped"] = 1.0
                metrics_buffer.clear()

                status = self.strategy.all_reduce(status)
                status_list.append(status)
        
        return status_list
    
    def _deepspeed_broadcast(self):
        _validate_vllm_sync_config(self.strategy.args, self.actor.train_mode, self.vllm_engine)
        use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
        if use_prefix_cache:
            self.vllm_engine.reset_prefix_cache()

        torch.cuda.empty_cache()
        model = self.actor.model.module
        with self._merged_lora_adapter(model):
            sync_params = list(self._iter_vllm_sync_params(model))
            count, num_params = 0, len(sync_params)
            for name, param in sync_params:
                count += 1  # empty_cache at last param
                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                    shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                    self.vllm_engine.update_weight(name, dtype=param.dtype, shape=shape, weight=param.data, empty_cache=(count == num_params)) 
    
    def _broadcast_to_vllm(self):
        use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            self.vllm_engine.reset_prefix_cache()

        torch.cuda.empty_cache()
        model = self.actor.model
        count, num_params = 0, len(list(model.named_parameters()))

        def _broadcast_param(param, count, num_params):
            if torch.distributed.get_rank() == 0:
                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                self.vllm_engine.update_weight(name, dtype=param.dtype, shape=shape, empty_cache=count == num_params) 
                
                self._model_update_group.broadcast(param.data, src=0, stream=torch.cuda.current_stream())

        def _handle_cuda_ipc(param, count, num_params):
            from torch.multiprocessing.reductions import reduce_tensor

            weight = param.data.clone()
            ipc_handle = reduce_tensor(weight)

            from vllm_utils.vllm_engine import get_physical_gpu_id
            ipc_handle = {get_physical_gpu_id(): ipc_handle}
            ipc_handle_list = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

            if torch.distributed.get_rank() == 0:
                ipc_handles = {}
                for d in ipc_handle_list:
                    ipc_handles.update(d)

                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                self.vllm_engine.update_weight_cuda_ipc(
                    name,
                    dtype=param.dtype,
                    shape=shape,
                    ipc_handles=ipc_handles,
                    empty_cache=count == num_params,
                )

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
            else:
                if self.strategy.args.ds_tensor_parallel_size > 1:
                    with deepspeed.module_inject.layers.GatherReplacedLayerParams([param], model, enabled=True):
                        _handle_cuda_ipc(param, count, num_params)
                else:
                    with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                        _handle_cuda_ipc(param, count, num_params)

        torch.cuda.empty_cache()
        torch_dist_barrier_and_cuda_sync()

    def _iter_vllm_sync_params(self, model):
        for name, param in model.named_parameters():
            if _should_skip_vllm_sync_param(name):
                continue
            yield _normalize_vllm_weight_name(name), param

    @contextmanager
    def _merged_lora_adapter(self, model):
        if isinstance(model, PeftModel):
            if not hasattr(model, "merge_adapter") or not hasattr(model, "unmerge_adapter"):
                raise RuntimeError("Current PEFT version does not support merge_adapter/unmerge_adapter required for vLLM sync.")
            model.merge_adapter()
            try:
                yield model
            finally:
                model.unmerge_adapter()
        else:
            yield model


class PolicyModel:
    def __init__(
        self,
        strategy,
        pretrain: str,
        max_steps: Optional[int] = None,
        vllm_engine=None,
    ):
        self.strategy = strategy
        args = strategy.args

        self.vllm_engine = vllm_engine
        self.max_steps = max_steps

        if getattr(args, "vllm_num_engines", 0) > 0:
            if getattr(args, "vllm_sync_backend", "nccl") == "nccl":
                os.environ["NCCL_CUMEM_ENABLE"] = "0"

        actor = Actor(
            pretrain,
            attn_implementation=args.attn_implementation,
            bf16=args.bf16,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            temperature=args.temperature,
            train_mode_cfg=args.train_mode_dict,
        )
        strategy.print(actor)
        if args.train_mode_dict.mode == "lora":
            actor.print_trainable_parameters()

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrain, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        actor_optim = strategy.create_optimizer(
            actor,
            lr=args.learning_rate,
            betas=args.adam_betas,
            weight_decay=args.weight_decay,
        )

        if max_steps is None:
            max_steps = int(getattr(args, "max_steps", 1_000_000))

        actor_scheduler = get_scheduler(
            args.lr_scheduler,
            actor_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
        )
        
        if args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
            is_rlhf=True,
        )
    
        if strategy.args.deepspeed_enable_sleep:
            from strategy.deepspeed import offload_deepspeed_states
            offload_deepspeed_states(self.actor.model)

        self.trainer = BatchPPOTrainer(
            strategy,
            self.actor,
            actor_optim=self.actor_optim,
            actor_scheduler=self.actor_scheduler,
            micro_train_batch_size=args.micro_train_batch_size,
            vllm_engine = vllm_engine,
        )
        self.micro_train_batch_size = self.strategy.args.micro_train_batch_size

    def fit(self, batch_data, kl_ctl: float = 0.0):
        torch.cuda.empty_cache()
        self.actor.train()
        status = self.trainer.train_batch(batch_data, kl_ctl)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return status

    @torch.no_grad()
    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return: action_log_probs [B, T_action]
        """
        self.actor.eval()
        device = torch.cuda.current_device()
        B = sequences.size(0)

        outs = []
        chunk_size = self.micro_train_batch_size

        for i in range(0, B, chunk_size):
            s = sequences[i : i + chunk_size].to(device)
            am = action_mask[i : i + chunk_size].to(device)
            attn = attention_mask[i : i + chunk_size].to(device)
            out = self.actor(
                s,
                action_mask=am,
                attention_mask=attn,
            )
            outs.append(out)
        return torch.cat(outs, dim=0)

    def broadcast_to_vllm(self):
        # self.trainer._broadcast_to_vllm()
        self.trainer._deepspeed_broadcast()

    def save_model(self):
        args = self.strategy.args
        self.strategy.save_model(
            self.actor,
            self.tokenizer,
            args.save_path,
        )
    @property
    def train_iter(self):
        return self.trainer.train_iter

    def reload_states(self):
        from strategy.deepspeed import reload_deepspeed_states
        reload_deepspeed_states(self.actor.model)

    def offload_states(self):
        from strategy.deepspeed import offload_deepspeed_states
        offload_deepspeed_states(self.actor.model)