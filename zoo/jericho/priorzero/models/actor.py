from typing import Optional, Union, List, Dict
import os
import math
from tqdm import tqdm

import deepspeed
from torch.optim import Optimizer
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers.trainer import get_scheduler

from utils import compute_approx_kl, compute_entropy, masked_mean, torch_dist_barrier_and_cuda_sync
from openrlhf.models.utils import log_probs_from_logits


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
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.temperature = temperature
        attn_impl = attn_implementation

        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            _ = HfDeepSpeedConfig(ds_config)
        else:
            _ = None

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrain_or_model,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16 if bf16 else "auto",
            device_map=device_map,
        )
        self.model.config.use_cache = False


    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        return_logprobs=False,
        return_entropy=False,
        logits_to_keep=None
    ) -> torch.Tensor:
        """Returns action log probs"""
        batch, seqlen = sequences.size()
        foward_attention_mask = attention_mask

        rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if logits_to_keep is not None:
            output = self.model(sequences, attention_mask=foward_attention_mask, position_ids=position_ids, logits_to_keep=logits_to_keep)
        else:
            output = self.model(sequences, attention_mask=foward_attention_mask, position_ids=position_ids)
            
        output["logits"] = output["logits"].to(torch.float32)

        if return_entropy:
            assert return_output
            entropy = compute_entropy(output["logits"])
            setattr(output, "entropy", entropy[:, :-1])

        return_action_log_probs = action_mask is not None
        if logits_to_keep is not None:
            logits_pred = output["logits"][:, :-1, :] 
            labels_tail = sequences[:, -action_mask.shape[1]:]
            log_probs = log_probs_from_logits(logits_pred.float(), labels_tail, temperature=self.temperature) 
            action_log_probs = log_probs * action_mask.float()
        else:
            log_probs = log_probs_from_logits(output["logits"], rolled_sequences, temperature=self.temperature)
            log_probs = log_probs[:, :-1]
            if not return_action_log_probs and return_logprobs:
                return (log_probs, output) if return_output else log_probs

            action_log_probs = log_probs[:, -action_mask.shape[1] :] * action_mask.float()
            
        if return_output:
            return action_log_probs, output 
        else:
            return action_log_probs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
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
        logits_to_keep: int = None
    ) -> torch.Tensor:
        """
        Return: action_log_probs [B, T_action]
        """
        device = torch.cuda.current_device()
        B = sequences.size(0)
        outs = []
        chunk_size = max(self.micro_train_batch_size, 32)
        
        sequences = sequences.to(device)
        attention_mask = attention_mask.to(device)
        action_mask = action_mask.to(device) 
        for i in range(0, B, chunk_size):
            s = sequences[i : i + chunk_size].to(device)
            am = action_mask[i : i + chunk_size].to(device)
            attn = attention_mask[i : i + chunk_size].to(device)

            out = self.model(
                s,
                action_mask=am,
                attention_mask=attn,
                logits_to_keep=logits_to_keep,
            )  
            outs.append(out)
        return torch.cat(outs, dim=0)

class BatchPPOTrainer:
    def __init__(
        self,
        strategy,
        actor,
        actor_optim,
        actor_scheduler=None,               
        micro_train_batch_size: int = 8,
        vllm_engines = None
    ):
        self.strategy = strategy
        self.args = strategy.args

        self.actor = actor
        self.actor_optim = actor_optim
        self.actor_scheduler = actor_scheduler
        self.vllm_engines = vllm_engines
        self.use_cuda_ipc = self.args.use_cuda_ipc

        self.micro_train_batch_size = micro_train_batch_size
        from models.loss import PolicyLoss
        self.policy_loss = PolicyLoss(
            clip_eps_low=self.args.eps_clip_low_high[0],
            clip_eps_high=self.args.eps_clip_low_high[1],
            policy_loss_type=self.args.policy_loss_type,
        )

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
        for micro_step, start_idx in enumerate(pbar):
            end_idx = min(start_idx + self.micro_train_batch_size, all_samples_size)
            micro_batch = {
                'input_ids': batch_data['input_ids'][start_idx:end_idx],
                "attention_mask": batch_data['attention_mask'][start_idx:end_idx],
                "action_mask": batch_data['action_mask'][start_idx:end_idx],
                "advantages": batch_data['advantages'][start_idx:end_idx],
                "old_action_logprob": batch_data['old_action_logprob'][start_idx:end_idx],
            }
            micro_batch['ref_action_log_probs'] = batch_data['ref_action_log_probs'][start_idx:end_idx] if batch_data['ref_action_log_probs'] is not None else None

            logits_to_keep = micro_batch['action_mask'].size(1) + 1
            action_log_probs, output = self.actor(
                micro_batch['input_ids'],
                micro_batch['action_mask'],
                attention_mask=micro_batch['attention_mask'],
                return_output=True,
                logits_to_keep=logits_to_keep, 
            )
            actor_loss, clip_ratio, ppo_kl, vllm_kl = self.policy_loss(
                action_log_probs,
                micro_batch['old_action_logprob'],
                micro_batch['advantages'],
                action_mask=micro_batch['action_mask'],
            )
            
            if self.args.rft_kl_coef > 0 and micro_batch['ref_action_log_probs'] is not None:
                kl = compute_approx_kl(
                    action_log_probs,
                    micro_batch['ref_action_log_probs'],
                    kl_estimator=self.args.kl_estimator
                )
                kl_loss = masked_mean(kl, micro_batch["action_mask"])
            else:
                kl_loss = 0.0
            
            loss = actor_loss + kl_loss * float(kl_ctl.value)
            
            self.strategy.backward(loss, self.actor, self.actor_optim)
            self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")

            status = {
                "policy_loss": actor_loss.detach().float().mean().item(),
                # "actor_lr": self.actor_scheduler.get_last_lr()[0],
                "actor_lr": self.args.learning_rate,
                "ppo_clip_ratio": clip_ratio.detach().float().mean().item(),
                "ppo_kl": ppo_kl.detach().float().mean().item(),
            }
            if isinstance(kl_loss, torch.Tensor):
                status["kl"] = kl_loss.detach().float().mean().item()
            else:
                status["kl"] = float(kl_loss)
            
            status = self.strategy.all_reduce(status)
            
            status_list.append(status)

            pbar.set_postfix({
                "act_loss": status["policy_loss"],
                "kl": status["kl"],
                "clip": status["ppo_clip_ratio"],
                "lr": status["actor_lr"],
            })

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean
        
    def _broadcast_to_vllm(self):
        use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            for engine in self.vllm_engines:
                engine.reset_prefix_cache()

        torch.cuda.empty_cache()
        model = self.actor.model
        count, num_params = 0, len(list(model.named_parameters()))

        def _broadcast_param(param, count, num_params):
            if torch.distributed.get_rank() == 0:
                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                for engine in self.vllm_engines:
                    engine.update_weight(name, dtype=param.dtype, shape=shape, empty_cache=count == num_params) 
                
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
                for engine in self.vllm_engines:    
                    engine.update_weight_cuda_ipc(
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


class PolicyModel:
    def __init__(
        self,
        strategy,
        pretrain: str,
        max_steps: Optional[int] = None,
        vllm_engines=None,
    ):
        self.strategy = strategy
        args = strategy.args

        self.vllm_engines = vllm_engines
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
        )
        strategy.print(actor)

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

        # actor_scheduler = get_scheduler(
        #     args.lr_scheduler,
        #     actor_optim,
        #     num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        #     num_training_steps=max_steps,
        #     scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
        # )
        
        if args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
            (actor, actor_optim, None),
            is_rlhf=True,
        )

        self.trainer = BatchPPOTrainer(
            strategy,
            self.actor,
            actor_optim=self.actor_optim,
            actor_scheduler=self.actor_scheduler,
            micro_train_batch_size=args.micro_train_batch_size,
            vllm_engines = vllm_engines,
        )

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
        action_mask: Optional[Union[int, list[int], torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_lens=None,
        to_cpu: bool = False,
    ) -> torch.Tensor:
        self.actor.eval()

        if action_mask is None:
            raise ValueError("action_mask is required for returning action_log_probs")

        device = torch.cuda.current_device()
        sequences = sequences.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True) if attention_mask is not None else None
        action_mask = action_mask.to(device, non_blocking=True) if torch.is_tensor(action_mask) else action_mask

        action_log_probs = self.actor(
            sequences,
            action_mask=action_mask,
            attention_mask=attention_mask,
            ring_attn_group=self.strategy.ring_attn_group, 
            packed_seq_lens=packed_seq_lens,
        )

        self.actor.train() 
        return action_log_probs.to("cpu") if to_cpu else action_log_probs

    def broadcast_to_vllm(self):
        self.trainer._broadcast_to_vllm()

    def save_model(self):
        args = self.strategy.args
        self.strategy.save_model(
            self.actor,
            self.tokenizer,
            args.save_path,
        )
