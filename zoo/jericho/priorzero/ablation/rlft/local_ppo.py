from __future__ import annotations

import gc
import math
import os
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers.trainer import get_scheduler


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim=None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(dim=dim)
    mask = mask.to(dtype=tensor.dtype, device=tensor.device)
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1.0)


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    log_probs = torch.log_softmax(logits / temperature, dim=-1)
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


class FixedKLController:
    def __init__(self, kl_coef: float):
        self.value = float(kl_coef)

    def update(self, current, n_steps):
        return None


class RLFTActor(nn.Module):
    def __init__(
        self,
        pretrain: str,
        attn_implementation: str,
        bf16: bool,
        ds_config: Optional[dict],
        temperature: float,
        train_mode_cfg=None,
        enable_value_head: bool = True,
    ) -> None:
        super().__init__()
        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            _ = HfDeepSpeedConfig(ds_config)

        self.temperature = temperature
        self.train_mode_cfg = train_mode_cfg if train_mode_cfg is not None else {"mode": "full"}
        self.train_mode = self.train_mode_cfg.get("mode", "full")
        self.enable_value_head = enable_value_head
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrain,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16 if bf16 else "auto",
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

        if enable_value_head:
            hidden_size = getattr(self.model.config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError("Cannot enable value head because model.config.hidden_size is missing.")
            self.v_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        return_output: bool = False,
        return_entropy: bool = False,
        return_values: bool = False,
    ):
        if return_values and not self.enable_value_head:
            raise RuntimeError("return_values=True requires enable_value_head=True.")

        rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        output = self.model(
            sequences,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=return_values,
        )
        logits = output.logits.to(torch.float32)

        if return_entropy:
            setattr(output, "entropy", entropy_from_logits(logits)[:, :-1])

        log_probs = log_probs_from_logits(logits, rolled_sequences, temperature=self.temperature)[:, :-1]
        action_log_probs = log_probs[:, -action_mask.shape[1] :] * action_mask.float()

        if return_values:
            values = self.v_head(output.hidden_states[-1]).squeeze(-1).to(torch.float32)[:, :-1]
            prompt_end_idx = (attention_mask.sum(dim=-1) - action_mask.sum(dim=-1) - 1).clamp(min=0).long()
            state_values = values.gather(dim=1, index=prompt_end_idx.unsqueeze(-1)).squeeze(-1)
            setattr(output, "state_values", state_values)

        return (action_log_probs, output) if return_output else action_log_probs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)


class RLFTReferenceModel:
    def __init__(self, strategy, pretrain: str):
        self.strategy = strategy
        model = RLFTActor(
            pretrain=pretrain,
            attn_implementation=strategy.args.attn_implementation,
            bf16=strategy.args.bf16,
            ds_config=strategy.get_ds_eval_config(offload=False),
            temperature=strategy.args.temperature,
            train_mode_cfg=strategy.args.train_mode_dict,
            enable_value_head=False,
        )
        self.model = strategy.prepare(model, is_rlhf=True)
        self.model.eval()
        self.micro_train_batch_size = strategy.args.micro_train_batch_size

    @torch.no_grad()
    def forward(self, sequences: torch.Tensor, action_mask: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        device = torch.cuda.current_device()
        outs = []
        chunk_size = max(1, self.micro_train_batch_size)
        sequences = sequences.to(device)
        attention_mask = attention_mask.to(device)
        action_mask = action_mask.to(device)
        for i in range(0, sequences.size(0), chunk_size):
            outs.append(
                self.model(
                    sequences[i : i + chunk_size],
                    action_mask=action_mask[i : i + chunk_size],
                    attention_mask=attention_mask[i : i + chunk_size],
                )
            )
        return torch.cat(outs, dim=0)


class RLFTPPOTrainer:
    def __init__(self, strategy, actor, actor_optim, actor_scheduler, micro_train_batch_size: int):
        self.strategy = strategy
        self.args = strategy.args
        self.actor = actor
        self.actor_optim = actor_optim
        self.actor_scheduler = actor_scheduler
        self.micro_train_batch_size = micro_train_batch_size
        self.train_iter = 0

    def train_batch(self, batch_data: Dict[str, torch.Tensor], kl_ctl: FixedKLController):
        device = torch.cuda.current_device()
        for k, v in batch_data.items():
            if torch.is_tensor(v):
                batch_data[k] = v.to(device)

        all_samples_size = batch_data["input_ids"].size(0)
        status_list = []
        metrics_buffer = defaultdict(list)
        pbar = tqdm(
            range(0, all_samples_size, self.micro_train_batch_size),
            desc="RLFT PPO batch",
            disable=not self.strategy.is_rank_0(),
        )
        acc_grad_steps = self.strategy.accumulated_gradient

        for micro_step, start_idx in enumerate(pbar):
            end_idx = min(start_idx + self.micro_train_batch_size, all_samples_size)
            micro_batch = {
                k: (v[start_idx:end_idx] if torch.is_tensor(v) else v)
                for k, v in batch_data.items()
            }
            micro_batch["log_status"] = batch_data["log_status"][start_idx:end_idx]

            action_log_probs, output = self.actor(
                micro_batch["input_ids"],
                micro_batch["action_mask"],
                attention_mask=micro_batch["attention_mask"],
                return_output=True,
                return_entropy=True,
                return_values=True,
            )
            current_action_logprobs = masked_mean(
                action_log_probs,
                micro_batch["action_mask"],
                dim=1,
            )

            actor_loss, clipfrac, clip_ratio, approx_kl = self._policy_loss(
                log_probs=current_action_logprobs,
                old_log_probs=micro_batch["old_action_log_probs"],
                advantages=micro_batch["advantages"],
            )

            if self.args.rft_kl_coef > 0 and micro_batch["ref_action_log_probs"] is not None:
                ref_action_logprobs = masked_mean(
                    micro_batch["ref_action_log_probs"],
                    micro_batch["action_mask"],
                    dim=1,
                )
                kl_loss = (current_action_logprobs - ref_action_logprobs).mean()
            else:
                kl_loss = torch.tensor(0.0, device=device)

            value_loss, value_clipfrac = self._value_loss(
                values=output.state_values,
                returns=micro_batch["returns"],
                old_values=micro_batch["old_values"],
            )
            entropy = masked_mean(output.entropy[:, -micro_batch["action_mask"].shape[1] :], micro_batch["action_mask"])

            loss = actor_loss + float(kl_ctl.value) * kl_loss + float(self.args.value_loss_coef) * value_loss
            if getattr(self.args, "entropy_loss_coef", 0.0) != 0:
                loss -= entropy * self.args.entropy_loss_coef

            self.strategy.backward(loss, self.actor, self.actor_optim)
            self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="rlft_actor")

            metrics_buffer["policy_loss"].append(actor_loss.detach().float().item())
            metrics_buffer["clipfrac"].append(clipfrac.detach().float().item())
            metrics_buffer["clip_ratio"].append(clip_ratio.detach().float().item())
            metrics_buffer["approx_kl"].append(approx_kl.detach().float().item())
            metrics_buffer["ref_kl"].append(kl_loss.detach().float().item())
            metrics_buffer["value_loss"].append(value_loss.detach().float().item())
            metrics_buffer["value_clipfrac"].append(value_clipfrac.detach().float().item())
            metrics_buffer["entropy"].append(entropy.detach().float().item())
            metrics_buffer["input_length"].append(
                (micro_batch["attention_mask"].sum() / micro_batch["attention_mask"].shape[0]).detach().float().item()
                - (micro_batch["action_mask"].sum() / micro_batch["action_mask"].shape[0]).detach().float().item()
            )
            metrics_buffer["response_length"].append(
                (micro_batch["action_mask"].sum() / micro_batch["action_mask"].shape[0]).detach().float().item()
            )
            for item in micro_batch["log_status"]:
                for k, v in item.items():
                    metrics_buffer[k].append(float(v))

            pbar.set_postfix(
                {
                    "policy_loss": metrics_buffer["policy_loss"][-1],
                    "value_loss": metrics_buffer["value_loss"][-1],
                    "iter": self.train_iter,
                }
            )

            if ((micro_step + 1) % acc_grad_steps == 0) or ((micro_step + 1) == pbar.total):
                self.train_iter += 1
                status = {
                    "iter": self.train_iter,
                    "policy_loss": float(np.mean(metrics_buffer["policy_loss"])),
                    "clipfrac": float(np.mean(metrics_buffer["clipfrac"])),
                    "clip_ratio": float(np.mean(metrics_buffer["clip_ratio"])),
                    "approx_kl": float(np.mean(metrics_buffer["approx_kl"])),
                    "ref_kl": float(np.mean(metrics_buffer["ref_kl"])),
                    "value_loss": float(np.mean(metrics_buffer["value_loss"])),
                    "value_clipfrac": float(np.mean(metrics_buffer["value_clipfrac"])),
                    "entropy": float(np.mean(metrics_buffer["entropy"])),
                    "input_length_mean": float(np.mean(metrics_buffer["input_length"])),
                    "response_length_mean": float(np.mean(metrics_buffer["response_length"])),
                    "valid_action_count_mean": float(np.mean(metrics_buffer["valid_action_count"])),
                    "value_advantage_mean": float(np.mean(metrics_buffer["value_advantage"])),
                    "value_advantage_max": float(np.max(metrics_buffer["value_advantage"])),
                    "value_advantage_min": float(np.min(metrics_buffer["value_advantage"])),
                    "lr": float(self.actor_scheduler.get_last_lr()[0]),
                }
                status = self.strategy.all_reduce(status)
                status_list.append(status)
                metrics_buffer.clear()

        return status_list

    def _policy_loss(self, log_probs, old_log_probs, advantages):
        log_ratio = log_probs - old_log_probs
        ratio = log_ratio.exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.args.eps_clip_low_high[0], 1 + self.args.eps_clip_low_high[1]) * advantages
        loss = -torch.min(surr1, surr2).mean()
        clipped = ratio.gt(1 + self.args.eps_clip_low_high[1]) | ratio.lt(1 - self.args.eps_clip_low_high[0])
        clipfrac = clipped.float().mean()
        clip_ratio = (surr2 < surr1).float().mean()
        approx_kl = (-log_ratio.detach()).mean()
        return loss, clipfrac, clip_ratio, approx_kl

    def _value_loss(self, values, returns, old_values):
        values_clipped = old_values + (values - old_values).clamp(
            -float(self.args.value_clip_eps), float(self.args.value_clip_eps)
        )
        value_loss_unclipped = (values - returns) ** 2
        value_loss_clipped = (values_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
        value_clipfrac = (value_loss_clipped > value_loss_unclipped).float().mean()
        return value_loss, value_clipfrac


class RLFTPolicyModel:
    def __init__(self, strategy, pretrain: str, max_steps: Optional[int] = None):
        self.strategy = strategy
        self.args = strategy.args
        self.max_steps = max_steps or int(getattr(self.args, "max_steps", 1_000_000))

        actor = RLFTActor(
            pretrain=pretrain,
            attn_implementation=self.args.attn_implementation,
            bf16=self.args.bf16,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            temperature=self.args.temperature,
            train_mode_cfg=self.args.train_mode_dict,
            enable_value_head=True,
        )
        strategy.print(actor)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        actor_optim = strategy.create_optimizer(
            actor,
            lr=self.args.learning_rate,
            betas=self.args.adam_betas,
            weight_decay=self.args.weight_decay,
        )
        actor_scheduler = get_scheduler(
            self.args.lr_scheduler,
            actor_optim,
            num_warmup_steps=math.ceil(self.max_steps * self.args.lr_warmup_ratio),
            num_training_steps=self.max_steps,
            scheduler_specific_kwargs={"min_lr": self.args.learning_rate * 0.1},
        )
        if self.args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": self.args.gradient_checkpointing_use_reentrant}
            )

        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
            is_rlhf=True,
        )
        self.trainer = RLFTPPOTrainer(
            strategy=strategy,
            actor=self.actor,
            actor_optim=self.actor_optim,
            actor_scheduler=self.actor_scheduler,
            micro_train_batch_size=self.args.micro_train_batch_size,
        )
        self.micro_train_batch_size = self.args.micro_train_batch_size

    def fit(self, batch_data, kl_ctl):
        torch.cuda.empty_cache()
        self.actor.train()
        status = self.trainer.train_batch(batch_data, kl_ctl)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return status

    @torch.no_grad()
    def forward_logprobs_values(self, sequences, action_mask, attention_mask) -> Tuple[torch.Tensor, torch.Tensor]:
        self.actor.eval()
        device = torch.cuda.current_device()
        sequences = sequences.to(device)
        attention_mask = attention_mask.to(device)
        action_mask = action_mask.to(device)
        logprob_outs, value_outs = [], []
        chunk_size = max(1, self.micro_train_batch_size)
        for i in range(0, sequences.size(0), chunk_size):
            log_probs, output = self.actor(
                sequences[i : i + chunk_size],
                action_mask=action_mask[i : i + chunk_size],
                attention_mask=attention_mask[i : i + chunk_size],
                return_output=True,
                return_values=True,
            )
            logprob_outs.append(log_probs)
            value_outs.append(output.state_values)
        return torch.cat(logprob_outs, dim=0), torch.cat(value_outs, dim=0)

    def save_model(self):
        if not self.strategy.is_rank_0():
            return
        os.makedirs(self.args.save_path, exist_ok=True)
        module = self.actor.module if hasattr(self.actor, "module") else self.actor
        module.model.save_pretrained(self.args.save_path)
        torch.save(module.v_head.state_dict(), os.path.join(self.args.save_path, "value_head.pt"))
        self.tokenizer.save_pretrained(self.args.save_path)

    @property
    def train_iter(self):
        return self.trainer.train_iter


class RLFTTrainer:
    def __init__(self, cfg, strategy, policy_model: RLFTPolicyModel, reference_model: Optional[RLFTReferenceModel]):
        self.cfg = cfg
        self.strategy = strategy
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.kl_ctl = FixedKLController(float(getattr(cfg, "rft_kl_coef", 0.0)))

    def train_batch(self, data, collect_env_steps: int):
        (
            input_ids,
            attention_mask,
            action_mask,
            advantages,
            rollout_lp,
            returns,
            old_values,
            log_status,
        ) = data
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "action_mask": action_mask,
            "advantages": advantages,
            "old_action_log_probs": rollout_lp,
            "returns": returns,
            "old_values": old_values,
            "log_status": log_status,
        }
        if self.reference_model is not None:
            batch["ref_action_log_probs"] = self.reference_model.forward(input_ids, action_mask, attention_mask)
        else:
            batch["ref_action_log_probs"] = None
        return self.policy_model.fit(batch, self.kl_ctl)
