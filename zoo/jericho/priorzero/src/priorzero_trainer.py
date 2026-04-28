from __future__ import annotations
import hashlib
import os
import copy
import json
import logging

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import ray
import numpy as np
from transformers import AutoTokenizer
import torch.distributed as dist

import ray
import torch

import numpy as np


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


def get_tokenizer(pretrain: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        pretrain, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

class PriorZeroLLMTrainer:

    def __init__(
        self,
        cfg, 
        pretrain: str,
        strategy,  
        vllm_engine,
        policy_model,  # RayActorGroup(PolicyModelActor)
        reference_model=None,  # RayActorGroup(ReferenceModelActor) or None
        exp_name: str = None,
        tb_logger = None,
        instance_name: str = "llm_ppo",
        llm_save_freq: int = 1000,
    ):
        self.cfg = cfg
        self.pretrain = pretrain
        self.strategy = strategy
        self.args = getattr(strategy, "args", None)

        self.policy_model = policy_model
        self.reference_model = reference_model
        self.vllm_engine = vllm_engine
        self.global_step = 0
        self.llm_save_freq = llm_save_freq

        self.tokenizer = get_tokenizer(self.pretrain)

        self.init_kl_coef = float(getattr(cfg, "rft_kl_coef", 0.0))

        self.kl_ctl = FixedKLController(self.init_kl_coef)
        self.rank = self.strategy.get_rank()
        self.world_size = self.strategy.world_size
        
        if tb_logger is not None:
            from ding.utils import build_logger
            self._logger, _ = build_logger(
                path=f'./{exp_name}/log/{instance_name}', name=instance_name, need_tb=False
            )
            self._tb_logger = tb_logger
        else:
            self._logger = None
            self._tb_logger = None
    
    def train_batch(self, data, collect_env_steps) -> Dict[str, float]:
        if data is None:
            return {}
        input_ids, attention_mask, action_mask, advantage, rollout_lp, log_status = data
        assert len(input_ids) == len(attention_mask) == len(action_mask) == len(advantage) == len(rollout_lp) == len(log_status)
        batch_input_stats = self._collect_input_ids_stats(input_ids)
        
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "action_mask": action_mask,
            "advantages": advantage,     
            "rollout_action_logprob": rollout_lp,
            "log_status": log_status,     
        }
        if self.reference_model is not None:
            base_action_log_probs = self.reference_model.forward(
                sequences = batch['input_ids'],
                action_mask = batch['action_mask'],
                attention_mask=batch['attention_mask'],
            )
            batch["ref_action_log_probs"] = base_action_log_probs
        else:
            batch["ref_action_log_probs"] = None
        
        if self.strategy.args.deepspeed_enable_sleep:
            self.policy_model.reload_states()
        
        old_action_log_probs = self.policy_model.forward(
            sequences = batch['input_ids'],
            action_mask = batch['action_mask'],
            attention_mask=batch['attention_mask'],
        )
        batch["old_action_log_probs"] = old_action_log_probs
            
        status = self.policy_model.fit(batch, self.kl_ctl)
        
        if self.vllm_engine is not None:
            self._broadcast_to_vllm()
        
        if self.strategy.args.deepspeed_enable_sleep:
            self.policy_model.offload_states()

        for tmp_dict in status:
            tmp_dict.update(batch_input_stats)
        
        if self._tb_logger is not None and self.strategy.is_rank_0():
            logging.getLogger("priorzero.train").info(
                f"[LLM] samples={int(batch_input_stats['input_ids_global_sample_count'])} "
                f"unique={int(batch_input_stats['input_ids_global_unique_count'])} "
                f"ratio={float(batch_input_stats['input_ids_global_unique_ratio']):.4f}"
            )
            for tmp_dict in status:
                for k, v in tmp_dict.items():
                    if k == 'iter':
                        continue
                    self._tb_logger.add_scalar(f"learner_llm_iter/{k}", float(v), int(tmp_dict['iter']))
                    self._tb_logger.add_scalar(f"learner_llm_envstep/{k}", float(v), int(collect_env_steps))
                    self.global_step = max(self.global_step, int(tmp_dict['iter']))
        
        self._sync_global_step_from_rank0()
        
        if self.global_step > 0 and self.global_step % self.llm_save_freq == 0:
            self.policy_model.save_model()

    def get_state(self) -> Dict[str, Any]:
        kl_val = float(self.kl_ctl.value) if hasattr(self.kl_ctl, "value") else float(self.init_kl_coef)
        return {"global_step": self.global_step, "kl_coef": kl_val}

    def _sync_global_step_from_rank0(self):
        if self.world_size <= 1:
            return 
        lst = [self.global_step] if self.rank == 0 else [None]
        dist.broadcast_object_list(lst, src=0)
        self.global_step = int(lst[0])

    def _collect_input_ids_stats(self, input_ids: torch.Tensor) -> Dict[str, float]:
        local_hashes = self._hash_input_rows(input_ids)
        local_sample_count = len(local_hashes)
        local_unique_count = len(set(local_hashes))

        global_hashes = local_hashes
        if self.world_size > 1:
            gathered_hashes = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered_hashes, local_hashes)
            global_hashes = [item for rank_hashes in gathered_hashes for item in rank_hashes]

        global_sample_count = len(global_hashes)
        global_unique_count = len(set(global_hashes))
        global_duplicate_count = global_sample_count - global_unique_count
        global_unique_ratio = global_unique_count / global_sample_count if global_sample_count > 0 else 0.0

        return {
            "input_ids_local_sample_count": float(local_sample_count),
            "input_ids_local_unique_count": float(local_unique_count),
            "input_ids_global_sample_count": float(global_sample_count),
            "input_ids_global_unique_count": float(global_unique_count),
            "input_ids_global_duplicate_count": float(global_duplicate_count),
            "input_ids_global_unique_ratio": float(global_unique_ratio),
        }

    def _hash_input_rows(self, input_ids: torch.Tensor) -> List[str]:
        input_ids_cpu = input_ids.detach().to("cpu")
        return [
            hashlib.blake2b(row.numpy().tobytes(), digest_size=16).hexdigest()
            for row in input_ids_cpu
        ]
    
    def _broadcast_to_vllm(self):
        if self.strategy.args.vllm_enable_sleep:
            self.vllm_engine.wake_up()
        
        logging.getLogger("priorzero.train").info("[LLM] vLLM weight sync start")
        self.policy_model.broadcast_to_vllm()
        logging.getLogger("priorzero.train").info("[LLM] vLLM weight sync done")

        if self.strategy.args.vllm_enable_sleep:
            self.vllm_engine.sleep()