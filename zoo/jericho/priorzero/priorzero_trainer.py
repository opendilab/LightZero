from __future__ import annotations
import os
import copy
import json

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import ray
import numpy as np
from transformers import AutoTokenizer

from openrlhf.trainer.ppo_utils import  FixedKLController 

import ray
import torch

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
        broadcast_every: int = 1,   # 每 N step 同步一次权重到 vLLM
        exp_name: str = None,
        tb_logger = None,
        instance_name: str = "llm_ppo"
    ):
        self.cfg = cfg
        self.pretrain = pretrain
        self.strategy = strategy
        self.args = getattr(strategy, "args", None)

        self.policy_model = policy_model
        self.reference_model = reference_model
        self.vllm_engine = vllm_engine

        self.broadcast_every = max(int(broadcast_every), 1)
        self.global_step = 0

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
    
    def train_batch(self, data) -> Dict[str, float]:
        if data is None:
            return {}
        input_ids, attention_mask, action_mask, gt, old_lp = data
        assert len(input_ids) == len(attention_mask) == len(action_mask) == len(gt) == len(old_lp)
        
        bsz = input_ids.size(0)
        per_rank = bsz // self.world_size
        start = self.rank * per_rank
        end = (self.rank + 1) * per_rank if self.rank != self.world_size - 1 else bsz  

        batch = {
            "input_ids": input_ids[start:end],
            "attention_mask": attention_mask[start:end],
            "action_mask": action_mask[start:end],
            "advantages": gt[start:end],     
            "old_action_logprob": old_lp[start:end],     
        }
        if self.reference_model is not None:
            base_action_log_probs = self.reference_model.forward(
                sequences = batch['input_ids'],
                action_mask = batch['action_mask'],
                attention_mask=batch['attention_mask'],
                logits_to_keep=batch['action_mask'].size(1) + 1
            )
            batch["ref_action_log_probs"] = base_action_log_probs
        else:
            batch["ref_action_log_probs"] = None
        status = self.policy_model.fit(batch, self.kl_ctl)
        
        self.global_step += 1
        
        if self.vllm_engine is not None and (self.global_step % self.broadcast_every == 0):
            self._broadcast_to_vllm()
        
        if self._tb_logger is not None and self.strategy.is_rank_0():
            for k, v in status.items():
                self._tb_logger.add_scalar(f"learner_llm_iter/{k}", float(v), self.global_step)
        
        # if self.strategy.args.deepspeed_enable_sleep:
        #     self.policy_model.reload_states()
        # if self.strategy.args.deepspeed_enable_sleep:
        #     self.policy_model.offload_states()

    def get_state(self) -> Dict[str, Any]:
        kl_val = float(self.kl_ctl.value) if hasattr(self.kl_ctl, "value") else float(self.init_kl_coef)
        return {"global_step": self.global_step, "kl_coef": kl_val}

    def _broadcast_to_vllm(self):
        if self.strategy.args.vllm_enable_sleep:
            self.vllm_engine.wake_up()

        self.policy_model.broadcast_to_vllm()

        if self.strategy.args.vllm_enable_sleep:
            self.vllm_engine.sleep()