import asyncio
import os
import sys
import re
from pathlib import Path
from functools import partial
from typing import Optional, List, Dict, Any, Callable, Awaitable, Tuple
import time
import json
from easydict import EasyDict
from dataclasses import dataclass, field
from omegaconf.listconfig import ListConfig


from ensure_local_lightzero import ensure_local_lightzero
ensure_local_lightzero()

import torch
import numpy as np
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed, get_rank
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter
from loguru import logger

from transformers import AutoTokenizer
import ray
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

# PriorZero imports
from priorzero_config import get_priorzero_config, get_priorzero_debug_config, HybridTrainingConfig, ORZConfig
from priorzero_collector import PriorZeroCollector
from priorzero_evaluator import PriorZeroEvaluator
import priorzero_policy  
from lzero.mcts.buffer.game_buffer_priorzero import PriorZeroGameBufferOptimized
from priorzero_orz_trainer import TempExp, JerichoPromptDataset, GameSegmentToORZAdapter, JerichoRewardTrainer
from orz.ppo.utils import get_strategy


async def train_priorzero_orz_entry(
    cfg: dict,
    create_cfg: dict,
    hybrid_cfg: HybridTrainingConfig,
    seed: int = 0,
    max_train_iter: int = 10000,
    max_env_step: Optional[int] = int(1e10),
):
    """
    Main hybrid training function with complete ORZ integration.
    """
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)

    logger.info("Creating vLLM engine...")
    tensor_parallel = cfg.policy.llm_policy_cfg.vllm_tensor_parallel_size
    distributed_backend = "ray" if tensor_parallel > 1 else None

    # gpu_mem_util = cfg.policy.llm_policy_cfg.gpu_memory_utilization
    gpu_mem_util = 0.05
    
    use_v1_env = os.environ.get('VLLM_USE_V1', None)
    if use_v1_env is None:
        os.environ['VLLM_USE_V1'] = '0'
        logger.info("✓ Using vLLM V0 engine for stability")

    
    engine_args = AsyncEngineArgs(
        model=cfg.policy.llm_policy_cfg.pretrain_llm_path,
        tensor_parallel_size=tensor_parallel,
        gpu_memory_utilization=gpu_mem_util,
        distributed_executor_backend=distributed_backend,
        trust_remote_code=True,
        enable_prefix_caching=False,
        enforce_eager=False,
    )
    vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    logger.info(f"✓ vLLM Engine created (backend: {distributed_backend or 'default'})")

    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)

    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=True)
    logger.info(f"✓ Environments created and seeded (seed={seed})")

    policy = create_policy(cfg.policy, enable_field=['learn', 'collect', 'eval'], exp_name=cfg.exp_name)
    logger.info("✓ Policy created")

    os.makedirs(f'./{cfg.exp_name}/log/', exist_ok=True)
    tb_logger = SummaryWriter(os.path.join(f'./{cfg.exp_name}/log/', 'serial')) if get_rank() == 0 else None
    logger.info(f"✓ TensorBoard logger: ./{cfg.exp_name}/log/")

    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    replay_buffer = PriorZeroGameBufferOptimized(cfg.policy)

    collector = PriorZeroCollector(
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        vllm_engine=vllm_engine,  
        policy_config=cfg.policy,
    )

    evaluator = PriorZeroEvaluator(
        eval_freq=cfg.policy.eval_freq,
        n_evaluator_episode=cfg.env.n_evaluator_episode,
        stop_value=cfg.env.stop_value,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        vllm_engine=vllm_engine,  
        policy_config=cfg.policy,
    )

    learner.call_hook('before_run')
    
    ### ORZ 准备阶段
    orz_adapter = GameSegmentToORZAdapter()

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
        logger.info("✓ Ray initialized")

    orz_tokenizer = AutoTokenizer.from_pretrained(
        cfg.policy.llm_policy_cfg.pretrain_llm_path,
        trust_remote_code=True
    )
    if orz_tokenizer.pad_token is None:
        orz_tokenizer.pad_token = orz_tokenizer.eos_token

    orz_strategy = get_strategy(EasyDict({
        'zero_stage': 2,
        'bf16': True,
        'gradient_checkpointing': True,
    }))
    orz_cfg = ORZConfig()
    logger.info("✓ ORZ trainer components ready")


    while learner.train_iter < max_train_iter and collector.envstep < max_env_step:
        current_iter = learner.train_iter

        if current_iter > 0 and evaluator.should_eval(current_iter):
            stop, reward = await evaluator.eval(
                save_ckpt_fn=learner.save_checkpoint,
                train_iter=current_iter,
                envstep=collector.envstep
            )
            if stop:
                break

        collect_kwargs = {'temperature': 0.25, 'epsilon': 0.0}
        new_data = await collector.collect(
            train_iter=current_iter,
            policy_kwargs=collect_kwargs
        )
        from lzero.entry.utils import calculate_update_per_collect
        update_per_collect = 1
        # update_per_collect = calculate_update_per_collect(cfg, new_data, world_size=1)

        replay_buffer.push_game_segments(new_data)
        replay_buffer.remove_oldest_data_to_fit()
        buffer_size = replay_buffer.get_num_of_transitions() if hasattr(replay_buffer, 'get_num_of_transitions') else 0
        logger.info(f"  ✓ Data collected, buffer size: {buffer_size} transitions")

        if current_iter % hybrid_cfg.wm_train_freq == 0:
            if replay_buffer.get_num_of_transitions() >= cfg.policy.batch_size:
                for _ in range(update_per_collect):
                    train_data = replay_buffer.sample(cfg.policy.batch_size, policy)
                    train_data.append(learner.train_iter)
                    
                    log_dict = learner.train(train_data, collector.envstep)
            else:
                logger.info(f"Skipping training - not enough data yet")

        if current_iter % hybrid_cfg.llm_train_freq == 0:
            logger.info(f"[Iter {current_iter}] Training LLM with ORZ...")
            training_data = orz_adapter.extract_training_data(new_data)
            num_samples = len(training_data['states'])

            logger.info(f"  Extracted {num_samples} training samples for ORZ")
            if num_samples > 0:
                dialogues = orz_adapter.convert_segments_to_prompts(
                    new_data,
                    orz_tokenizer
                )
                orz_dataset = JerichoPromptDataset(
                    dialogues,
                    orz_tokenizer,
                    orz_cfg.prompt_max_len,
                    orz_strategy,
                    pretrain_mode=False,
                    num_processors=1
                )
                temp_exp = TempExp()
                vllm_engines = temp_exp.create_inference_engine()
                logger.info(f"  ✓ Created {len(vllm_engines)} vLLM engines")

                colocate_pg = temp_exp.get_colocate_pg if orz_cfg.colocate_all else None

                orz_trainer = JerichoRewardTrainer(
                    cfg=orz_cfg,
                    strategy=orz_strategy,
                    tokenizer=orz_tokenizer,
                    train_dataset=orz_dataset,
                    eval_dataset=None,  
                    vllm_engines=vllm_engines,
                    colocate_pg=colocate_pg
                )
                logger.info("  ✓ ORZ RayPPOTrainer initialized")

                logger.info(f"  Running ORZ PPO training (episode {current_iter // hybrid_cfg.llm_train_freq})...")
                await orz_trainer.fit_episode()
                logger.info(f"  ✓ ORZ training completed for iteration {current_iter}")

            else:
                logger.warning("  No training samples extracted from game_segments")




async def main():
    hybrid_cfg = HybridTrainingConfig()
    
    
    await train_priorzero_orz_entry(
        cfg=hybrid_cfg.priorzero_cfg,
        create_cfg=hybrid_cfg.priorzero_create_cfg,
        hybrid_cfg=hybrid_cfg,
        seed=hybrid_cfg.priorzero_cfg.seed,
        max_train_iter=10000,
    )


if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    asyncio.run(main())
