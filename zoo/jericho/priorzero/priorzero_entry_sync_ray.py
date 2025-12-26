import asyncio
import os
import sys
from functools import partial
from pathlib import Path
from typing import Tuple, Optional

import torch
import wandb
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed, get_rank, get_world_size
from ding.worker import create_buffer, BaseLearner
from tensorboardX import SummaryWriter
from loguru import logger
from lzero.config.utils import lz_to_ddp_config

from priorzero_config import get_priorzero_config, get_priorzero_debug_config
from priorzero_collector import PriorZeroCollector
from priorzero_evaluator import PriorZeroEvaluator
from priorzero_policy import *
from lzero.mcts.buffer.game_buffer_priorzero import PriorZeroGameBufferOptimized
from lzero.entry.utils import calculate_update_per_collect
from priorzero_trainer import PriorZeroLLMTrainer


def train_priorzero(
    cfg: dict,
    create_cfg: dict,
    llm_cfg,
    seed: int = 0,
    max_train_iter: int = int(1e6),
    max_env_step: Optional[int] = int(1e10),
):
    """
    [PRIORZERO-MODIFIED]
    Main async training function for PriorZero.

    Args:
        cfg: Main configuration dictionary
        create_cfg: Creation configuration for DI-engine components
        seed: Random seed
        max_train_iter: Maximum training iterations
    """
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)

    logger.info("Creating environments...")
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager( cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager( cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=True)

    logger.info("Creating policy, buffer, and components...")
    policy = create_policy( cfg.policy, enable_field=['learn', 'collect', 'eval'], exp_name=cfg.exp_name)
    logger.info("✓ Policy created")

    os.makedirs(f'./{cfg.exp_name}/log/', exist_ok=True)
    tb_logger = SummaryWriter(os.path.join(f'./{cfg.exp_name}/log/', 'serial')) if get_rank() == 0 else None
    logger.info(f"✓ TensorBoard logger: ./{cfg.exp_name}/log/")

    llm_prior_generator = None
    if llm_cfg.enable_llm:
        import ray
        from ray.util.placement_group import placement_group
        if not ray.is_initialized():
            ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "false", "NCCL_DEBUG": "WARN", "RAY_DEBUG": "1"}})
            # ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "false", "NCCL_DEBUG": "WARN"}})
            # ray.init(local_model=True)
        from openrlhf.utils import get_strategy
        strategy = get_strategy(llm_cfg)
        strategy.print(llm_cfg)
        
        pg = None
        # 分配 reference model的资源
        if llm_cfg.rft_kl_coef > 0:
            bundles = [{"GPU": 1, "CPU": 1} for _ in range(llm_cfg.policy_model_num_gpus)]
            pg = placement_group(bundles, strategy="PACK")
            ray.get(pg.ready())
        

        vllm_engine = None
        if llm_cfg.vllm_num_engines > 0:
            from utils.vllm_engine import create_vllm_engines
            vllm_engines = create_vllm_engines(
                num_engines=llm_cfg.vllm_num_engines,
                tensor_parallel_size=llm_cfg.vllm_tensor_parallel_size,
                pretrain=llm_cfg.model_name_or_path,
                seed=llm_cfg.seed,
                full_determinism=False,
                enable_prefix_caching=llm_cfg.enable_prefix_caching,
                enforce_eager=False,
                max_model_len=llm_cfg.prompt_max_len + llm_cfg.generate_max_len,
                gpu_memory_utilization=llm_cfg.gpu_memory_utilization,
                shared_pg=pg,
                vllm_enable_sleep=llm_cfg.vllm_enable_sleep,
            )
        from openrlhf.trainer.ray.launcher import RayActorGroup
        from utils.ray.model import ReferenceModel, PolicyModel
        actor_model = RayActorGroup(
            num_nodes=1,
            num_gpus_per_node=llm_cfg.policy_model_num_gpus,
            ray_actor_type=PolicyModel,
            pg=pg,
            num_gpus_per_actor=0.3 if pg else 1,
            duplicate_actors=llm_cfg.ring_attn_size * llm_cfg.ds_tensor_parallel_size,
        )
        if llm_cfg.rft_kl_coef > 0:
            ref_model = RayActorGroup(
                num_nodes=1,
                num_gpus_per_node=llm_cfg.reference_model_num_gpus,
                ray_actor_type=ReferenceModel,
                pg=pg,
                num_gpus_per_actor=0.3 if pg else 1,
                duplicate_actors=llm_cfg.ring_attn_size * llm_cfg.ds_tensor_parallel_size,
            )
        else:
            ref_model = None

        # trainer = PriorZeroLLMTrainer.remote(
        #     cfg=llm_cfg,
        #     pretrain=llm_cfg.model_name_or_path,
        #     strategy= strategy,
        #     actor_model_group=actor_model,
        #     reference_model_group=ref_model,
        #     vllm_engines=vllm_engines,
        #     broadcast_every=llm_cfg.broadcast_every
        # )
        trainer = PriorZeroLLMTrainer(
            cfg=llm_cfg,
            pretrain=llm_cfg.model_name_or_path,
            strategy= strategy,
            actor_model_group=actor_model,
            reference_model_group=ref_model,
            vllm_engines=vllm_engines,
            broadcast_every=llm_cfg.broadcast_every
        )
        refs = []
        if ref_model is not None:
            refs.extend(ref_model.async_init_model_from_pretrained(strategy, llm_cfg.model_name_or_path))
        refs.extend(actor_model.async_init_model_from_pretrained(strategy, llm_cfg.model_name_or_path, vllm_engines))
        ray.get(refs)
        
        from jericho.LightZero.zoo.jericho.priorzero.utils.vllm.generator import SamplesGenerator
        from priorzero_trainer import get_tokenizer
        llm_prior_generator = SamplesGenerator(vllm_engines, strategy, get_tokenizer(llm_cfg.model_name_or_path))

    learner = BaseLearner(
        cfg.policy.learn.learner,
        policy.learn_mode,
        tb_logger,
        exp_name=cfg.exp_name
    )
    logger.info("✓ BaseLearner created")

    
    replay_buffer = PriorZeroGameBufferOptimized(cfg.policy)
    logger.info("✓ PriorZero replay buffer created (with game_segments support)")

    # Create collector
    collector = PriorZeroCollector(
        env=collector_env,
        policy=policy.collect_mode,
        llm_config=llm_cfg,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        llm_prior_generator=llm_prior_generator if llm_cfg.enable_llm else None,
        policy_config=cfg.policy,
    )
    logger.info("✓ Collector created")

    # Create evaluator
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
    logger.info("✓ Evaluator created")
    learner.call_hook('before_run')

    buffer_reanalyze_count = 0
    train_epoch = 0
    reanalyze_batch_size = cfg.policy.reanalyze_batch_size
    batch_size = cfg.policy.batch_size
    
    if cfg.policy.multi_gpu:
        world_size = get_world_size()
        rank = get_rank()
    else:
        world_size = 1
        rank = 0
    
    while True:
        if learner.train_iter > 0 and evaluator.should_eval(learner.train_iter):
            logger.info(f"\n[Iter {learner.train_iter}] Evaluating...")
            stop, reward = evaluator.eval(
                save_ckpt_fn=learner.save_checkpoint,
                train_iter=learner.train_iter,
                envstep=collector.envstep
            )
            if stop:
                break

        collect_kwargs = {
            'temperature': 0.25,
            'epsilon': 0.0
        }

        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        update_per_collect = calculate_update_per_collect(cfg, new_data, world_size=world_size)

        replay_buffer.push_game_segments(new_data)
        replay_buffer.remove_oldest_data_to_fit()
        num_of_transitions = replay_buffer.get_num_of_transitions() 
        new_num_of_transitions = replay_buffer.get_num_of_transitions() - replay_buffer.last_pos_in_transition
        logger.info(f"  ✓ Data collected, num_of_transitions: {num_of_transitions} transitions")

        if cfg.policy.buffer_reanalyze_freq >= 1:
            reanalyze_interval = update_per_collect // cfg.policy.buffer_reanalyze_freq
        else:
            if train_epoch > 0 and train_epoch % int(1/cfg.policy.buffer_reanalyze_freq) == 0:
                logger.info(f"[Reanalyze] Starting buffer reanalysis...")
                replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                buffer_reanalyze_count += 1
                logger.info(f"  ✓ Buffer reanalyze count: {buffer_reanalyze_count}")

        if collector.envstep <= cfg.policy.train_start_after_envsteps:
            continue
        
        if cfg.policy.sample_type == 'episode':
            data_sufficient = num_of_transitions > batch_size
        else:
            data_sufficient = num_of_transitions > batch_size

        if not data_sufficient:
            logger.warning(
                f'  ⚠ Data in replay_buffer is not sufficient: '
                f'batch_size: {batch_size}, replay_buffer: {replay_buffer}. Continue to collect...'
            )
            continue

        logger.info(f"[Iter {learner.train_iter}] Training...")
        for i in range(update_per_collect):
            train_data = replay_buffer.sample(batch_size, policy)
            train_data.append(learner.train_iter)

            log_vars = learner.train(train_data, collector.envstep)
            if cfg.policy.use_priority:
                replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])

        if llm_cfg.enable_llm and new_num_of_transitions >= llm_cfg.llm_learn_num_samples:
            priorzero_batch = replay_buffer.fetch_latest_batch(batch_size=llm_cfg.llm_learn_num_samples, policy=policy)
            # ray.get(trainer.train_batch.remote(priorzero_batch))
            trainer.train_batch(priorzero_batch)

        train_epoch += 1
        policy.recompute_pos_emb_diff_and_clear_cache()

        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            logger.info("Stopping condition met, training ends!")
            break


    return policy


def main():
    """
    Main entry point with argument parsing.
    """
    import argparse

    parser = argparse.ArgumentParser(description='PriorZero Training')
    parser.add_argument('--env_id', type=str, default='zork1.z5', help='Jericho game ID')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--max_iter', type=int, default=int(1e6), help='Max training iterations')
    parser.add_argument('--quick_test', action='store_true', help='Use quick test config')
    parser.add_argument('--no_save', action='store_true', help='Disable checkpoint saving')
    parser.add_argument('--debug', action='store_true', help='Enable detailed debug logging (obs, action, LLM output)')

    args = parser.parse_args()
    
    args.quick_test = True
    use_cot=True
    if args.quick_test:
        logger.info("Using quick test configuration")
        main_cfg, create_cfg, llm_cfg = get_priorzero_debug_config(args.env_id, args.seed, use_cot=use_cot, exp_name=f'data_priorzero/priorzero_sync_debug_{args.env_id}_seed0')
    else:
        main_cfg, create_cfg, llm_cfg = get_priorzero_config(args.env_id, args.seed, use_cot=use_cot, exp_name=f'data_priorzero/priorzero_sync_rft_reinforce++_{args.env_id}_seed0')

    train_priorzero(
        main_cfg,
        create_cfg,
        llm_cfg,
        seed=args.seed,
        max_train_iter=args.max_iter,
    )


if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()
