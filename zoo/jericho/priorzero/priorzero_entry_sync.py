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
import deepspeed

from priorzero_config import get_priorzero_config, get_priorzero_debug_config
from priorzero_collector import PriorZeroCollector
from priorzero_evaluator import PriorZeroEvaluator
from priorzero_policy import *
from lzero.mcts.buffer.game_buffer_priorzero import PriorZeroGameBufferOptimized
from lzero.entry.utils import calculate_update_per_collect

def prepare_unizero(rank, cfg, create_cfg, llm_cfg, seed, data_processor=None):
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    
    policy = create_policy( cfg.policy, enable_field=['learn', 'collect', 'eval'], exp_name=cfg.exp_name)
    logger.info(f"[Rank {rank}]  Policy created")

    os.makedirs(f'./{cfg.exp_name}/log/', exist_ok=True)
    tb_logger = SummaryWriter(os.path.join(f'./{cfg.exp_name}/log/', 'serial')) if get_rank() == 0 else None
    logger.info(f"[Rank {rank}] TensorBoard logger: ./{cfg.exp_name}/log/")
    
    learner = BaseLearner(
        cfg.policy.learn.learner,
        policy.learn_mode,
        tb_logger,
        exp_name=cfg.exp_name
    )
    logger.info(f"[Rank {rank}] BaseLearner created")

    
    replay_buffer = PriorZeroGameBufferOptimized(cfg.policy)
    logger.info(f"[Rank {rank}] PriorZero replay buffer created (with game_segments support)")

    # Create collector
    collector = PriorZeroCollector(
        env=collector_env,
        policy=policy.collect_mode,
        llm_config=llm_cfg,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        data_processor=data_processor,
        policy_config=cfg.policy,
    )
    logger.info(f"[Rank {rank}] Collector created")

    # Create evaluator
    evaluator = PriorZeroEvaluator(
        eval_freq=cfg.policy.eval_freq,
        n_evaluator_episode=cfg.env.n_evaluator_episode,
        stop_value=cfg.env.stop_value,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=cfg.policy,
    )
    logger.info(f"[Rank {rank}] Evaluator created")
    learner.call_hook('before_run')

    return cfg, replay_buffer, tb_logger, policy, collector, evaluator, learner
    
def bcast_obj(world_size, obj, rank, src=0):
    if world_size <= 1:
        return obj
    lst = [obj] if rank == src else [None]
    dist.broadcast_object_list(lst, src=src)
    return lst[0]    

def train_priorzero(
    cfg: dict,
    create_cfg: dict,
    llm_cfg,
    seed: int = 0,
    max_train_iter: int = int(1e6),
    max_env_step: Optional[int] = int(1e10),
):
    rank = int(os.environ.get("RANK", "0"))
    print(f"rank={rank}")
    if rank == 0:
        cfg, replay_buffer, tb_logger, policy, collector, evaluator, learner = prepare_unizero( 
                                                                            rank=rank,
                                                                            cfg=cfg,
                                                                            create_cfg=create_cfg, 
                                                                            llm_cfg=llm_cfg, 
                                                                            seed=seed, 
                                                                            data_processor=None)
        batch_size = cfg.policy.batch_size

    from strategy.deepspeed import get_strategy, torch_dist_barrier_and_cuda_sync
    strategy = get_strategy(llm_cfg)
    strategy.print(llm_cfg)
    
    strategy.setup_distributed()   # torchrun 下：绑定 local_rank + init_distributed
    world_size = getattr(strategy, "world_size", 1)
    
    logger.info(f"[Rank {rank}] Initializing LLM Actor...")
    set_pkg_seed(seed + rank, use_cuda=True)
    
    from models.actor import PolicyModel, ReferenceModel
    if llm_cfg.rft_kl_coef > 0:
        ref_model = ReferenceModel(
            strategy=strategy,
            pretrain=llm_cfg.model_name_or_path
        )
    else:
        ref_model = None
    
    from vllm_utils.vllm_engine import create_vllm_engine
    vllm_engine = create_vllm_engine(
        tensor_parallel_size=llm_cfg.vllm_tensor_parallel_size,
        pretrain=llm_cfg.model_name_or_path,
        enable_prefix_caching=llm_cfg.enable_prefix_caching,
        max_model_len=llm_cfg.prompt_max_len + llm_cfg.generate_max_len,
        gpu_memory_utilization=llm_cfg.gpu_memory_utilization,
        vllm_enable_sleep=llm_cfg.vllm_enable_sleep,
    )

    print(f'[Rank {rank}] Vllm engine successfully created!')
    
    from priorzero_datafactory import DataProcessor
    data_processor = DataProcessor(rank=rank, 
                                   vllm_engine=vllm_engine, 
                                   strategy=strategy, 
                                   model_path=llm_cfg.model_name_or_path,
                                   exp_name=cfg.exp_name if rank == 0 else None,
                                   )
    if rank == 0:
        collector.data_processor = data_processor
    
    policy_model = PolicyModel(
        strategy=strategy,
        pretrain=llm_cfg.model_name_or_path,
        vllm_engine=vllm_engine
    )
    from priorzero_trainer import PriorZeroLLMTrainer
    trainer = PriorZeroLLMTrainer(
        cfg=llm_cfg,
        pretrain=llm_cfg.model_name_or_path,
        strategy= strategy,
        vllm_engine = vllm_engine,
        policy_model=policy_model,
        reference_model=ref_model,
        broadcast_every=llm_cfg.broadcast_every,
        exp_name=cfg.exp_name if rank == 0 else None,
        tb_logger=tb_logger if rank == 0 else None,
    )
        
    torch_dist_barrier_and_cuda_sync()
    
    while True:
        cmd = "noop"
        train_samples = None
        if rank == 0:
            if learner.train_iter > 0 and evaluator.should_eval(learner.train_iter):
                logger.info(f"\n[Rank {rank}: Iter {learner.train_iter}] Evaluating...")
                stop, reward = evaluator.eval(
                    save_ckpt_fn=learner.save_checkpoint,
                    train_iter=learner.train_iter,
                    envstep=collector.envstep
                )
                if stop:
                    cmd = "stop"
                    
            if cmd != "stop":
                if llm_cfg.vllm_enable_sleep and vllm_engine is not None:
                    vllm_engine.wake_up()
                    
                new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs={'temperature': 0.25, 'epsilon': 0.0})
                data_processor.get_llm_output_log()
                
                if llm_cfg.vllm_enable_sleep and vllm_engine is not None:
                    vllm_engine.sleep()
                
                update_per_collect = calculate_update_per_collect(cfg, new_data, world_size=1)                
                
                replay_buffer.push_game_segments(new_data)
                replay_buffer.remove_oldest_data_to_fit()
                
                num_of_transitions = replay_buffer.get_num_of_transitions() 
                new_num_of_transitions = replay_buffer.get_num_of_transitions() - replay_buffer.last_pos_in_transition
                logger.info(f"[Rank {rank}] Data collected, num_of_transitions: {num_of_transitions} transitions")
            
                if not (num_of_transitions > batch_size):
                    logger.warning(
                        f'  ⚠ Data in replay_buffer is not sufficient: '
                        f'batch_size: {batch_size}, replay_buffer: {replay_buffer}. Continue to collect...'
                    )
                    cmd = "noop" 

                logger.info(f"[Rank {rank}: World Model] [Iter {learner.train_iter}] Training...")
                for i in range(update_per_collect):
                    train_data = replay_buffer.sample(batch_size, policy)
                    train_data.append(learner.train_iter)

                    log_vars = learner.train(train_data, collector.envstep)
                    if cfg.policy.use_priority:
                        replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])
                policy.recompute_pos_emb_diff_and_clear_cache()
                
                if  new_num_of_transitions >= llm_cfg.llm_learn_num_samples:
                    priorzero_batch = replay_buffer.fetch_latest_batch(batch_size=llm_cfg.llm_learn_num_samples, policy=policy)
                    train_samples = data_processor.make_llm_train_samples(priorzero_batch)
                    cmd = "llm"

                if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
                    cmd = "stop"
        
        cmd = bcast_obj(world_size, cmd, rank, src=0)
        if cmd == "stop":
            break
        elif cmd == "llm":
            logger.info(f"[Rank {rank}] Waiting for broadcast of train_samples from Rank 0...")
            train_samples = bcast_obj(world_size, train_samples, rank, src=0) 
            logger.info(f"[Rank {rank}] Received broadcast. train_samples count: {len(train_samples[0])}. Starting LLM training...")

            trainer.train_batch(train_samples)
            torch_dist_barrier_and_cuda_sync()
            

def main():
    """
    Main entry point with argument parsing.
    """
    import argparse

    parser = argparse.ArgumentParser(description='PriorZero Training')
    parser.add_argument('--env_id', type=str, default='detective.z5', help='Jericho game ID')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--max_iter', type=int, default=int(1e6), help='Max training iterations')
    parser.add_argument('--quick_test', action='store_true', help='Use quick test config')
    parser.add_argument('--no_save', action='store_true', help='Disable checkpoint saving')
    parser.add_argument('--debug', action='store_true', help='Enable detailed debug logging (obs, action, LLM output)')

    args = parser.parse_args()
    
    args.quick_test = False
    use_cot=True
    if args.quick_test:
        logger.info("Using quick test configuration")
        main_cfg, create_cfg, llm_cfg = get_priorzero_debug_config(args.env_id, args.seed, use_cot=use_cot, exp_name=f'data_priorzero/priorzero_sync_debug_{args.env_id}_seed0')
    else:
        main_cfg, create_cfg, llm_cfg = get_priorzero_config(args.env_id, args.seed, use_cot=use_cot, exp_name=f'data_priorzero/priorzero_ppo_{args.env_id}_seed0')

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
