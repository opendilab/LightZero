import sys
import os
from pathlib import Path

import asyncio
import os
import sys
from functools import partial
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.distributed as dist
import wandb

from ding.config import compile_config, save_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed, get_rank, get_world_size
from ding.worker import create_buffer, BaseLearner
from tensorboardX import SummaryWriter
from loguru import logger
import deepspeed

from priorzero_config import (
    get_priorzero_config,
    get_priorzero_debug_config,
    get_available_models,
)
from priorzero_collector import PriorZeroCollector
from priorzero_evaluator import PriorZeroEvaluator
from priorzero_policy import *
from lzero.mcts.buffer.game_buffer_priorzero import PriorZeroGameBufferOptimized
from utils import dump_dataclass_cfg_py

from lzero.entry.utils import calculate_update_per_collect

def prepare_unizero(rank, cfg, create_cfg, llm_cfg, seed):
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    
    policy = create_policy( cfg.policy, enable_field=['learn', 'collect', 'eval'], exp_name=cfg.exp_name, llm_cfg=llm_cfg)
    if cfg.policy.model_path is not None:
        logging.info(f"[Rank {rank}] Loading pretrained model from {cfg.policy.model_path}...")
        policy.learn_mode.load_state_dict(torch.load(cfg.policy.model_path, map_location=cfg.policy.device))
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
        policy_config=cfg.policy,
    )
    logger.info(f"[Rank {rank}] Collector created")

    # Create evaluator
    evaluator = PriorZeroEvaluator(
        n_evaluator_episode=cfg.env.n_evaluator_episode,
        stop_value=cfg.env.stop_value,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=cfg.policy,
        llm_config=llm_cfg,
    )
    logger.info(f"[Rank {rank}] Evaluator created")
    learner.call_hook('before_run')

    return cfg, replay_buffer, tb_logger, policy, collector, evaluator, learner

def all_gather_cmd(world_size, obj) -> List:
    if world_size <= 1:
        return [obj]
    lst = [None] * dist.get_world_size()
    dist.all_gather_object(lst, obj)
    return lst

def train_priorzero(
    cfg: dict,
    create_cfg: dict,
    llm_cfg,
    seed: int = 0,
    max_train_iter: int = int(1e6),
    max_env_step: Optional[int] = int(1e10),
    enable_profile: bool = False
):
    rank = int(os.environ.get("RANK", "0"))
    print(f"DEBUG: Is dist initialized at start? {dist.is_initialized()}")
    if dist.is_initialized():
        print(f"DEBUG: Backend is {dist.get_backend()}")
    from strategy.deepspeed import get_strategy, torch_dist_barrier_and_cuda_sync
    strategy = get_strategy(llm_cfg)
    strategy.print(llm_cfg)
    
    strategy.setup_distributed()   # torchrun 下：绑定 local_rank + init_distributed
    world_size = getattr(strategy, "world_size", 1)
    
    
    cfg, replay_buffer, tb_logger, policy, collector, evaluator, learner = prepare_unizero( 
                                                                        rank=rank,
                                                                        cfg=cfg,
                                                                        create_cfg=create_cfg, 
                                                                        llm_cfg=llm_cfg, 
                                                                        seed=seed)
    batch_size = cfg.policy.batch_size
    logger.info(f"[Rank {rank}] World Model components initialized")
    if rank == 0:
        dump_dataclass_cfg_py(llm_cfg, path=f"{cfg.exp_name}/llm_cfg.py")
        llm_cfg.save_path = f'./{cfg.exp_name}/llm_ckpt/'

    from utils import Profiler
    prof = Profiler(log_interval=10, stats_file=f'./{cfg.exp_name}/log/profiler.txt', enable_profile=enable_profile)

    
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
                                   world_size=world_size,
                                   vllm_engine=vllm_engine, 
                                   strategy=strategy, 
                                   model_path=llm_cfg.model_name_or_path,
                                   exp_name=cfg.exp_name if rank == 0 else None,
                                )
    # 在collector中初始化data_processor 和prof对象
    collector.data_processor = data_processor
    collector.prof = prof
    evaluator.data_processor = data_processor
    
    policy_model = PolicyModel(
        strategy=strategy,
        pretrain=llm_cfg.model_name_or_path,
        vllm_engine=vllm_engine,
        max_steps=llm_cfg.max_steps
    )
    from priorzero_trainer import PriorZeroLLMTrainer
    trainer = PriorZeroLLMTrainer(
        cfg=llm_cfg,
        pretrain=llm_cfg.model_name_or_path,
        strategy= strategy,
        vllm_engine = vllm_engine,
        policy_model=policy_model,
        reference_model=ref_model,
        exp_name=cfg.exp_name if rank == 0 else None,
        tb_logger=tb_logger if rank == 0 else None,
        llm_save_freq=llm_cfg.llm_save_freq
    )
        
    torch_dist_barrier_and_cuda_sync()
    train_schedule = llm_cfg.train_schedule
    train_alternate = train_schedule["alternate"]
    current_phase = None
    llm_collect_mode = None
    if train_alternate:
        current_phase = train_schedule["start_phase"]
        last_wm_train_iter = 0
        last_llm_train_iter = 0
        llm_collect_mode = train_schedule["llm_collect_mode"]

    while True:
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break
        
        # 1.评估阶段
        if learner.train_iter != 0 and evaluator.should_eval(wm_train_iter=learner.train_iter, llm_train_iter=policy_model.train_iter, phase=current_phase):
            logger.info(f"[Evaluator][Rank {rank}: Iter {learner.train_iter}] Evaluating...")
            if llm_cfg.vllm_enable_sleep and vllm_engine is not None:
                vllm_engine.wake_up()
            evaluator.eval(wm_train_iter=learner.train_iter, llm_train_iter=policy_model.train_iter, phase=current_phase)
            if llm_cfg.vllm_enable_sleep and vllm_engine is not None:
                vllm_engine.sleep()
        
        # 2.数据收集阶段         
        if not train_alternate or (train_alternate and current_phase == "wm") or (train_alternate and current_phase == "llm" and llm_collect_mode != "no_collect"):
            if llm_cfg.vllm_enable_sleep and vllm_engine is not None:
                vllm_engine.wake_up()      
            
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs={'temperature': 0.25, 'epsilon': 0.0}, phase=current_phase)
            data_processor.get_llm_output_log(wm_train_iter=learner.train_iter, llm_train_iter=policy_model.train_iter)
            
            if llm_cfg.vllm_enable_sleep and vllm_engine is not None:
                vllm_engine.sleep()
            
            replay_buffer.push_game_segments(new_data)
            replay_buffer.remove_oldest_data_to_fit()
            
        num_of_transitions = replay_buffer.get_num_of_transitions() 
        torch_dist_barrier_and_cuda_sync()   
              
        # 3.world model训练阶段
        if llm_cfg.enable_world_model and (not train_alternate or (train_alternate and current_phase == "wm")):
            if not (num_of_transitions > batch_size):
                logger.warning(f'[WM Training] Data in replay_buffer is not sufficient: batch_size: {batch_size}, replay_buffer: {replay_buffer}. Continue to collect...')
                cmd = 0
            else:
                cmd = 1
            if min(all_gather_cmd(world_size=world_size, obj=cmd)) == 0:
                continue
            
            update_per_collect = calculate_update_per_collect(cfg, new_data, world_size=world_size)
            logger.info(f"[WM Training] Rank {rank} | Iter {learner.train_iter} | Updates: {update_per_collect}")
            
            for i in range(update_per_collect):
                with prof.block("train_world_model", rank=rank):
                    train_data = replay_buffer.sample(batch_size, policy)
                    train_data.append(learner.train_iter)

                    log_vars = learner.train(train_data, collector.envstep)
                    if cfg.policy.use_priority:
                        replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])
            policy.recompute_pos_emb_diff_and_clear_cache()
            if llm_cfg.enable_rft and train_alternate and learner.train_iter - last_wm_train_iter >= train_schedule["wm_update_iters"]:
                current_phase = "llm"
                last_wm_train_iter = learner.train_iter
                if llm_collect_mode != "no_collect":
                    replay_buffer.mark_latest_transitions_consumed()
                print(f"[WM Training][Rank {rank}] Switching to LLM training phase at wm iter: {learner.train_iter}")
                continue
        
        # 4. llm 训练阶段
        if llm_cfg.enable_rft and (not train_alternate or (train_alternate and current_phase == "llm")):
            new_num_of_transitions = replay_buffer.get_num_of_transitions() - replay_buffer.last_pos_in_transition
            logger.info(f"[LLM Training] Rank {rank} | Total transitions: {num_of_transitions} | New transitions: {new_num_of_transitions}")
            
            if llm_collect_mode != "no_collect":
                priorzero_batch = replay_buffer.fetch_latest_batch(batch_size=-1, policy=policy, select_last=True)
            else:
                priorzero_batch = replay_buffer.fetch_latest_batch(batch_size=256, policy=policy, select_last=False)
            # 清理 policy的cahce，防止OOM
            torch.cuda.empty_cache()
            with prof.block("train_llm", rank=rank):
                llm_need_sample_cnt = llm_cfg.train_batch_size * llm_cfg.max_rollout_staleness // world_size
                flag, train_samples = data_processor.make_llm_train_samples(priorzero_batch, ddp=True, max_samples=llm_need_sample_cnt)
                
                if not flag:
                    local_llm_ready = 0
                else:
                    local_llm_ready = 1
                gathered_llm_ready = all_gather_cmd(world_size=world_size, obj=local_llm_ready)
                
                if min(gathered_llm_ready) == 0:
                    logger.info(
                        f"[Rank {rank}] Skip LLM training because not all ranks have enough samples. "
                        f"ready_flags={gathered_llm_ready}, local_ready={local_llm_ready}, required_samples_per_rank={llm_need_sample_cnt}, train_samples={len(train_samples[0])}"
                    )
                    continue
                
                trainer.train_batch(train_samples, collect_env_steps=collector.envstep)
                if llm_collect_mode != "no_collect":
                    replay_buffer.mark_latest_transitions_consumed()
                
                torch_dist_barrier_and_cuda_sync()
                if llm_cfg.enable_world_model and train_alternate and trainer.global_step - last_llm_train_iter >= train_schedule["llm_update_iters"]:
                    current_phase = "wm"
                    last_llm_train_iter = trainer.global_step
                    data_processor.clear_statis()
                    print(f"[Rank {rank}] Switching to World Model training phase at llm iter: {trainer.global_step}")
                    
def main():
    """
    Main entry point with argument parsing.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='PriorZero Training with Auto Model Configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default model (qwen2.5-1.5b)
  torchrun --nproc_per_node 2 priorzero_entry_sync.py

  # Use specific model
  torchrun --nproc_per_node 2 priorzero_entry_sync.py --model qwen2.5-0.5b
  torchrun --nproc_per_node 2 priorzero_entry_sync.py --model qwen2.5-7b

  # List all available models
  python priorzero_entry_sync.py --list-models

  # Different environment
  torchrun --nproc_per_node 2 priorzero_entry_sync.py --env_id zork1.z5 --model qwen2.5-1.5b
        """
    )
    parser.add_argument('--env_id', type=str, default='detective.z5', help='Jericho game ID')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--max_iter', type=int, default=int(1e6), help='Max training iterations')
    parser.add_argument('--quick_test', action='store_true', default=False, help='Use quick test config')
    # Model selection
    parser.add_argument('--model', type=str, default="qwen2.5-3b", choices=get_available_models())
    parser.add_argument('--enable_profile', action='store_true', default=False)
    parser.add_argument('--use_cot', action='store_true', default=False)
    args = parser.parse_args()

    model_key = args.model if args.model else "qwen2.5-1.5b"
    print(f"\n{'='*80}")
    print(f"PriorZero Training Configuration")
    print(f"{'='*80}")
    print(f"Environment: {args.env_id}")
    print(f"Model: {model_key}")
    print(f"Seed: {args.seed}")
    print(f"Quick Test: {args.quick_test}")
    print(f"use cot: {args.use_cot}")
    print(f"enable_profile: {args.enable_profile}")
    print(f"{'='*80}\n")

    if args.quick_test:
        logger.info("Using quick test configuration")
        main_cfg, create_cfg, llm_cfg = get_priorzero_debug_config(
            args.env_id, args.seed, use_cot=args.use_cot,
            exp_name=f'data_priorzero/priorzero_debug_{args.env_id}',
            model_key=model_key,
        )
    else:
        main_cfg, create_cfg, llm_cfg = get_priorzero_config(
            args.env_id, args.seed, use_cot=args.use_cot,
            model_key=model_key,
            multi_gpu=True
        )

    train_priorzero(
        main_cfg,
        create_cfg,
        llm_cfg,
        seed=args.seed,
        max_train_iter=args.max_iter,
        enable_profile=args.enable_profile,    # 是否要对各个耗时部分进行 profile
    )


if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()
