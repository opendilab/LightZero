import sys
import os
import logging
from pathlib import Path

import asyncio
import os
import sys
from functools import partial
from pathlib import Path
from typing import Tuple, Optional, List

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

from zoo.priorzero.src.priorzero_config import (
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
    obs_type = getattr(cfg.policy.model.world_model_cfg, 'obs_type', 'text')
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
        obs_type=obs_type,
        env_id=cfg.env.env_id,
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
        obs_type=obs_type,
        env_id=cfg.env.env_id,
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
    requested_obs_type = getattr(cfg.policy.model.world_model_cfg, 'obs_type', 'text')
    if requested_obs_type == 'image':
        if hasattr(llm_cfg, 'validate'):
            llm_cfg.validate()
        if llm_cfg.enable_rft:
            raise NotImplementedError(
                "Image-mode RFT is not supported yet: the current PPO data path does not pass "
                "pixel_values/image_grid_thw to the trainable VL model. Use --vl_fixed for "
                "WM training with a frozen visual prior."
            )

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
    obs_type = getattr(cfg.policy.model.world_model_cfg, 'obs_type', 'text')
    logger.info(f"[Rank {rank}] World Model components initialized")
    if rank == 0:
        dump_dataclass_cfg_py(llm_cfg, path=f"{cfg.exp_name}/llm_cfg.py")
        llm_cfg.save_path = f'./{cfg.exp_name}/llm_ckpt/'

    from utils import Profiler
    prof = Profiler(log_interval=10, stats_file=f'./{cfg.exp_name}/log/profiler.txt', enable_profile=enable_profile)

    
    logger.info(f"[Rank {rank}] Initializing prior actor (obs_type={obs_type})...")
    set_pkg_seed(seed + rank, use_cuda=True)
    
    ref_model = None
    if llm_cfg.enable_rft:
        from models.actor import PolicyModel, ReferenceModel
    if llm_cfg.enable_rft and llm_cfg.rft_kl_coef > 0:
        ref_model = ReferenceModel(
            strategy=strategy,
            pretrain=llm_cfg.model_name_or_path
        )
    
    prior_generator = None
    if obs_type == 'image':
        from vl_engine import create_vl_engine
        vlm_image_mode = getattr(llm_cfg, 'vlm_image_mode', 'current_only')
        image_limit = 1 if vlm_image_mode == 'current_only' else llm_cfg.history_length + 1
        prior_engine = create_vl_engine(
            model_name=llm_cfg.vl_model_type,
            model_path=llm_cfg.model_name_or_path,
            tensor_parallel_size=llm_cfg.tensor_parallel_size,
            gpu_memory_utilization=llm_cfg.gpu_memory_utilization,
            max_model_len=llm_cfg.prompt_max_len + llm_cfg.generate_max_len,
            limit_mm_per_prompt={'image': image_limit},
            enable_sleep=llm_cfg.vllm_enable_sleep,
        )
        from prior_generator import VLPriorGenerator
        prior_generator = VLPriorGenerator(
            vl_engine=prior_engine,
            model_name=llm_cfg.model_name_or_path,
            use_cot=llm_cfg.use_cot,
            game_description=getattr(llm_cfg, 'game_description', ''),
            vlm_image_mode=vlm_image_mode,
            prompt_style=getattr(llm_cfg, 'prompt_style', 'legacy'),
            logprob_extraction_mode=getattr(llm_cfg, 'logprob_extraction_mode', 'approximate'),
            max_new_tokens=getattr(llm_cfg, 'max_new_tokens', 128),
        )
    else:
        from vllm_utils.vllm_engine import create_vllm_engine
        prior_engine = create_vllm_engine(
            tensor_parallel_size=llm_cfg.vllm_tensor_parallel_size,
            pretrain=llm_cfg.model_name_or_path,
            enable_prefix_caching=llm_cfg.enable_prefix_caching,
            max_model_len=llm_cfg.prompt_max_len + llm_cfg.generate_max_len,
            gpu_memory_utilization=llm_cfg.gpu_memory_utilization,
            vllm_enable_sleep=llm_cfg.vllm_enable_sleep,
        )

    print(f'[Rank {rank}] Prior engine successfully created (obs_type={obs_type})!')
    
    data_processor = None
    if obs_type != 'image' or llm_cfg.enable_rft:
        from priorzero_datafactory import DataProcessor
        data_processor = DataProcessor(rank=rank,
                                       world_size=world_size,
                                       vllm_engine=prior_engine,
                                       strategy=strategy,
                                       model_path=llm_cfg.model_name_or_path,
                                       exp_name=cfg.exp_name if rank == 0 else None,
                                       obs_type=obs_type,
                                       prior_generator=prior_generator,
                                    )
    # 在collector中初始化data_processor 和prof对象
    collector.data_processor = data_processor
    collector.prior_generator = prior_generator
    collector.prof = prof
    evaluator.data_processor = data_processor
    evaluator.prior_generator = prior_generator
    
    policy_model = None
    trainer = None
    if llm_cfg.enable_rft:
        policy_model = PolicyModel(
            strategy=strategy,
            pretrain=llm_cfg.model_name_or_path,
            vllm_engine=prior_engine,
            max_steps=llm_cfg.max_steps
        )
        from priorzero_trainer import PriorZeroLLMTrainer
        trainer = PriorZeroLLMTrainer(
            cfg=llm_cfg,
            pretrain=llm_cfg.model_name_or_path,
            strategy=strategy,
            vllm_engine=prior_engine,
            policy_model=policy_model,
            reference_model=ref_model,
            exp_name=cfg.exp_name if rank == 0 else None,
            tb_logger=tb_logger if rank == 0 else None,
            llm_save_freq=getattr(llm_cfg, 'llm_save_freq', getattr(llm_cfg, 'vl_save_freq', 1000))
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
        llm_collect_mode = train_schedule.get("llm_collect_mode", "wm_llm_collect")

    while True:
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break
        
        llm_train_iter = policy_model.train_iter if policy_model is not None else 0

        # 1.评估阶段
        if learner.train_iter != 0 and evaluator.should_eval(wm_train_iter=learner.train_iter, llm_train_iter=llm_train_iter, phase=current_phase):
            logger.info(f"[Evaluator][Rank {rank}: Iter {learner.train_iter}] Evaluating...")
            if llm_cfg.vllm_enable_sleep and prior_engine is not None:
                prior_engine.wake_up()
            evaluator.eval(wm_train_iter=learner.train_iter, llm_train_iter=llm_train_iter, phase=current_phase)
            if llm_cfg.vllm_enable_sleep and prior_engine is not None:
                prior_engine.sleep()
            torch_dist_barrier_and_cuda_sync()
        
        # 2.数据收集阶段         
        if not train_alternate or (train_alternate and current_phase == "wm") or (train_alternate and current_phase == "llm" and llm_collect_mode != "no_collect"):
            if llm_cfg.vllm_enable_sleep and prior_engine is not None:
                prior_engine.wake_up()
            
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs={'temperature': 0.25, 'epsilon': 0.0}, phase=current_phase)
            if obs_type == 'image':
                prior_generator.get_vl_output_log(
                    wm_train_iter=learner.train_iter,
                    vl_train_iter=llm_train_iter,
                )
            else:
                data_processor.get_llm_output_log(
                    wm_train_iter=learner.train_iter,
                    llm_train_iter=llm_train_iter,
                )
            
            if llm_cfg.vllm_enable_sleep and prior_engine is not None:
                prior_engine.sleep()
            
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
                priorzero_batch = replay_buffer.fetch_latest_batch(
                    batch_size=min(256, num_of_transitions), policy=policy, select_last=False
                )
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

    parser = argparse.ArgumentParser(description='PriorZero training')
    parser.add_argument(
        '--input_type', choices=['text', 'image'], default='text',
        help='Prior input type. This selects the minimal text/VL branch.'
    )
    parser.add_argument('--env_id', type=str, default='detective.z5', help='Jericho game ID')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--max_iter', type=int, default=int(1e6), help='Max training iterations')
    parser.add_argument('--quick_test', action='store_true', default=False, help='Use quick test config')
    # Model selection
    parser.add_argument('--model', type=str, default="qwen2.5-3b", choices=get_available_models())
    parser.add_argument('--vl_model', type=str, default='Qwen2.5-VL-3b')
    parser.add_argument('--enable_profile', action='store_true', default=False)
    cot_group = parser.add_mutually_exclusive_group()
    cot_group.add_argument('--use_cot', dest='use_cot', action='store_true')
    cot_group.add_argument('--no_cot', dest='use_cot', action='store_false')
    parser.set_defaults(use_cot=None)

    parser.add_argument('--cot_weight', type=float, default=0.1)
    parser.add_argument(
        '--mcts_mode', choices=['llm_logits', 'wm_logits', 'llm_plus_wm_logits'],
        default='llm_plus_wm_logits'
    )
    parser.add_argument(
        '--vlm_image_mode', choices=['current_only', 'first_and_current', 'all_history'],
        default='current_only'
    )
    parser.add_argument('--prompt_style', choices=['concise', 'legacy'], default='legacy')
    parser.add_argument('--logprob_mode', choices=['exact', 'approximate'], default='approximate')
    vl_fixed_group = parser.add_mutually_exclusive_group()
    vl_fixed_group.add_argument(
        '--vl_fixed', dest='vl_fixed', action='store_true',
        help='Use the VL model as a frozen prior (currently required for image input).'
    )
    vl_fixed_group.add_argument(
        '--no_vl_fixed', dest='vl_fixed', action='store_false',
        help='Request VL RFT; rejected until multimodal PPO inputs are implemented.'
    )
    parser.set_defaults(vl_fixed=True)
    args = parser.parse_args()

    model_key = args.model if args.input_type == 'text' else args.vl_model
    print(f"\n{'='*80}")
    print(f"PriorZero Training Configuration")
    print(f"{'='*80}")
    print(f"Environment: {args.env_id}")
    print(f"Input type: {args.input_type}")
    print(f"Model: {model_key}")
    print(f"Seed: {args.seed}")
    print(f"Quick Test: {args.quick_test}")
    print(f"use cot: {args.use_cot if args.use_cot is not None else 'config default'}")
    print(f"enable_profile: {args.enable_profile}")
    print(f"{'='*80}\n")

    if args.input_type == 'image':
        from zoo.priorzero.src.vl_config import get_priorzero_vl_config
        main_cfg, create_cfg, llm_cfg = get_priorzero_vl_config(
            env_id=args.env_id,
            seed=args.seed,
            vl_model_key=args.vl_model,
            use_prior=True,
            multi_gpu=int(os.environ.get('WORLD_SIZE', '1')) > 1,
            quick_test=args.quick_test,
        )
        if llm_cfg is None:
            raise ValueError("Image mode requires use_prior=True")
        if args.use_cot is not None:
            llm_cfg.use_cot = args.use_cot
        llm_cfg.cot_weight = args.cot_weight
        llm_cfg.mcts_root_logits_dict.mode = args.mcts_mode
        llm_cfg.vlm_image_mode = args.vlm_image_mode
        llm_cfg.prompt_style = args.prompt_style
        llm_cfg.logprob_extraction_mode = args.logprob_mode
        llm_cfg.vl_fixed = args.vl_fixed
        if args.vl_fixed:
            llm_cfg.enable_rft = False
    else:
        text_use_cot = bool(args.use_cot)
        if args.quick_test:
            logger.info("Using quick test configuration")
            main_cfg, create_cfg, llm_cfg = get_priorzero_debug_config(
                args.env_id, args.seed, use_cot=text_use_cot,
                exp_name=f'all_experiments/data_priorzero/priorzero_debug_{args.env_id}',
                model_key=args.model,
            )
        else:
            main_cfg, create_cfg, llm_cfg = get_priorzero_config(
                args.env_id, args.seed, use_cot=text_use_cot,
                model_key=args.model,
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
