import sys
import os
import logging
from pathlib import Path

# Add Jericho PriorZero src to path for shared modules
_jericho_src = str(Path(__file__).resolve().parent.parent.parent.parent / "jericho" / "priorzero" / "src")
# Local src dir first so priorzero_config resolves to BabyAI version
_local_src = str(Path(__file__).resolve().parent)
sys.path.insert(0, _jericho_src)
sys.path.insert(0, _local_src)

import asyncio
from functools import partial
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

from priorzero_config import (
    get_priorzero_config,
    get_priorzero_debug_config,
    get_available_models,
)
from priorzero_collector import PriorZeroCollector
from priorzero_evaluator import PriorZeroEvaluator
from priorzero_policy import *
from lzero.mcts.buffer.game_buffer_priorzero import PriorZeroGameBufferOptimized
from utils import dump_dataclass_cfg_py, setup_priorzero_logging

from lzero.entry.utils import calculate_update_per_collect

_log_main = logging.getLogger("priorzero.main")
_log_train = logging.getLogger("priorzero.train")
_log_eval = logging.getLogger("priorzero.eval")

def prepare_unizero(rank, cfg, create_cfg, llm_cfg, seed):
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)

    policy = create_policy(cfg.policy, enable_field=['learn', 'collect', 'eval'], exp_name=cfg.exp_name, llm_cfg=llm_cfg)
    if cfg.policy.model_path is not None:
        _log_main.info(f"Loading pretrained model from {cfg.policy.model_path}")
        policy.learn_mode.load_state_dict(torch.load(cfg.policy.model_path, map_location=cfg.policy.device))

    os.makedirs(f'./{cfg.exp_name}/log/', exist_ok=True)
    tb_logger = SummaryWriter(os.path.join(f'./{cfg.exp_name}/log/', 'serial')) if get_rank() == 0 else None

    learner = BaseLearner(
        cfg.policy.learn.learner,
        policy.learn_mode,
        tb_logger,
        exp_name=cfg.exp_name
    )

    replay_buffer = PriorZeroGameBufferOptimized(cfg.policy)

    collector = PriorZeroCollector(
        env=collector_env,
        policy=policy.collect_mode,
        llm_config=llm_cfg,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=cfg.policy,
    )

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
    learner.call_hook('before_run')
    _log_main.info("Policy, Learner, Collector, Evaluator created")

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
    from strategy.deepspeed import get_strategy, torch_dist_barrier_and_cuda_sync
    strategy = get_strategy(llm_cfg)

    strategy.setup_distributed()
    world_size = getattr(strategy, "world_size", 1)

    cfg, replay_buffer, tb_logger, policy, collector, evaluator, learner = prepare_unizero(
        rank=rank, cfg=cfg, create_cfg=create_cfg, llm_cfg=llm_cfg, seed=seed
    )
    batch_size = cfg.policy.batch_size

    # Initialize structured logging after exp_name is known
    setup_priorzero_logging(cfg.exp_name, rank)
    _log_main.info(f"=== PriorZero Training Start | rank={rank}/{world_size} | exp={cfg.exp_name} ===")

    if rank == 0:
        dump_dataclass_cfg_py(llm_cfg, path=f"{cfg.exp_name}/llm_cfg.py")
        # Save config snapshot
        import yaml
        config_path = os.path.join(cfg.exp_name, "run_logs", "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump({"llm_cfg": str(llm_cfg), "policy_cfg": str(cfg.policy)}, f, default_flow_style=False)
        llm_cfg.save_path = f'./{cfg.exp_name}/llm_ckpt/'

    from utils import Profiler
    prof = Profiler(log_interval=10, stats_file=f'./{cfg.exp_name}/log/profiler.txt', enable_profile=enable_profile)

    _log_main.info("Initializing LLM Actor...")
    set_pkg_seed(seed + rank, use_cuda=True)

    from models.actor import PolicyModel, ReferenceModel
    if llm_cfg.rft_kl_coef > 0:
        ref_model = ReferenceModel(strategy=strategy, pretrain=llm_cfg.model_name_or_path)
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
    _log_main.info("vLLM engine created")

    from priorzero_datafactory import DataProcessor
    data_processor = DataProcessor(
        rank=rank, world_size=world_size, vllm_engine=vllm_engine,
        strategy=strategy, model_path=llm_cfg.model_name_or_path,
        exp_name=cfg.exp_name if rank == 0 else None,
    )
    collector.data_processor = data_processor
    collector.prof = prof
    evaluator.data_processor = data_processor

    policy_model = PolicyModel(
        strategy=strategy, pretrain=llm_cfg.model_name_or_path,
        vllm_engine=vllm_engine, max_steps=llm_cfg.max_steps
    )
    from priorzero_trainer import PriorZeroLLMTrainer
    trainer = PriorZeroLLMTrainer(
        cfg=llm_cfg, pretrain=llm_cfg.model_name_or_path,
        strategy=strategy, vllm_engine=vllm_engine,
        policy_model=policy_model, reference_model=ref_model,
        exp_name=cfg.exp_name if rank == 0 else None,
        tb_logger=tb_logger if rank == 0 else None,
        llm_save_freq=llm_cfg.llm_save_freq
    )

    torch_dist_barrier_and_cuda_sync()
    train_schedule = llm_cfg.train_schedule
    train_alternate = train_schedule["alternate"]
    current_phase = None
    if train_alternate:
        current_phase = train_schedule["start_phase"]
        last_wm_train_iter = 0
        last_llm_train_iter = 0

    _log_eval.info("=== Initial Evaluation ===")
    if llm_cfg.vllm_enable_sleep and vllm_engine is not None:
        vllm_engine.wake_up()
    evaluator.eval(wm_train_iter=0, llm_train_iter=0, phase=current_phase, env_step=collector.envstep)
    if llm_cfg.vllm_enable_sleep and vllm_engine is not None:
        vllm_engine.sleep()
    torch_dist_barrier_and_cuda_sync()

    _log_main.info(f"=== Training Loop Start | phase={current_phase} ===")
    while True:
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

        if learner.train_iter != 0 and evaluator.should_eval(wm_train_iter=learner.train_iter, llm_train_iter=policy_model.train_iter, phase=current_phase, env_step=collector.envstep):
            _log_eval.info(f"=== Eval | wm_iter={learner.train_iter} llm_iter={policy_model.train_iter} phase={current_phase} envstep={collector.envstep} ===")
            if llm_cfg.vllm_enable_sleep and vllm_engine is not None:
                vllm_engine.wake_up()
            evaluator.eval(wm_train_iter=learner.train_iter, llm_train_iter=policy_model.train_iter, phase=current_phase, env_step=collector.envstep)
            if llm_cfg.vllm_enable_sleep and vllm_engine is not None:
                vllm_engine.sleep()

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

        if llm_cfg.enable_world_model and (not train_alternate or (train_alternate and current_phase == "wm")):
            if not (num_of_transitions > batch_size):
                _log_train.warning(f"[WM] Data insufficient: buffer={num_of_transitions} < batch={batch_size}")
                cmd = 0
            else:
                cmd = 1
            if min(all_gather_cmd(world_size=world_size, obj=cmd)) == 0:
                continue

            update_per_collect = calculate_update_per_collect(cfg, new_data, world_size=world_size)
            _log_train.info(f"[WM] Iter {learner.train_iter} | updates={update_per_collect} | buffer={num_of_transitions}")

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
                replay_buffer.mark_latest_transitions_consumed()
                _log_main.info(f"=== Phase Switch: WM -> LLM | wm_iter={learner.train_iter} ===")
                continue

        if llm_cfg.enable_rft and (not train_alternate or (train_alternate and current_phase == "llm")):
            new_num_of_transitions = replay_buffer.get_num_of_transitions() - replay_buffer.last_pos_in_transition
            _log_train.info(f"[LLM] Total={num_of_transitions} | New={new_num_of_transitions}")

            with prof.block("fetch_latest_batch", rank=rank):
                priorzero_batch = replay_buffer.fetch_latest_batch(batch_size=-1, policy=policy)
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
                    _log_train.debug(f"Skip LLM training: not all ranks ready. flags={gathered_llm_ready}")
                    continue

                trainer.train_batch(train_samples, collect_env_steps=collector.envstep)
                replay_buffer.mark_latest_transitions_consumed()

                torch_dist_barrier_and_cuda_sync()
                if llm_cfg.enable_world_model and train_alternate and trainer.global_step - last_llm_train_iter >= train_schedule["llm_update_iters"]:
                    current_phase = "wm"
                    last_llm_train_iter = trainer.global_step
                    data_processor.clear_statis()
                    _log_main.info(f"=== Phase Switch: LLM -> WM | llm_iter={trainer.global_step} ===")

def main():
    import argparse
    import requests as req

    parser = argparse.ArgumentParser(description='PriorZero BabyAI Training')
    parser.add_argument('--env_id', type=str, default='babyai', help='Environment ID')
    parser.add_argument('--env_addr', type=str, default='http://127.0.0.1:8000', help='BabyAI server address')
    parser.add_argument('--use_high_level_actions', action='store_true', default=True, help='Use server high-level actions')
    parser.add_argument('--use_low_level_actions', action='store_true', default=False, help='Use 7 atomic actions')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--max_iter', type=int, default=int(1e6), help='Max training iterations')
    parser.add_argument('--quick_test', action='store_true', default=False, help='Use debug config')
    parser.add_argument('--model', type=str, default="qwen2.5-3b", choices=get_available_models())
    parser.add_argument('--enable_profile', action='store_true', default=False)
    parser.add_argument('--use_cot', action='store_true', default=False)
    args = parser.parse_args()

    use_high_level = not args.use_low_level_actions
    model_key = args.model

    args.seed = 1


    rank = int(os.environ.get("RANK", "0"))

    if rank == 0:
        try:
            r = req.get(f"{args.env_addr}/", timeout=5)
            assert r.status_code == 200, f"Server returned status {r.status_code}"
        except Exception as e:
            raise RuntimeError(
                f"BabyAI server not reachable at {args.env_addr}: {e}\n"
                f"Start it first: cd <AgentGym-RL>/AgentGym/agentenv-babyai && python -m agentenv_babyai.launch --port 8000"
            )
        print(f"[PriorZero] model={model_key} | server={args.env_addr} | 18 levels | seed={args.seed} | cot={args.use_cot}")

    if args.quick_test:
        main_cfg, create_cfg, llm_cfg = get_priorzero_debug_config(
            args.env_id, args.seed, use_cot=args.use_cot,
            exp_name='data_priorzero/babyai/priorzero_debug_multitask',
            model_key=model_key, env_addr=args.env_addr,
            use_high_level_actions=use_high_level,
        )
    else:
        main_cfg, create_cfg, llm_cfg = get_priorzero_config(
            args.env_id, args.seed, use_cot=args.use_cot,
            model_key=model_key, multi_gpu=True,
            env_addr=args.env_addr,
            use_high_level_actions=use_high_level,
        )

    train_priorzero(
        main_cfg, create_cfg, llm_cfg,
        seed=args.seed, max_train_iter=args.max_iter,
        enable_profile=args.enable_profile,
    )


if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()
