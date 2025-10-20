# priorzero_entry.py
import asyncio
import os
from functools import partial

import ray
import torch
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.worker import create_buffer
from tensorboardX import SummaryWriter
from loguru import logger

from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

from priorzero_config import get_priorzero_config
from priorzero_collector import PriorZeroCollector
from priorzero_evaluator import PriorZeroEvaluator

async def train_priorzero(cfg: dict, create_cfg: dict, seed: int = 0):
    """
    [PRIORZERO-NEW]
    PriorZero 的异步训练主函数。
    """
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)
    
    # 初始化 Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # 1. 创建 vLLM 引擎
    engine_args = AsyncEngineArgs(
        model=cfg.policy.llm_policy_cfg.pretrain_llm_path,
        tensor_parallel_size=cfg.policy.llm_policy_cfg.vllm_tensor_parallel_size,
        gpu_memory_utilization=cfg.policy.llm_policy_cfg.gpu_memory_utilization,
        worker_use_ray=True,
    )
    vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    logger.info("vLLM Engine created successfully.")

    # 2. 创建环境
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=True)
    
    # 3. 创建策略、Buffer、Collector、Evaluator
    policy = create_policy(cfg.policy, enable_field=['learn', 'collect', 'eval'])
    replay_buffer = create_buffer(cfg.replay_buffer)
    tb_logger = SummaryWriter(os.path.join(f'./{cfg.exp_name}/log/', 'serial'))
    
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

    # --- 主异步训练循环 ---
    train_iter = 0
    while True:
        # 收集数据
        collect_kwargs = {} # 可以传入温度等
        new_data = await collector.collect(train_iter=train_iter, policy_kwargs=collect_kwargs)
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        
        # 训练
        if train_iter > cfg.policy.train_start_after_envsteps:
            for _ in range(cfg.policy.update_per_collect):
                train_data = replay_buffer.sample(cfg.policy.batch_size)
                if train_data is None:
                    break
                
                # 在 learn 方法中传入 game_segments 用于 SFT
                train_data_with_segments = (*train_data, train_iter, train_data.game_segments)
                log_vars = policy.learn(data=train_data_with_segments)
                
                # 日志
                for k, v in log_vars.items():
                    tb_logger.add_scalar(f'train/{k}', v, train_iter)
                train_iter += 1

        # 评估
        if evaluator.should_eval(train_iter):
            stop, reward = await evaluator.eval(policy.save, train_iter, collector.envstep)
            if stop:
                break

if __name__ == "__main__":
    main_cfg, create_cfg = get_priorzero_config()
    asyncio.run(train_priorzero(main_cfg, create_cfg))