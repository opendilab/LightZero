import logging
import os
from functools import partial
from typing import Tuple, Optional, List

import torch
import numpy as np
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import set_pkg_seed, get_rank, get_world_size
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter

from lzero.entry.utils import log_buffer_memory_usage
from lzero.policy import visit_count_temperature
from lzero.mcts import UniZeroGameBuffer as GameBuffer
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.worker import MuZeroSegmentCollector as Collector
from ding.utils import EasyTimer
timer = EasyTimer()
import torch.distributed as dist


def train_unizero_multitask_segment(
        input_cfg_list: List[Tuple[int, Tuple[dict, dict]]],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':
    """
    Overview:
        The train entry for UniZero, proposed in our paper UniZero: Generalized and Efficient Planning with Scalable Latent World Models.
        UniZero aims to enhance the planning capabilities of reinforcement learning agents by addressing the limitations found in MuZero-style algorithms,
        particularly in environments requiring the capture of long-term dependencies. More details can be found in https://arxiv.org/abs/2406.10667.
    Arguments:
        - input_cfg_list (List[Tuple[int, Tuple[dict, dict]]]): List of configurations for different tasks.
        - seed (int): Random seed.
        - model (Optional[torch.nn.Module]): Instance of torch.nn.Module.
        - model_path (Optional[str]): The pretrained model path, which should point to the ckpt file of the pretrained model.
        - max_train_iter (Optional[int]): Maximum policy update iterations in training.
        - max_env_step (Optional[int]): Maximum collected environment interaction steps.
    Returns:
        - policy (Policy): Converged policy.
    """
    # 获取当前进程的 rank 和总的进程数
    rank = get_rank()
    world_size = get_world_size()

    # 任务划分
    total_tasks = len(input_cfg_list)
    tasks_per_rank = total_tasks // world_size
    remainder = total_tasks % world_size

    if rank < remainder:
        start_idx = rank * (tasks_per_rank + 1)
        end_idx = start_idx + tasks_per_rank + 1
    else:
        start_idx = rank * tasks_per_rank + remainder
        end_idx = start_idx + tasks_per_rank

    tasks_for_this_rank = input_cfg_list[start_idx:end_idx]

    # 确保至少有一个任务
    if len(tasks_for_this_rank) == 0:
        print(f"Rank {rank}: No tasks assigned, exiting.")
        return

    print(f"Rank {rank}/{world_size}, handling tasks {start_idx} to {end_idx - 1}")

    cfgs = []
    game_buffers = []
    collector_envs = []
    evaluator_envs = []
    collectors = []
    evaluators = []

    # 使用第一个任务的配置来创建共享的 policy
    task_id, [cfg, create_cfg] = input_cfg_list[0]

    # 确保指定的 policy 类型是支持的
    assert create_cfg.policy.type in ['unizero_multitask'], "train_unizero entry now only supports 'unizero_multitask'"

    # 根据 CUDA 可用性设置设备
    cfg.policy.device = cfg.policy.model.world_model_cfg.device if torch.cuda.is_available() else 'cpu'
    logging.info(f'cfg.policy.device: {cfg.policy.device}')

    # 编译配置
    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
    # 创建共享的 policy
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    # 如果指定了预训练模型，则加载
    if model_path is not None:
        logging.info(f'Loading model from {model_path} begin...')
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))
        logging.info(f'Loading model from {model_path} end!')

    # 创建 TensorBoard 的日志记录器
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')) if get_rank() == 0 else None

    # 创建共享的 learner
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

    policy_config = cfg.policy
    batch_size = policy_config.batch_size[0]

    # 只处理当前进程分配到的任务
    for local_task_id, (task_id, [cfg, create_cfg]) in enumerate(tasks_for_this_rank):
        # 设置每个任务自己的随机种子
        cfg.policy.device = 'cuda' if cfg.policy.cuda and torch.cuda.is_available() else 'cpu'
        cfg = compile_config(cfg, seed=seed + task_id, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
        policy_config = cfg.policy
        policy.collect_mode.get_attribute('cfg').n_episode = policy_config.n_episode
        policy.eval_mode.get_attribute('cfg').n_episode = policy_config.n_episode

        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
        collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
        evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
        collector_env.seed(cfg.seed + task_id)
        evaluator_env.seed(cfg.seed + task_id, dynamic_seed=False)
        set_pkg_seed(cfg.seed + task_id, use_cuda=cfg.policy.cuda)

        # 为每个任务创建不同的 game buffer、collector、evaluator
        replay_buffer = GameBuffer(policy_config)
        collector = Collector(
            env=collector_env,
            policy=policy.collect_mode,
            tb_logger=tb_logger,
            exp_name=cfg.exp_name,
            policy_config=policy_config,
            task_id=task_id
        )
        evaluator = Evaluator(
            eval_freq=cfg.policy.eval_freq,
            n_evaluator_episode=cfg.env.n_evaluator_episode,
            stop_value=cfg.env.stop_value,
            env=evaluator_env,
            policy=policy.eval_mode,
            tb_logger=tb_logger,
            exp_name=cfg.exp_name,
            policy_config=policy_config,
            task_id=task_id
        )

        cfgs.append(cfg)
        replay_buffer.batch_size = cfg.policy.batch_size[task_id]
        game_buffers.append(replay_buffer)
        collector_envs.append(collector_env)
        evaluator_envs.append(evaluator_env)
        collectors.append(collector)
        evaluators.append(evaluator)

    learner.call_hook('before_run')
    value_priority_tasks = {}

    buffer_reanalyze_count = 0
    train_epoch = 0
    reanalyze_batch_size = cfg.policy.reanalyze_batch_size
    update_per_collect = cfg.policy.update_per_collect

    while True:
        # 预先计算位置嵌入矩阵（如果需要）
        policy._collect_model.world_model.precompute_pos_emb_diff_kv()
        policy._target_model.world_model.precompute_pos_emb_diff_kv()

        # 对于当前进程的每个任务，进行数据收集和评估
        for idx, (cfg, collector, evaluator, replay_buffer) in enumerate(
                zip(cfgs, collectors, evaluators, game_buffers)):
            log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger)

            collect_kwargs = {
                'temperature': visit_count_temperature(
                    policy_config.manual_temperature_decay,
                    policy_config.fixed_temperature_value,
                    policy_config.threshold_training_steps_for_final_temperature,
                    trained_steps=learner.train_iter
                ),
                'epsilon': 0.0  # 默认的 epsilon 值
            }

            if policy_config.eps.eps_greedy_exploration_in_collect:
                epsilon_greedy_fn = get_epsilon_greedy_fn(
                    start=policy_config.eps.start,
                    end=policy_config.eps.end,
                    decay=policy_config.eps.decay,
                    type_=policy_config.eps.type
                )
                collect_kwargs['epsilon'] = epsilon_greedy_fn(collector.envstep)

            if evaluator.should_eval(learner.train_iter):
                print('=' * 20)
                print(f'Rank {rank} evaluates task_id: {cfg.policy.task_id}...')
                stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
                if stop:
                    break

            print('=' * 20)
            print(f'Rank {rank} collects task_id: {cfg.policy.task_id}...')

            # 在每次收集之前重置初始数据，这对于多任务设置非常重要
            collector._policy.reset(reset_init_data=True)
            # 收集数据
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)

            # 更新 replay buffer
            replay_buffer.push_game_segments(new_data)
            replay_buffer.remove_oldest_data_to_fit()

            # 周期性地重新分析缓冲区
            if cfg.policy.buffer_reanalyze_freq >= 1:
                # 在一个训练 epoch 中重新分析缓冲区 <buffer_reanalyze_freq> 次
                reanalyze_interval = update_per_collect // cfg.policy.buffer_reanalyze_freq
            else:
                # 每 <1/buffer_reanalyze_freq> 个训练 epoch 重新分析一次缓冲区
                if train_epoch % int(1/cfg.policy.buffer_reanalyze_freq) == 0 and replay_buffer.get_num_of_transitions()//cfg.policy.num_unroll_steps > int(reanalyze_batch_size/cfg.policy.reanalyze_partition):
                    with timer:
                        # 每个重新分析过程将重新分析 <reanalyze_batch_size> 个序列
                        replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                    buffer_reanalyze_count += 1
                    logging.info(f'Buffer reanalyze count: {buffer_reanalyze_count}')
                    logging.info(f'Buffer reanalyze time: {timer.value}')

        # 检查是否有足够的数据进行训练
        not_enough_data = any(replay_buffer.get_num_of_transitions() < batch_size for replay_buffer in game_buffers)

        # 学习策略
        if not not_enough_data:
            # 同步训练前所有 rank 的准备状态
            dist.barrier()

            # Learner 将在一次迭代中训练 update_per_collect 次
            for i in range(update_per_collect):
                train_data_multi_task = []
                envstep_multi_task = 0
                for idx, (cfg, collector, replay_buffer) in enumerate(zip(cfgs, collectors, game_buffers)):
                    envstep_multi_task += collector.envstep
                    if replay_buffer.get_num_of_transitions() > batch_size:
                        batch_size = cfg.policy.batch_size[cfg.policy.task_id]

                        if cfg.policy.buffer_reanalyze_freq >= 1:
                            # 在一个训练 epoch 中重新分析缓冲区 <buffer_reanalyze_freq> 次
                            if i % reanalyze_interval == 0 and replay_buffer.get_num_of_transitions() // cfg.policy.num_unroll_steps > int(
                                    reanalyze_batch_size / cfg.policy.reanalyze_partition):
                                with timer:
                                    # 每个重新分析过程将重新分析 <reanalyze_batch_size> 个序列
                                    replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                                buffer_reanalyze_count += 1
                                logging.info(f'Buffer reanalyze count: {buffer_reanalyze_count}')
                                logging.info(f'Buffer reanalyze time: {timer.value}')

                        train_data = replay_buffer.sample(batch_size, policy)
                        # 追加 task_id，以便在训练时区分任务
                        train_data.append(cfg.policy.task_id)
                        train_data_multi_task.append(train_data)
                    else:
                        logging.warning(
                            f'The data in replay_buffer is not sufficient to sample a mini-batch: '
                            f'batch_size: {batch_size}, replay_buffer: {replay_buffer}'
                        )
                        break

                if train_data_multi_task:
                    # 在训练时，DDP 会自动同步梯度和参数
                    log_vars = learner.train(train_data_multi_task, envstep_multi_task)

                if cfg.policy.use_priority:
                    for idx, (cfg, replay_buffer) in enumerate(zip(cfgs, game_buffers)):
                        # 更新任务特定的 replay buffer 的优先级
                        task_id = cfg.policy.task_id
                        replay_buffer.update_priority(train_data_multi_task[idx], log_vars[0][f'value_priority_task{task_id}'])

                        current_priorities = log_vars[0][f'value_priority_task{task_id}']

                        mean_priority = np.mean(current_priorities)
                        std_priority = np.std(current_priorities)

                        alpha = 0.1  # 运行均值的平滑因子
                        if f'running_mean_priority_task{task_id}' not in value_priority_tasks:
                            # 如果不存在，则初始化运行均值
                            value_priority_tasks[f'running_mean_priority_task{task_id}'] = mean_priority
                        else:
                            # 更新运行均值
                            value_priority_tasks[f'running_mean_priority_task{task_id}'] = (
                                alpha * mean_priority + (1 - alpha) * value_priority_tasks[f'running_mean_priority_task{task_id}']
                            )

                        # 使用运行均值计算归一化的优先级
                        running_mean_priority = value_priority_tasks[f'running_mean_priority_task{task_id}']
                        normalized_priorities = (current_priorities - running_mean_priority) / (std_priority + 1e-6)

                        # 如果需要，可以将归一化的优先级存储回 replay buffer
                        # replay_buffer.update_priority(train_data_multi_task[idx], normalized_priorities)

                        # 如果设置了 print_task_priority_logs 标志，则记录统计信息
                        if cfg.policy.print_task_priority_logs:
                            print(f"Task {task_id} - Mean Priority: {mean_priority:.8f}, "
                                  f"Running Mean Priority: {running_mean_priority:.8f}, "
                                  f"Standard Deviation: {std_priority:.8f}")

        train_epoch += 1
        policy.recompute_pos_emb_diff_and_clear_cache()

        # 同步所有 Rank，确保所有 Rank 都完成了训练
        dist.barrier()

        # 检查是否需要终止训练

        # 收集所有进程的 envstep
        local_envsteps = torch.tensor([collector.envstep for collector in collectors], device=cfg.policy.device)
        total_envsteps = [torch.zeros_like(local_envsteps) for _ in range(world_size)]
        dist.all_gather(total_envsteps, local_envsteps)
        all_envsteps = torch.cat(total_envsteps)
        max_envstep_reached = torch.all(all_envsteps >= max_env_step)

        # 收集所有进程的 train_iter
        global_train_iter = torch.tensor([learner.train_iter], device=cfg.policy.device)
        all_train_iters = [torch.zeros_like(global_train_iter) for _ in range(world_size)]
        dist.all_gather(all_train_iters, global_train_iter)
        max_train_iter_reached = torch.any(torch.stack(all_train_iters) >= max_train_iter)

        if max_envstep_reached.item() or max_train_iter_reached.item():
            break

    learner.call_hook('after_run')
    return policy