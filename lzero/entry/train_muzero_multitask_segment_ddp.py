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
from lzero.mcts import MuZeroGameBuffer as GameBuffer
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.worker import MuZeroSegmentCollector as Collector
from ding.utils import EasyTimer
timer = EasyTimer()
import torch.distributed as dist

import concurrent.futures


#  ========== TODO ==========
# 设置超时时间 (秒)
TIMEOUT = 3600  # 例如60min


# def safe_eval(evaluator, learner, collector, rank, world_size):
#     try:
#         print(f"=========before eval Rank {rank}/{world_size}===========")
#         # 重置 stop_event，确保每次评估前都处于未设置状态
#         evaluator.stop_event.clear()
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             # 提交 evaluator.eval 任务
#             future = executor.submit(evaluator.eval, learner.save_checkpoint, learner.train_iter, collector.envstep)
            
#             try:
#                 stop, reward = future.result(timeout=TIMEOUT)
#             except concurrent.futures.TimeoutError:
#                 # 超时，设置 evaluator 的 stop_event
#                 evaluator.stop_event.set()
#                 print(f"Eval operation timed out after {TIMEOUT} seconds on Rank {rank}/{world_size}.")

#                 return None, None
        
#         print(f"======after eval Rank {rank}/{world_size}======")
#         return stop, reward
#     except Exception as e:
#         print(f"An error occurred during evaluation on Rank {rank}/{world_size}: {e}")
#         return None, None

def safe_eval(evaluator, learner, collector, rank, world_size):
    print(f"=========before eval Rank {rank}/{world_size}===========")
    # 重置 stop_event，确保每次评估前都处于未设置状态
    evaluator.stop_event.clear()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交 evaluator.eval 任务
        future = executor.submit(evaluator.eval, learner.save_checkpoint, learner.train_iter, collector.envstep)
        
        try:
            stop, reward = future.result(timeout=TIMEOUT)
        except concurrent.futures.TimeoutError:
            # 超时，设置 evaluator 的 stop_event
            evaluator.stop_event.set()
            print(f"Eval operation timed out after {TIMEOUT} seconds on Rank {rank}/{world_size}.")

            return None, None
    
    print(f"======after eval Rank {rank}/{world_size}======")
    return stop, reward



def allocate_batch_size(cfgs, game_buffers, alpha=1.0, clip_scale=1):
    """
    根据不同任务的 num_of_collected_episodes 反比分配 batch_size，
    并动态调整 batch_size 限制范围以提高训练的稳定性和效率。
    
    参数:
    - cfgs: 每个任务的配置列表
    - game_buffers: 每个任务的 replay_buffer 实例列表
    - alpha: 控制反比程度的超参数 (默认为1.0)
    
    返回:
    - 分配后的 batch_size 列表
    """
    
    # 提取每个任务的 num_of_collected_episodes
    buffer_num_of_collected_episodes = [buffer.num_of_collected_episodes for buffer in game_buffers]
    
    # 获取当前的 world_size 和 rank
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # 收集所有 rank 的 num_of_collected_episodes 列表
    all_task_num_of_collected_episodes = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(all_task_num_of_collected_episodes, buffer_num_of_collected_episodes)

    # 将所有 rank 的 num_of_collected_episodes 拼接成一个大列表
    all_task_num_of_collected_episodes = [item for sublist in all_task_num_of_collected_episodes for item in sublist]
    if rank == 0:
        print(f'all_task_num_of_collected_episodes:{all_task_num_of_collected_episodes}')

    # 计算每个任务的反比权重
    inv_episodes = np.array([1.0 / (episodes + 1) for episodes in all_task_num_of_collected_episodes])
    inv_sum = np.sum(inv_episodes)

    # 计算总的 batch_size (所有任务 cfg.policy.max_batch_size 的和)
    max_batch_size = cfgs[0].policy.max_batch_size

    # 动态调整的部分：最小和最大的 batch_size 范围
    avg_batch_size = max_batch_size / world_size
    min_batch_size = avg_batch_size / clip_scale
    max_batch_size = avg_batch_size * clip_scale

    # 动态调整 alpha，让 batch_size 的变化更加平滑
    task_weights = (inv_episodes / inv_sum) ** alpha
    batch_sizes = max_batch_size * task_weights
    
    # 控制 batch_size 在 [min_batch_size, max_batch_size] 之间
    batch_sizes = np.clip(batch_sizes, min_batch_size, max_batch_size)
    
    # 确保 batch_size 是整数
    batch_sizes = [int(size) for size in batch_sizes]
    
    # 返回最终分配的 batch_size 列表
    return batch_sizes


"""
对所有game的任务继续均匀划分：
每个game 对应 1个gpu进程
或多个game对应 1个gpu进程

collector和learner是串行的
evaluator通过ThreadPoolExecutor的timeout和 threading.Event() 强制退出eval()，以避免一个环境评估时的一局步数过长会导致超时

修复了当games>gpu数量时的bug
"""
def train_muzero_multitask_segment_ddp(
        input_cfg_list: List[Tuple[int, Tuple[dict, dict]]],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':
    """
    Overview:
        The train entry for multi-task MuZero, adapted from UniZero's multi-task training.
        This script aims to enhance the planning capabilities of reinforcement learning agents
        by leveraging multi-task learning to address diverse environments.
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
        logging.warning(f"Rank {rank}: No tasks assigned, continuing without tasks.")
        # 初始化一些空列表以避免后续代码报错
        cfgs, game_buffers, collector_envs, evaluator_envs, collectors, evaluators = [], [], [], [], [], []
        return

    print(f"Rank {rank}/{world_size}, handling tasks {start_idx} to {end_idx - 1}")

    cfgs = []
    game_buffers = []
    collector_envs = []
    evaluator_envs = []
    collectors = []
    evaluators = []

    # 使用第一个任务的配置来创建共享的 policy
    task_id, [cfg, create_cfg] = tasks_for_this_rank[0]

    # 设置每个任务的随机种子和任务编号
    for config in tasks_for_this_rank:
        config[1][0].policy.task_num = len(tasks_for_this_rank)

    # 确保指定的 policy 类型是支持的
    # supported_policies = [
    #     'efficientzero', 'muzero', 'muzero_multitask','muzero_context', 'muzero_rnn_full_obs',
    #     'sampled_efficientzero', 'sampled_muzero', 'gumbel_muzero', 'stochastic_muzero'
    # ]
    # assert create_cfg.policy.type in supported_policies, \
    #     f"train_muzero_multitask_segment now only supports {supported_policies}"

    # 根据 CUDA 可用性设置设备
    cfg.policy.device = cfg.policy.model.device if torch.cuda.is_available() else 'cpu'
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
    log_dir = os.path.join('./{}/log'.format(cfg.exp_name), f'serial_rank_{rank}')
    tb_logger = SummaryWriter(log_dir)

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
        torch.cuda.empty_cache()

        if cfg.policy.allocated_batch_sizes:
            # TODO==========
            # 线性变化的 随着 train_epoch 从 0 增加到 1000, clip_scale 从 1 线性增加到 4
            clip_scale = np.clip(1 + (3 * train_epoch / 1000), 1, 4)
            allocated_batch_sizes = allocate_batch_size(cfgs, game_buffers, alpha=1.0, clip_scale=clip_scale)
            if rank == 0:
                print("分配后的 batch_sizes: ", allocated_batch_sizes)
            for idx, (cfg, collector, evaluator, replay_buffer) in enumerate(
                    zip(cfgs, collectors, evaluators, game_buffers)):
                cfg.policy.batch_size = allocated_batch_sizes[idx]
                policy._cfg.batch_size[idx] = allocated_batch_sizes[idx]

        # 对于当前进程的每个任务，进行数据收集和评估
        for idx, (cfg, collector, evaluator, replay_buffer) in enumerate(
                zip(cfgs, collectors, evaluators, game_buffers)):
            
            log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger, cfg.policy.task_id)

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

            if learner.train_iter == 0 or evaluator.should_eval(learner.train_iter):
                print('=' * 20)
                print(f'Rank {rank} evaluates task_id: {cfg.policy.task_id}...')
                
                # 在训练进程中调用 safe_eval
                stop, reward = safe_eval(evaluator, learner, collector, rank, world_size)
                # 判断评估是否成功
                if stop is None or reward is None:
                    print(f"Rank {rank} encountered an issue during evaluation. Continuing training...")
                else:
                    print(f"Evaluation successful: stop={stop}, reward={reward}")

            print('=' * 20)
            print(f'entry: Rank {rank} collects task_id: {cfg.policy.task_id}...')

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

            # 数据收集结束后添加日志
            logging.info(f'Rank {rank}: Completed data collection for task {cfg.policy.task_id}')

        # 检查是否有足够的数据进行训练
        not_enough_data = any(replay_buffer.get_num_of_transitions() < cfgs[0].policy.max_batch_size/world_size for replay_buffer in game_buffers)

        # 同步训练前所有 rank 的准备状态
        try:
            dist.barrier()
            logging.info(f'Rank {rank}: Passed barrier before training')
        except Exception as e:
            logging.error(f'Rank {rank}: Barrier failed with error {e}')
            break  # 或者进行其他错误处理

        # 学习策略
        if not not_enough_data:
            # Learner 将在一次迭代中训练 update_per_collect 次
            for i in range(update_per_collect):
                train_data_multi_task = []
                envstep_multi_task = 0
                for idx, (cfg, collector, replay_buffer) in enumerate(zip(cfgs, collectors, game_buffers)):
                    envstep_multi_task += collector.envstep
                    batch_size = cfg.policy.batch_size[cfg.policy.task_id]
                    if replay_buffer.get_num_of_transitions() > batch_size:
                        # batch_size = cfg.policy.batch_size[cfg.policy.task_id]

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

        # 同步所有 Rank，确保所有 Rank 都完成了训练
        try:
            dist.barrier()
            logging.info(f'Rank {rank}: Passed barrier after training')
        except Exception as e:
            logging.error(f'Rank {rank}: Barrier failed with error {e}')
            break  # 或者进行其他错误处理


        # 检查是否需要终止训练
        try:
            # local_envsteps 不再需要填充
            local_envsteps = [collector.envstep for collector in collectors]

            total_envsteps = [None for _ in range(world_size)]
            dist.all_gather_object(total_envsteps, local_envsteps)

            # 将所有 envsteps 拼接在一起
            all_envsteps = torch.cat([torch.tensor(envsteps, device=cfg.policy.device) for envsteps in total_envsteps])
            max_envstep_reached = torch.all(all_envsteps >= max_env_step)

            # 收集所有进程的 train_iter
            global_train_iter = torch.tensor([learner.train_iter], device=cfg.policy.device)
            all_train_iters = [torch.zeros_like(global_train_iter) for _ in range(world_size)]
            dist.all_gather(all_train_iters, global_train_iter)

            max_train_iter_reached = torch.any(torch.stack(all_train_iters) >= max_train_iter)

            if max_envstep_reached.item() or max_train_iter_reached.item():
                logging.info(f'Rank {rank}: Termination condition met')
                dist.barrier()  # 确保所有进程同步
                break
            else:
                pass

        except Exception as e:
            logging.error(f'Rank {rank}: Termination check failed with error {e}')
            break  # 或者进行其他错误处理

    learner.call_hook('after_run')
    return policy