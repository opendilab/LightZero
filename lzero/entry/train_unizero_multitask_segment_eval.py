import logging
import os
from functools import partial
from typing import Tuple, Optional, List, Dict, Any

import torch
import numpy as np
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import set_pkg_seed, get_rank, get_world_size, EasyTimer
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter

from lzero.entry.utils import log_buffer_memory_usage
from lzero.policy import visit_count_temperature
from lzero.mcts import UniZeroGameBuffer as GameBuffer
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.worker import MuZeroSegmentCollector as Collector

import torch.distributed as dist
import concurrent.futures

# 设置超时时间 (秒)
TIMEOUT = 12000  # 例如200分钟


def safe_eval(
        evaluator: Evaluator,
        learner: BaseLearner,
        collector: Collector,
        rank: int,
        world_size: int
) -> Tuple[Optional[bool], Optional[float]]:
    """
    Safely evaluates the policy using the evaluator with a timeout.

    Args:
        evaluator (Evaluator): The evaluator instance.
        learner (BaseLearner): The learner instance.
        collector (Collector): The collector instance.
        rank (int): The rank of the current process.
        world_size (int): Total number of processes.

    Returns:
        Tuple[Optional[bool], Optional[float]]: A tuple containing the stop flag and reward.
    """
    try:
        print(f"=========before eval Rank {rank}/{world_size}===========")
        # 重置 stop_event，确保每次评估前都处于未设置状态
        evaluator.stop_event.clear()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 提交 evaluator.eval 任务
            future = executor.submit(
                evaluator.eval,
                learner.save_checkpoint,
                learner.train_iter,
                collector.envstep
            )

            try:
                stop, reward = future.result(timeout=TIMEOUT)
            except concurrent.futures.TimeoutError:
                # 超时，设置 evaluator 的 stop_event
                evaluator.stop_event.set()
                print(f"Eval operation timed out after {TIMEOUT} seconds on Rank {rank}/{world_size}.")
                return None, None

        print(f"======after eval Rank {rank}/{world_size}======")
        return stop, reward
    except Exception as e:
        print(f"An error occurred during evaluation on Rank {rank}/{world_size}: {e}")
        return None, None


def allocate_batch_size(
        cfgs: List[Any],
        game_buffers: List[GameBuffer],
        alpha: float = 1.0,
        clip_scale: int = 1
) -> List[int]:
    """
    Allocates batch sizes inversely proportional to the number of collected episodes for each task.
    Dynamically adjusts batch size within a specified range to enhance training stability and efficiency.

    Args:
        cfgs (List[Any]): List of configurations for each task.
        game_buffers (List[GameBuffer]): List of replay buffer instances for each task.
        alpha (float): The hyperparameter controlling the degree of inverse proportionality. Default is 1.0.
        clip_scale (int): The scaling factor to clip the batch size. Default is 1.

    Returns:
        List[int]: A list of allocated batch sizes for each task.
    """
    # 提取每个任务的 num_of_collected_episodes
    buffer_num_of_collected_episodes = [
        buffer.num_of_collected_episodes for buffer in game_buffers
    ]

    # 获取当前的 world_size 和 rank
    world_size = get_world_size()
    rank = get_rank()

    # 收集所有 rank 的 num_of_collected_episodes 列表
    all_task_num_of_collected_episodes = [None for _ in range(world_size)]
    dist.all_gather_object(all_task_num_of_collected_episodes, buffer_num_of_collected_episodes)

    # 将所有 rank 的 num_of_collected_episodes 拼接成一个大列表
    all_task_num_of_collected_episodes = [
        item for sublist in all_task_num_of_collected_episodes for item in sublist
    ]
    if rank == 0:
        print(f'all_task_num_of_collected_episodes: {all_task_num_of_collected_episodes}')

    # 计算每个任务的反比权重
    inv_episodes = np.array([
        1.0 / (episodes + 1) for episodes in all_task_num_of_collected_episodes
    ])
    inv_sum = np.sum(inv_episodes)

    # 计算总的 batch_size (所有任务 cfg.policy.batch_size 的和)
    total_batch_size = cfgs[0].policy.total_batch_size

    # 动态调整的部分：最小和最大的 batch_size 范围
    avg_batch_size = total_batch_size / world_size
    min_batch_size = avg_batch_size / clip_scale
    max_batch_size = avg_batch_size * clip_scale

    # 动态调整 alpha，让 batch_size 的变化更加平滑
    task_weights = (inv_episodes / inv_sum) ** alpha
    batch_sizes = total_batch_size * task_weights

    # 控制 batch_size 在 [min_batch_size, max_batch_size] 之间
    batch_sizes = np.clip(batch_sizes, min_batch_size, max_batch_size)

    # 确保 batch_size 是整数
    batch_sizes = [int(size) for size in batch_sizes]

    # 返回最终分配的 batch_size 列表
    return batch_sizes


def train_unizero_multitask_segment_eval(
        input_cfg_list: List[Tuple[int, Tuple[Dict[str, Any], Dict[str, Any]]]],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':
    """
    Overview:
        The training entry point for UniZero, as proposed in the paper "UniZero: Generalized and Efficient Planning with Scalable Latent World Models".
        UniZero aims to enhance the planning capabilities of reinforcement learning agents by addressing limitations found in MuZero-style algorithms,
        particularly in environments requiring the capture of long-term dependencies. More details can be found in https://arxiv.org/abs/2406.10667.

    Args:
        input_cfg_list (List[Tuple[int, Tuple[Dict[str, Any], Dict[str, Any]]]]):
            List of configurations for different tasks. Each item is a tuple containing a task ID and a tuple of configuration dictionaries.
        seed (int):
            Random seed for reproducibility.
        model (Optional[torch.nn.Module]):
            Instance of torch.nn.Module representing the model. If None, a new model will be created.
        model_path (Optional[str]):
            Path to a pretrained model checkpoint. Should point to the ckpt file of the pretrained model.
        max_train_iter (Optional[int]):
            Maximum number of policy update iterations during training. Default is a very large number.
        max_env_step (Optional[int]):
            Maximum number of environment interaction steps to collect. Default is a very large number.

    Returns:
        'Policy':
            The converged policy after training.
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
        cfgs, game_buffers, collectors, evaluators = [], [], [], []
    else:
        print(f"Rank {rank}/{world_size}, handling tasks {start_idx} to {end_idx - 1}")

        cfgs: List[Any] = []
        game_buffers: List[GameBuffer] = []
        collectors: List[Collector] = []
        evaluators: List[Evaluator] = []

        # 使用本rank的第一个任务的配置来创建共享的 policy
        task_id, (cfg, create_cfg) = tasks_for_this_rank[0]

        # 设置每个任务的 task_num 以用于 learner_log
        for config in tasks_for_this_rank:
            config[1][0].policy.task_num = tasks_per_rank

        # 确保指定的 policy 类型是支持的
        assert create_cfg.policy.type in [
            'unizero_multitask'], "train_unizero entry now only supports 'unizero_multitask'"

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
        log_dir = os.path.join('./{}/log'.format(cfg.exp_name), f'serial_rank_{rank}')
        tb_logger = SummaryWriter(log_dir)

        # 创建共享的 learner
        learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

        policy_config = cfg.policy
        batch_size = policy_config.batch_size[0]

        # 只处理当前进程分配到的任务
        for local_task_id, (task_id, (cfg, create_cfg)) in enumerate(tasks_for_this_rank):
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
            collectors.append(collector)
            evaluators.append(evaluator)

    learner.call_hook('before_run')
    buffer_reanalyze_count = 0
    train_epoch = 0
    reanalyze_batch_size = cfg.policy.reanalyze_batch_size
    update_per_collect = cfg.policy.update_per_collect

    while True:
        # 预先计算位置嵌入矩阵（如果需要）
        # policy._collect_model.world_model.precompute_pos_emb_diff_kv()
        # policy._target_model.world_model.precompute_pos_emb_diff_kv()

        if cfg.policy.allocated_batch_sizes:
            # 动态调整 clip_scale 随着 train_epoch 从 0 增加到 1000, clip_scale 从 1 线性增加到 4
            clip_scale = np.clip(1 + (3 * train_epoch / 1000), 1, 4)
            allocated_batch_sizes = allocate_batch_size(cfgs, game_buffers, alpha=1.0, clip_scale=clip_scale)
            if rank == 0:
                print("分配后的 batch_sizes: ", allocated_batch_sizes)
            for cfg, _collector, _evaluator, replay_buffer in zip(cfgs, collectors, evaluators, game_buffers):
                cfg.policy.batch_size = allocated_batch_sizes
                policy._cfg.batch_size = allocated_batch_sizes

        # 对于当前进程的每个任务，进行数据收集和评估
        for cfg, collector, evaluator, replay_buffer in zip(cfgs, collectors, evaluators, game_buffers):
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

            if evaluator.should_eval(learner.train_iter):
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

            # NOTE: 在每次收集之前重置初始数据，这对于多任务设置非常重要
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
                if (train_epoch % int(1 / cfg.policy.buffer_reanalyze_freq) == 0 and
                        replay_buffer.get_num_of_transitions() // cfg.policy.num_unroll_steps >
                        int(reanalyze_batch_size / cfg.policy.reanalyze_partition)):
                    with EasyTimer() as timer:
                        # 每个重新分析过程将重新分析 <reanalyze_batch_size> 个序列
                        replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                    buffer_reanalyze_count += 1
                    logging.info(f'Buffer reanalyze count: {buffer_reanalyze_count}')
                    logging.info(f'Buffer reanalyze time: {timer.value}')

            # 数据收集结束后添加日志
            logging.info(f'Rank {rank}: Completed data collection for task {cfg.policy.task_id}')

        # 检查是否有足够的数据进行训练
        not_enough_data = any(
            replay_buffer.get_num_of_transitions() < cfgs[0].policy.total_batch_size / world_size
            for replay_buffer in game_buffers
        )

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
                for cfg, collector, replay_buffer in zip(cfgs, collectors, game_buffers):
                    envstep_multi_task += collector.envstep
                    batch_size = cfg.policy.batch_size[cfg.policy.task_id]
                    if replay_buffer.get_num_of_transitions() > batch_size:
                        if cfg.policy.buffer_reanalyze_freq >= 1:
                            # 在一个训练 epoch 中重新分析缓冲区 <buffer_reanalyze_freq> 次
                            if (i % reanalyze_interval == 0 and
                                    replay_buffer.get_num_of_transitions() // cfg.policy.num_unroll_steps >
                                    int(reanalyze_batch_size / cfg.policy.reanalyze_partition)):
                                with EasyTimer() as timer:
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

                # 同步训练前所有 rank 的准备状态
                try:
                    dist.barrier()
                    logging.info(f'Rank {rank}: Passed barrier during training')
                except Exception as e:
                    logging.error(f'Rank {rank}: Barrier failed with error {e}')
                    break  # 或者进行其他错误处理

                # TODO: 可选：终止进程
                import sys
                sys.exit(0)

        train_epoch += 1
        policy.recompute_pos_emb_diff_and_clear_cache()

        # 同步所有 Rank，确保所有 Rank 都完成了训练
        try:
            dist.barrier()
            logging.info(f'Rank {rank}: Passed barrier after training')
        except Exception as e:
            logging.error(f'Rank {rank}: Barrier failed with error {e}')
            break  # 或者进行其他错误处理

        # 检查是否需要终止训练
        try:
            # 收集本地的 envsteps
            local_envsteps = [collector.envstep for collector in collectors]

            # 收集所有进程的 envsteps
            total_envsteps: List[Optional[int]] = [None for _ in range(world_size)]
            dist.all_gather_object(total_envsteps, local_envsteps)

            # 将所有 envsteps 拼接在一起进行检查
            all_envsteps = torch.cat([
                torch.tensor(envsteps, device=cfg.policy.device) for envsteps in total_envsteps
            ])
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
        except Exception as e:
            logging.error(f'Rank {rank}: Termination check failed with error {e}')
            break  # 或者进行其他错误处理

    learner.call_hook('after_run')
    return policy