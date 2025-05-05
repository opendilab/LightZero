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

from lzero.entry.utils import log_buffer_memory_usage, TemperatureScheduler
from lzero.policy import visit_count_temperature
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.worker import MuZeroSegmentCollector as Collector
from ding.utils import EasyTimer
import torch.nn.functional as F

import torch.distributed as dist

import concurrent.futures
# ====== UniZero-MT 归一化所需基准分数 (26 Atari100k task_id 对应索引) ======
RANDOM_SCORES = np.array([
    227.8, 5.8, 222.4, 210.0, 14.2, 2360.0, 0.1, 1.7, 811.0, 10780.5,
    152.1, 0.0, 65.2, 257.6, 1027.0, 29.0, 52.0, 1598.0, 258.5, 307.3,
    -20.7, 24.9, 163.9, 11.5, 68.4, 533.4
])
HUMAN_SCORES = np.array([
    7127.7, 1719.5, 742.0, 8503.3, 753.1, 37187.5, 12.1, 30.5, 7387.8, 35829.4,
    1971.0, 29.6, 4334.7, 2412.5, 30826.4, 302.8, 3035.0, 2665.5, 22736.3, 6951.6,
    14.6, 69571.3, 13455.0, 7845.0, 42054.7, 11693.2
])

# 保存最近一次评估回报：{task_id: eval_episode_return_mean}
from collections import defaultdict
GLOBAL_EVAL_RETURNS: dict[int, float] = defaultdict(lambda: None)
def compute_unizero_mt_normalized_stats(
        eval_returns: dict[int, float]
) -> tuple[Optional[float], Optional[float]]:
    """
    由 eval_returns 计算 Human-Normalized Mean 和 Median。
    若暂无样本，返回 (None, None)。
    """
    normalized = []
    for tid, ret in eval_returns.items():
        if ret is None:
            continue
        denom = HUMAN_SCORES[tid] - RANDOM_SCORES[tid]
        if denom == 0:
            continue
        normalized.append((ret - RANDOM_SCORES[tid]) / denom)

    if not normalized:
        return None, None
    arr = np.asarray(normalized, dtype=np.float32)
    return float(arr.mean()), float(np.median(arr))

# 设置超时时间 (秒)
TIMEOUT = 12000  # 例如200分钟

timer = EasyTimer()


def safe_eval(
        evaluator: Evaluator,
        learner: BaseLearner,
        collector: Collector,
        rank: int,
        world_size: int
) -> Tuple[Optional[bool], Optional[float]]:
    """
    Safely执行评估任务，避免超时。

    Args:
        evaluator (Evaluator): 评估器实例。
        learner (BaseLearner): 学习器实例。
        collector (Collector): 数据收集器实例。
        rank (int): 当前进程的rank。
        world_size (int): 总进程数。

    Returns:
        Tuple[Optional[bool], Optional[float]]: 如果评估成功，返回停止标志和奖励，否则返回（None, None）。
    """
    try:
        print(f"=========评估开始 Rank {rank}/{world_size}===========")
        # 重置 stop_event，确保每次评估前都处于未设置状态
        evaluator.stop_event.clear()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 提交评估任务
            future = executor.submit(evaluator.eval, learner.save_checkpoint, learner.train_iter, collector.envstep)
            try:
                stop, reward = future.result(timeout=TIMEOUT)
            except concurrent.futures.TimeoutError:
                # 超时，设置 stop_event
                evaluator.stop_event.set()
                print(f"评估操作在 Rank {rank}/{world_size} 上超时，耗时 {TIMEOUT} 秒。")
                return None, None

        print(f"======评估结束 Rank {rank}/{world_size}======")
        return stop, reward
    except Exception as e:
        print(f"Rank {rank}/{world_size} 评估过程中发生错误: {e}")
        return None, None


def allocate_batch_size(
        cfgs: List[dict],
        game_buffers,
        alpha: float = 1.0,
        clip_scale: int = 1
) -> List[int]:
    """
    根据不同任务的收集剧集数反比分配batch_size，并动态调整batch_size范围以提高训练稳定性和效率。

    Args:
        cfgs (List[dict]): 每个任务的配置列表。
        game_buffers (List[GameBuffer]): 每个任务的重放缓冲区实例列表。
        alpha (float, optional): 控制反比程度的超参数。默认为1.0。
        clip_scale (int, optional): 动态调整的clip比例。默认为1。

    Returns:
        List[int]: 分配后的batch_size列表。
    """
    # 提取每个任务的 collected episodes 数量
    buffer_num_of_collected_episodes = [buffer.num_of_collected_episodes for buffer in game_buffers]

    # 获取当前的 world_size 和 rank
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # 收集所有 rank 的 collected episodes 列表
    all_task_num_of_collected_episodes = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(all_task_num_of_collected_episodes, buffer_num_of_collected_episodes)

    # 将所有 rank 的 collected episodes 合并为一个大列表
    all_task_num_of_collected_episodes = [
        episode for sublist in all_task_num_of_collected_episodes for episode in sublist
    ]
    if rank == 0:
        print(f'所有任务的 collected episodes: {all_task_num_of_collected_episodes}')

    # 计算每个任务的反比权重
    inv_episodes = np.array([1.0 / (episodes + 1) for episodes in all_task_num_of_collected_episodes])
    inv_sum = np.sum(inv_episodes)

    # 计算总的batch_size (所有任务 cfg.policy.batch_size 的和)
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

    return batch_sizes

import numpy as np


def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Symlog 归一化，减少目标值的幅度差异。
    symlog(x) = sign(x) * log(|x| + 1)
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1)

def inv_symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Symlog 的逆操作，用于恢复原始值。
    inv_symlog(x) = sign(x) * (exp(|x|) - 1)
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

# 全局最大值和最小值（用于 "run-max-min"）
GLOBAL_MAX = -float('inf')
GLOBAL_MIN = float('inf')

def compute_task_weights(
    task_returns: dict,
    option: str = "symlog",
    epsilon: float = 1e-6,
    temperature: float = 1.0,
    use_softmax: bool = False,  # 是否使用 Softmax
    reverse: bool = False,  # 正比 (False) 或反比 (True)
    clip_min: float = 1e-2,  # 权重的最小值
    clip_max: float = 1.0,  # 权重的最大值
) -> dict:
    """
    改进后的任务权重计算函数，支持多种标准化方式、Softmax 和正反比权重计算，并增加权重范围裁剪功能。

    Args:
        task_returns (dict): 每个任务的字典，键为 task_id，值为评估奖励或损失。
        option (str): 标准化方式，可选值为 "symlog", "max-min", "run-max-min", "rank", "none"。
        epsilon (float): 避免分母为零的小值。
        temperature (float): 控制权重分布的温度系数。
        use_softmax (bool): 是否使用 Softmax 进行权重分配。
        reverse (bool): 若为 True，权重与值反比；若为 False，权重与值正比。
        clip_min (float): 权重的最小值，用于裁剪。
        clip_max (float): 权重的最大值，用于裁剪。

    Returns:
        dict: 每个任务的权重，键为 task_id，值为归一化后的权重。
    """
    import torch
    import torch.nn.functional as F

    global GLOBAL_MAX, GLOBAL_MIN

    # 如果输入为空字典，直接返回空结果
    if not task_returns:
        return {}

    # Step 1: 对 task_returns 的值构造张量
    task_ids = list(task_returns.keys())
    returns_tensor = torch.tensor(list(task_returns.values()), dtype=torch.float32)

    if option == "symlog":
        # 使用 symlog 标准化
        scaled_returns = symlog(returns_tensor)
    elif option == "max-min":
        # 使用最大最小值归一化
        max_reward = returns_tensor.max().item()
        min_reward = returns_tensor.min().item()
        scaled_returns = (returns_tensor - min_reward) / (max_reward - min_reward + epsilon)
    elif option == "run-max-min":
        # 使用全局最大最小值归一化
        GLOBAL_MAX = max(GLOBAL_MAX, returns_tensor.max().item())
        GLOBAL_MIN = min(GLOBAL_MIN, returns_tensor.min().item())
        scaled_returns = (returns_tensor - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN + epsilon)
    elif option == "rank":
        # 使用 rank 标准化
        # Rank 是基于值大小的排名，1 表示最小值，越大排名越高
        sorted_indices = torch.argsort(returns_tensor)
        scaled_returns = torch.empty_like(returns_tensor)
        rank_values = torch.arange(1, len(returns_tensor) + 1, dtype=torch.float32)  # 1 到 N
        scaled_returns[sorted_indices] = rank_values
    elif option == "none":
        # 不进行标准化
        scaled_returns = returns_tensor
    else:
        raise ValueError(f"Unsupported option: {option}")

    # Step 2: 根据 reverse 确定权重是正比还是反比
    if not reverse:
        # 正比：权重与值正相关
        raw_weights = scaled_returns
    else:
        # 反比：权重与值负相关
        # 避免 scaled_returns 为负数或零
        scaled_returns = torch.clamp(scaled_returns, min=epsilon)
        raw_weights = 1.0 / scaled_returns

    # Step 3: 根据是否使用 Softmax 进行权重计算
    if use_softmax:
        # 使用 Softmax 进行权重分配
        beta = 1.0 / max(temperature, epsilon)  # 确保 temperature 不为零
        logits = -beta * raw_weights
        softmax_weights = F.softmax(logits, dim=0).numpy()
        weights = dict(zip(task_ids, softmax_weights))
    else:
        # 不使用 Softmax，直接计算权重
        # 温度缩放
        scaled_weights = raw_weights ** (1 / max(temperature, epsilon))  # 确保温度不为零

        # 归一化权重
        total_weight = scaled_weights.sum()
        normalized_weights = scaled_weights / total_weight

        # 转换为字典
        weights = dict(zip(task_ids, normalized_weights.numpy()))

    # Step 4: Clip 权重范围
    for task_id in weights:
        weights[task_id] = max(min(weights[task_id], clip_max), clip_min)

    return weights

def train_unizero_multitask_segment_ddp(
        input_cfg_list: List[Tuple[int, Tuple[dict, dict]]],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':
    """
    Overview:
        UniZero的训练入口，旨在通过解决MuZero类算法在需要捕捉长期依赖环境中的局限性，提高强化学习代理的规划能力。
        详细信息请参阅 https://arxiv.org/abs/2406.10667。

    Args:
        - input_cfg_list (:obj:`List[Tuple[int, Tuple[dict, dict]]]`): 不同任务的配置列表。
        - seed (:obj:`int`): 随机种子。
        - model (:obj:`Optional[torch.nn.Module]`): torch.nn.Module实例。
        - model_path (:obj:`Optional[str]`): 预训练模型路径，应指向预训练模型的ckpt文件。
        - max_train_iter (:obj:`Optional[int]`): 训练中的最大策略更新迭代次数。
        - max_env_step (:obj:`Optional[int]`): 最大收集环境交互步数。

    Returns:
        - policy (:obj:`Policy`): 收敛的策略。
    """
    # 初始化温度调度器
    initial_temperature = 10.0
    final_temperature = 1.0
    threshold_steps = int(1e4)  # 训练步数达到 10k 时，温度降至 1.0
    temperature_scheduler = TemperatureScheduler(
        initial_temp=initial_temperature,
        final_temp=final_temperature,
        threshold_steps=threshold_steps,
        mode='linear'  # 或 'exponential'
    )

    # 获取当前进程的rank和总进程数
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
        logging.warning(f"Rank {rank}: 未分配任务，继续执行。")
        # 初始化空列表以避免后续代码报错
        cfgs, game_buffers, collector_envs, evaluator_envs, collectors, evaluators = [], [], [], [], [], []
    else:
        print(f"Rank {rank}/{world_size}, 处理任务 {start_idx} 到 {end_idx - 1}")

    cfgs = []
    game_buffers = []
    collector_envs = []
    evaluator_envs = []
    collectors = []
    evaluators = []

    if tasks_for_this_rank:
        # 使用第一个任务的配置创建共享的policy
        task_id, [cfg, create_cfg] = tasks_for_this_rank[0]

        for config in tasks_for_this_rank:
            config[1][0].policy.task_num = tasks_per_rank

        # 确保指定的策略类型受支持
        assert create_cfg.policy.type in ['unizero_multitask',
                                          'sampled_unizero_multitask'], "train_unizero entry 目前仅支持 'unizero_multitask'"

        if create_cfg.policy.type == 'unizero_multitask':
            from lzero.mcts import UniZeroGameBuffer as GameBuffer
        if create_cfg.policy.type == 'sampled_unizero_multitask':
            from lzero.mcts import SampledUniZeroGameBuffer as GameBuffer


        # 根据CUDA可用性设置设备
        cfg.policy.device = cfg.policy.model.world_model_cfg.device if torch.cuda.is_available() else 'cpu'
        logging.info(f'配置的设备: {cfg.policy.device}')

        # 编译配置
        cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
        # 创建共享的policy
        policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

        # 加载预训练模型（如果提供）
        if model_path is not None:
            logging.info(f'开始加载模型: {model_path}')
            policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))
            logging.info(f'完成加载模型: {model_path}')

        # 创建TensorBoard日志记录器
        log_dir = os.path.join('./{}/log'.format(cfg.exp_name), f'serial_rank_{rank}')
        tb_logger = SummaryWriter(log_dir)

        # 创建共享的learner
        learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

        policy_config = cfg.policy

        # 处理当前进程分配到的每个任务
        for local_task_id, (task_id, [cfg, create_cfg]) in enumerate(tasks_for_this_rank):
            # 设置每个任务的随机种子
            cfg.policy.device = 'cuda' if cfg.policy.cuda and torch.cuda.is_available() else 'cpu'
            cfg = compile_config(cfg, seed=seed + task_id, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
            policy_config = cfg.policy
            policy.collect_mode.get_attribute('cfg').n_episode = policy_config.n_episode
            policy.eval_mode.get_attribute('cfg').n_episode = policy_config.n_episode

            # 创建环境
            env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
            collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
            evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
            collector_env.seed(cfg.seed + task_id)
            evaluator_env.seed(cfg.seed + task_id, dynamic_seed=False)
            set_pkg_seed(cfg.seed + task_id, use_cuda=cfg.policy.cuda)

            # 创建不同的game buffer、collector和evaluator
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

    # 调用learner的before_run钩子
    learner.call_hook('before_run')
    value_priority_tasks = {}

    buffer_reanalyze_count = 0
    train_epoch = 0
    reanalyze_batch_size = cfg.policy.reanalyze_batch_size
    update_per_collect = cfg.policy.update_per_collect

    # use_task_exploitation_weight = cfg.policy.use_task_exploitation_weight
    task_exploitation_weight = None

    # 创建任务奖励字典
    task_returns = {}  # {task_id: reward}

    while True:
        # 动态调整batch_size
        if cfg.policy.allocated_batch_sizes:
            clip_scale = np.clip(1 + (3 * train_epoch / 1000), 1, 4)
            allocated_batch_sizes = allocate_batch_size(cfgs, game_buffers, alpha=1.0, clip_scale=clip_scale)
            if rank == 0:
                print("分配后的 batch_sizes: ", allocated_batch_sizes)
            for idx, (cfg, collector, evaluator, replay_buffer) in enumerate(
                    zip(cfgs, collectors, evaluators, game_buffers)):
                cfg.policy.batch_size = allocated_batch_sizes
                policy._cfg.batch_size = allocated_batch_sizes

        # 对于当前进程的每个任务，进行数据收集和评估
        for idx, (cfg, collector, evaluator, replay_buffer) in enumerate(
                zip(cfgs, collectors, evaluators, game_buffers)):

            # 记录缓冲区内存使用情况
            log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger, cfg.policy.task_id)

            collect_kwargs = {
                'temperature': visit_count_temperature(
                    policy_config.manual_temperature_decay,
                    policy_config.fixed_temperature_value,
                    policy_config.threshold_training_steps_for_final_temperature,
                    trained_steps=learner.train_iter
                ),
                'epsilon': 0.0  # 默认的epsilon值
            }

            if policy_config.eps.eps_greedy_exploration_in_collect:
                epsilon_greedy_fn = get_epsilon_greedy_fn(
                    start=policy_config.eps.start,
                    end=policy_config.eps.end,
                    decay=policy_config.eps.decay,
                    type_=policy_config.eps.type
                )
                collect_kwargs['epsilon'] = epsilon_greedy_fn(collector.envstep)

            # 判断是否需要进行评估
            if learner.train_iter == 0 or evaluator.should_eval(learner.train_iter):
            # if learner.train_iter > 10 and evaluator.should_eval(learner.train_iter): # only for debug
            # if evaluator.should_eval(learner.train_iter):
                print('=' * 20)
                print(f'Rank {rank} 评估任务_id: {cfg.policy.task_id}...')

                # =========TODO=========
                evaluator._policy.reset(reset_init_data=True, task_id=cfg.policy.task_id)

                # 执行安全评估
                stop, reward = safe_eval(evaluator, learner, collector, rank, world_size)
                # 判断评估是否成功
                if stop is None or reward is None:
                    print(f"Rank {rank} 在评估过程中遇到问题，继续训练...")
                    task_returns[cfg.policy.task_id] = float('inf')  # 如果评估失败，将任务难度设为最大值
                else:
                    # 确保从评估结果中提取 `eval_episode_return_mean` 作为奖励值
                    try:
                        eval_mean_reward = reward.get('eval_episode_return_mean', float('inf'))
                        print(f"任务 {cfg.policy.task_id} 的评估奖励: {eval_mean_reward}")
                        task_returns[cfg.policy.task_id] = eval_mean_reward
                    except Exception as e:
                        print(f"提取评估奖励时发生错误: {e}")
                        task_returns[cfg.policy.task_id] = float('inf')  # 出现问题时，将奖励设为最大值


            print('=' * 20)
            print(f'开始收集 Rank {rank} 的任务_id: {cfg.policy.task_id}...')
            print(f'Rank {rank}: cfg.policy.task_id={cfg.policy.task_id} ')


            while replay_buffer.get_num_of_transitions() < cfg.policy.batch_size[cfg.policy.task_id]:
                # for ddp training, 避免后面 train 时replay buffer中样本小于batch size 导致ddp hangs

                # 在每次收集之前重置初始数据，这对于多任务设置非常重要
                collector._policy.reset(reset_init_data=True, task_id=cfg.policy.task_id)
                # 收集数据
                new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)

                # 更新重放缓冲区
                replay_buffer.push_game_segments(new_data)
                replay_buffer.remove_oldest_data_to_fit()

            # # 在每次收集之前重置初始数据，这对于多任务设置非常重要
            # collector._policy.reset(reset_init_data=True, task_id=cfg.policy.task_id)
            # # 收集数据
            # new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
            # # 更新重放缓冲区
            # replay_buffer.push_game_segments(new_data)
            # replay_buffer.remove_oldest_data_to_fit()

            # # ===== only for debug =====
            # if train_epoch > 2:
            #     with timer:
            #         replay_buffer.reanalyze_buffer(2, policy)
            #     buffer_reanalyze_count += 1
            #     logging.info(f'缓冲区重新分析次数: {buffer_reanalyze_count}')
            #     logging.info(f'缓冲区重新分析耗时: {timer.value}') 
            # # ===== only for debug =====


            # 周期性地重新分析缓冲区
            if cfg.policy.buffer_reanalyze_freq >= 1:
                reanalyze_interval = update_per_collect // cfg.policy.buffer_reanalyze_freq
            else:
                if train_epoch > 0 and train_epoch % int(1 / cfg.policy.buffer_reanalyze_freq) == 0 and \
                        replay_buffer.get_num_of_transitions() // cfg.policy.num_unroll_steps > int(
                    reanalyze_batch_size / cfg.policy.reanalyze_partition):
                    with timer:
                        replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                    buffer_reanalyze_count += 1
                    logging.info(f'缓冲区重新分析次数: {buffer_reanalyze_count}')
                    logging.info(f'缓冲区重新分析耗时: {timer.value}')

            # 数据收集结束后添加日志
            logging.info(f'Rank {rank}: 完成任务 {cfg.policy.task_id} 的数据收集')

        # 检查是否有足够的数据进行训练
        not_enough_data = any(
            replay_buffer.get_num_of_transitions() < cfgs[0].policy.total_batch_size / world_size
            for replay_buffer in game_buffers
        )

        # 获取当前温度
        current_temperature_task_weight = temperature_scheduler.get_temperature(learner.train_iter)

        # 计算任务权重
        try:
            # 汇聚任务奖励
            dist.barrier()
            # if cfg.policy.task_complexity_weight:
            all_task_returns = [None for _ in range(world_size)]
            dist.all_gather_object(all_task_returns, task_returns)
            # 合并任务奖励
            merged_task_returns = {}
            for returns in all_task_returns:
                if returns:
                    merged_task_returns.update(returns)
            
            logging.warning(f"Rank {rank}: merged_task_returns: {merged_task_returns}")

            # 计算全局任务权重
            task_weights = compute_task_weights(merged_task_returns, temperature=current_temperature_task_weight)
            
            # ---------- 维护 UniZero-MT 全局评估结果 ----------
            for tid, ret in merged_task_returns.items():
                GLOBAL_EVAL_RETURNS[tid] = ret   # solved 的任务同样更新

            # 计算 Human-Normalized Mean / Median
            uni_mean, uni_median = compute_unizero_mt_normalized_stats(GLOBAL_EVAL_RETURNS)

            if uni_mean is not None:            # 至少评估过 1 个任务
                if rank == 0:                   # 仅在 rank0 写 TensorBoard，防止重复
                    tb_logger.add_scalar('UniZero-MT/NormalizedMean',   uni_mean,   global_step=learner.train_iter)
                    tb_logger.add_scalar('UniZero-MT/NormalizedMedian', uni_median, global_step=learner.train_iter)
                logging.info(f"Rank {rank}: UniZero-MT Norm Mean={uni_mean:.4f}, Median={uni_median:.4f}")
            else:
                logging.info(f"Rank {rank}: 暂无数据计算 UniZero-MT 归一化指标")

            # 同步任务权重
            dist.broadcast_object_list([task_weights], src=0)
            print(f"rank{rank}, 全局任务权重 (按 task_id 排列): {task_weights}")
            # else:
            #     task_weights = None
        except Exception as e:
            logging.error(f'Rank {rank}: 同步任务权重失败，错误: {e}')
            break


        # 学习策略
        if not not_enough_data:
            for i in range(update_per_collect):
                train_data_multi_task = []
                envstep_multi_task = 0
                for idx, (cfg, collector, replay_buffer) in enumerate(zip(cfgs, collectors, game_buffers)):
                    envstep_multi_task += collector.envstep
                    batch_size = cfg.policy.batch_size[cfg.policy.task_id]
                    if replay_buffer.get_num_of_transitions() > batch_size:
                        if cfg.policy.buffer_reanalyze_freq >= 1:
                            if i % reanalyze_interval == 0 and \
                                    replay_buffer.get_num_of_transitions() // cfg.policy.num_unroll_steps > int(
                                reanalyze_batch_size / cfg.policy.reanalyze_partition):
                                with timer:
                                    replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                                buffer_reanalyze_count += 1
                                logging.info(f'缓冲区重新分析次数: {buffer_reanalyze_count}')
                                logging.info(f'缓冲区重新分析耗时: {timer.value}')

                        train_data = replay_buffer.sample(batch_size, policy)
                        train_data.append(cfg.policy.task_id)  # 追加task_id以区分任务
                        train_data_multi_task.append(train_data)
                    else:
                        logging.warning(
                            f'重放缓冲区中的数据不足以采样mini-batch: '
                            f'batch_size: {batch_size}, replay_buffer: {replay_buffer}'
                        )
                        break

                if train_data_multi_task:
                    # learn_kwargs = {'task_exploitation_weight':task_exploitation_weight, 'task_weights':task_weights, }
                    learn_kwargs = {'task_weights': task_weights, }
                    # learn_kwargs = {'task_weights':task_exploitation_weight}

                    # 在训练时，DDP会自动同步梯度和参数
                    log_vars = learner.train(train_data_multi_task, envstep_multi_task, policy_kwargs=learn_kwargs)

                    # 判断是否需要计算task_exploitation_weight
                    if i == 0:
                        # 计算任务权重
                        try:
                            dist.barrier()  # 等待所有进程同步
                            if cfg.policy.use_task_exploitation_weight: # use obs loss now, new polish
                                # 收集所有任务的 obs_loss
                                all_obs_loss = [None for _ in range(world_size)]
                                # 构建当前进程的任务 obs_loss 数据
                                merged_obs_loss_task = {}
                                for cfg, replay_buffer in zip(cfgs, game_buffers):
                                    task_id = cfg.policy.task_id
                                    if f'noreduce_obs_loss_task{task_id}' in log_vars[0]:
                                        merged_obs_loss_task[task_id] = log_vars[0][f'noreduce_obs_loss_task{task_id}']
                                # 汇聚所有进程的 obs_loss 数据
                                dist.all_gather_object(all_obs_loss, merged_obs_loss_task)
                                # 合并所有进程的 obs_loss 数据
                                global_obs_loss_task = {}
                                for obs_loss_task in all_obs_loss:
                                    if obs_loss_task:
                                        global_obs_loss_task.update(obs_loss_task)
                                # 计算全局任务权重
                                if global_obs_loss_task:
                                    task_exploitation_weight = compute_task_weights(
                                        global_obs_loss_task,
                                        option="rank",
                                        # temperature=current_temperature_task_weight # TODO
                                        temperature=1,
                                    )
                                    # 广播任务权重到所有进程
                                    dist.broadcast_object_list([task_exploitation_weight], src=0)
                                    print(f"rank{rank}, task_exploitation_weight (按 task_id 排列): {task_exploitation_weight}")
                                else:
                                    logging.warning(f"Rank {rank}: 未能计算全局 obs_loss 任务权重，obs_loss 数据为空。")
                                    task_exploitation_weight = None
                            else:
                                task_exploitation_weight = None
                            # 更新训练参数，使其包含计算后的任务权重
                            learn_kwargs['task_weight'] = task_exploitation_weight
                        except Exception as e:
                            logging.error(f'Rank {rank}: 同步任务权重失败，错误: {e}')
                            raise e  # 保留异常抛出，便于外部捕获和分析



                    if cfg.policy.use_priority:
                        for idx, (cfg, replay_buffer) in enumerate(zip(cfgs, game_buffers)):
                            # 更新任务特定的重放缓冲区优先级
                            task_id = cfg.policy.task_id
                            replay_buffer.update_priority(
                                train_data_multi_task[idx],
                                log_vars[0][f'value_priority_task{task_id}']
                            )

                            current_priorities = log_vars[0][f'value_priority_task{task_id}']
                            mean_priority = np.mean(current_priorities)
                            std_priority = np.std(current_priorities)

                            alpha = 0.1  # 平滑因子
                            if f'running_mean_priority_task{task_id}' not in value_priority_tasks:
                                value_priority_tasks[f'running_mean_priority_task{task_id}'] = mean_priority
                            else:
                                value_priority_tasks[f'running_mean_priority_task{task_id}'] = (
                                        alpha * mean_priority +
                                        (1 - alpha) * value_priority_tasks[f'running_mean_priority_task{task_id}']
                                )

                            # 使用运行均值计算归一化的优先级
                            running_mean_priority = value_priority_tasks[f'running_mean_priority_task{task_id}']
                            normalized_priorities = (current_priorities - running_mean_priority) / (std_priority + 1e-6)

                            # 如果需要，可以将归一化的优先级存储回重放缓冲区
                            # replay_buffer.update_priority(train_data_multi_task[idx], normalized_priorities)

                            # 记录优先级统计信息
                            if cfg.policy.print_task_priority_logs:
                                print(f"任务 {task_id} - 平均优先级: {mean_priority:.8f}, "
                                      f"运行平均优先级: {running_mean_priority:.8f}, "
                                      f"标准差: {std_priority:.8f}")

        train_epoch += 1
        policy.recompute_pos_emb_diff_and_clear_cache()

        # 同步所有Rank，确保所有Rank完成训练
        try:
            dist.barrier()
            logging.info(f'Rank {rank}: 通过训练后的同步障碍')
        except Exception as e:
            logging.error(f'Rank {rank}: 同步障碍失败，错误: {e}')
            break

        # 检查是否需要终止训练
        try:
            local_envsteps = [collector.envstep for collector in collectors]
            total_envsteps = [None for _ in range(world_size)]
            dist.all_gather_object(total_envsteps, local_envsteps)

            all_envsteps = torch.cat([torch.tensor(envsteps, device=cfg.policy.device) for envsteps in total_envsteps])
            max_envstep_reached = torch.all(all_envsteps >= max_env_step)

            # 收集所有进程的train_iter
            global_train_iter = torch.tensor([learner.train_iter], device=cfg.policy.device)
            all_train_iters = [torch.zeros_like(global_train_iter) for _ in range(world_size)]
            dist.all_gather(all_train_iters, global_train_iter)

            max_train_iter_reached = torch.any(torch.stack(all_train_iters) >= max_train_iter)

            if max_envstep_reached.item() or max_train_iter_reached.item():
                logging.info(f'Rank {rank}: 达到终止条件')
                dist.barrier()  # 确保所有进程同步
                break
        except Exception as e:
            logging.error(f'Rank {rank}: 终止检查失败，错误: {e}')
            break

    # 调用learner的after_run钩子
    learner.call_hook('after_run')
    return policy