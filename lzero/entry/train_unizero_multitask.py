import logging
import os
from functools import partial
from typing import Tuple, Optional, List, Dict, Any

import torch
import numpy as np
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy, Policy
# from ding.rl_utils import get_epsilon_greedy_fn # get_epsilon_greedy_fn 已被弃用，如果需要需要从 ding.exploration 导入
from ding.utils import set_pkg_seed
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter

from lzero.entry.utils import log_buffer_memory_usage, TemperatureScheduler
from lzero.policy import visit_count_temperature
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.worker import MuZeroCollector as Collector
from ding.utils import EasyTimer
import torch.nn.functional as F

import concurrent.futures

# 设置超时时间 (秒)
TIMEOUT = 12000  # 例如200分钟

timer = EasyTimer()

# --- Helper Functions (Modified for Non-DDP) ---

def safe_eval(
        evaluator: Evaluator,
        learner: BaseLearner,
        collector: Collector
) -> Tuple[Optional[bool], Optional[float]]:
    """
    Safely执行评估任务，避免超时 (非 DDP 版本)。

    Args:
        evaluator (Evaluator): 评估器实例。
        learner (BaseLearner): 学习器实例。
        collector (Collector): 数据收集器实例。

    Returns:
        Tuple[Optional[bool], Optional[float]]: 如果评估成功，返回停止标志和奖励，否则返回（None, None）。
    """
    try:
        print(f"=========评估开始=========")
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
                print(f"评估操作超时，耗时 {TIMEOUT} 秒。")
                return None, None

        print(f"======评估结束======")
        return stop, reward
    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        return None, None

def allocate_batch_size_local(
        cfgs: List[Dict[str, Any]], # 使用 Dict[str, Any] 替代 dict
        game_buffers,
        alpha: float = 1.0,
        clip_scale: int = 1
) -> List[int]:
    """
    根据不同任务的收集剧集数反比分配batch_size (非 DDP 版本)。

    Args:
        cfgs (List[Dict[str, Any]]): 每个任务的配置列表。
        game_buffers (List[Any]): 每个任务的重放缓冲区实例列表 (使用 Any 避免具体类型依赖)。
        alpha (float, optional): 控制反比程度的超参数。默认为1.0。
        clip_scale (int, optional): 动态调整的clip比例。默认为1。

    Returns:
        List[int]: 分配后的batch_size列表。
    """
    # 提取每个任务的 collected episodes 数量 (假设 buffer 有此属性)
    buffer_num_of_collected_episodes = [buffer.num_of_collected_episodes for buffer in game_buffers]
    print(f'所有本地任务的 collected episodes: {buffer_num_of_collected_episodes}')

    # 计算每个任务的反比权重
    inv_episodes = np.array([1.0 / (episodes + 1) for episodes in buffer_num_of_collected_episodes])
    inv_sum = np.sum(inv_episodes)

    # 计算总的batch_size (从第一个任务的配置中获取)
    # 假设 total_batch_size 指的是当前进程需要处理的总批次大小
    total_batch_size = cfgs[0].policy.total_batch_size

    # 动态调整的部分：最小和最大的 batch_size 范围
    num_local_tasks = len(cfgs)
    avg_batch_size = total_batch_size / max(num_local_tasks, 1) # 防止除零
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

# symlog, inv_symlog, GLOBAL_MAX, GLOBAL_MIN 保持不变

# compute_task_weights 保持不变，因为它本身不依赖 DDP，只依赖输入的字典
def compute_task_weights(
    task_rewards: dict,
    option: str = "symlog",
    epsilon: float = 1e-6,
    temperature: float = 1.0,
    use_softmax: bool = False,
    reverse: bool = False,
    clip_min: float = 1e-2,
    clip_max: float = 1.0,
) -> dict:
    """
    改进后的任务权重计算函数 (保持不变，不依赖 DDP)。
    ... (函数体省略，与原版相同) ...
    """
    global GLOBAL_MAX, GLOBAL_MIN

    if not task_rewards:
        return {}

    task_ids = list(task_rewards.keys())
    rewards_tensor = torch.tensor(list(task_rewards.values()), dtype=torch.float32)

    if option == "symlog":
        scaled_rewards = symlog(rewards_tensor)
    elif option == "max-min":
        max_reward = rewards_tensor.max().item()
        min_reward = rewards_tensor.min().item()
        scaled_rewards = (rewards_tensor - min_reward) / (max_reward - min_reward + epsilon)
    elif option == "run-max-min":
        GLOBAL_MAX = max(GLOBAL_MAX, rewards_tensor.max().item())
        GLOBAL_MIN = min(GLOBAL_MIN, rewards_tensor.min().item())
        scaled_rewards = (rewards_tensor - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN + epsilon)
    elif option == "rank":
        sorted_indices = torch.argsort(rewards_tensor)
        scaled_rewards = torch.empty_like(rewards_tensor)
        rank_values = torch.arange(1, len(rewards_tensor) + 1, dtype=torch.float32)
        scaled_rewards[sorted_indices] = rank_values
    elif option == "none":
        scaled_rewards = rewards_tensor
    else:
        raise ValueError(f"Unsupported option: {option}")

    if not reverse:
        raw_weights = scaled_rewards
    else:
        scaled_rewards = torch.clamp(scaled_rewards, min=epsilon)
        raw_weights = 1.0 / scaled_rewards

    if use_softmax:
        beta = 1.0 / max(temperature, epsilon)
        logits = -beta * raw_weights # 注意：这里原始代码是 -beta * raw_weights，如果期望值高权重高，应为 beta*raw_weights
        softmax_weights = F.softmax(logits, dim=0).numpy()
        weights = dict(zip(task_ids, softmax_weights))
    else:
        scaled_weights = raw_weights ** (1 / max(temperature, epsilon))
        total_weight = scaled_weights.sum()
        normalized_weights = scaled_weights / max(total_weight, epsilon) # 防止除零
        weights = dict(zip(task_ids, normalized_weights.numpy()))

    for task_id in weights:
        weights[task_id] = max(min(weights[task_id], clip_max), clip_min)

    return weights

# --- Main Training Function (Non-DDP) ---

def train_unizero_multitask(
        input_cfg_list: List[Tuple[int, Tuple[dict, dict]]],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':
    """
    Overview:
        UniZero的多任务训练入口 (非 DDP 版本)。
    Args:
        - input_cfg_list (:obj:`List[Tuple[int, Tuple[dict, dict]]]`): 不同任务的配置列表。
        - seed (:obj:`int`): 随机种子。
        - model (:obj:`Optional[torch.nn.Module]`): torch.nn.Module实例。
        - model_path (:obj:`Optional[str]`): 预训练模型路径。
        - max_train_iter (:obj:`Optional[int]`): 最大策略更新迭代次数。
        - max_env_step (:obj:`Optional[int]`): 最大收集环境交互步数。
    Returns:
        - policy (:obj:`Policy`): 收敛的策略。
    """
    # 初始化温度调度器 (保持不变)
    initial_temperature = 10.0
    final_temperature = 1.0
    threshold_steps = int(1e4)
    temperature_scheduler = TemperatureScheduler(
        initial_temp=initial_temperature,
        final_temp=final_temperature,
        threshold_steps=threshold_steps,
        mode='linear'
    )

    # 移除 rank 和 world_size 相关逻辑
    # rank = 0
    # world_size = 1

    # 单进程处理所有任务
    tasks = input_cfg_list
    total_tasks = len(tasks)
    print(f"单进程处理所有 {total_tasks} 个任务。")

    # 初始化列表
    cfgs = []
    game_buffers = []
    collector_envs = []
    evaluator_envs = []
    collectors = []
    evaluators = []

    # 确保至少有一个任务
    if not tasks:
        logging.error("没有提供任何任务配置，无法进行训练。")
        return None # 或者抛出异常

    # 使用第一个任务的配置创建共享的policy和learner
    task_id_first, [cfg_first, create_cfg_first] = tasks[0]

    # 确保指定的策略类型受支持
    assert create_cfg_first.policy.type in ['unizero_multitask',
                                       'sampled_unizero_multitask'], "train_unizero_multitask entry 目前仅支持 'unizero_multitask' 或 'sampled_unizero_multitask'"

    GameBuffer = None
    if create_cfg_first.policy.type == 'unizero_multitask':
        from lzero.mcts import UniZeroGameBuffer as GB
        GameBuffer = GB
    elif create_cfg_first.policy.type == 'sampled_unizero_multitask':
        from lzero.mcts import SampledUniZeroGameBuffer as SGB
        GameBuffer = SGB
    else:
      raise NotImplementedError(f"Policy type {create_cfg_first.policy.type} not fully supported for GameBuffer import.")


    # 根据CUDA可用性设置设备
    cfg_first.policy.device = 'cuda' if cfg_first.policy.cuda and torch.cuda.is_available() else 'cpu'
    logging.info(f'使用的设备: {cfg_first.policy.device}')

    # 编译主配置 (仅用于创建 policy 和 learner)
    # 注意：这里我们编译一次主配置，但后续循环中会为每个任务再次编译其特定配置
    compiled_cfg_first = compile_config(cfg_first, seed=seed, env=None, auto=True, create_cfg=create_cfg_first, save_cfg=True)

    # 创建共享的policy
    policy = create_policy(compiled_cfg_first.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    # 加载预训练模型（如果提供）
    if model_path is not None:
        logging.info(f'开始加载模型: {model_path}')
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=compiled_cfg_first.policy.device))
        logging.info(f'完成加载模型: {model_path}')

    # 创建TensorBoard日志记录器 (简化路径)
    log_dir = os.path.join('./{}/log/'.format(compiled_cfg_first.exp_name), 'serial') # 移除 rank
    tb_logger = SummaryWriter(log_dir)

    # 创建共享的learner
    learner = BaseLearner(compiled_cfg_first.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=compiled_cfg_first.exp_name)

    # policy_config = compiled_cfg_first.policy # 后续会被覆盖

    # 处理每个任务
    for local_task_id, (task_id, [cfg, create_cfg]) in enumerate(tasks):
        # 设置每个任务的随机种子
        current_seed = seed + task_id
        cfg.policy.device = 'cuda' if cfg.policy.cuda and torch.cuda.is_available() else 'cpu'
        # 编译当前任务的特定配置
        cfg = compile_config(cfg, seed=current_seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
        # 获取当前任务的 policy_config
        policy_config = cfg.policy
        policy_config.task_id = task_id # 显式设置 task_id

   
        policy.collect_mode.get_attribute('cfg').n_episode = policy_config.n_episode
        policy.eval_mode.get_attribute('cfg').n_episode = policy_config.n_episode

        # 创建环境
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
        collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
        evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
        collector_env.seed(current_seed)
        evaluator_env.seed(current_seed, dynamic_seed=False)
        set_pkg_seed(current_seed, use_cuda=cfg.policy.cuda)

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
    # 确保 reanalyze_batch_size 和 update_per_collect 从配置中获取
    # 使用第一个任务的配置，假设这些参数在任务间共享
    reanalyze_batch_size = compiled_cfg_first.policy.reanalyze_batch_size
    update_per_collect = compiled_cfg_first.policy.update_per_collect


    task_complexity_weight = compiled_cfg_first.policy.task_complexity_weight
    use_task_exploitation_weight = compiled_cfg_first.policy.use_task_exploitation_weight
    task_exploitation_weight = None

    # 创建任务奖励字典 (本地)
    task_rewards = {}  # {task_id: reward}

    # --- 主训练循环 ---
    while True:
        # 动态调整batch_size (使用本地版本)
        if compiled_cfg_first.policy.allocated_batch_sizes:
            clip_scale = np.clip(1 + (3 * train_epoch / 1000), 1, 4)
            allocated_batch_sizes_list = allocate_batch_size_local(cfgs, game_buffers, alpha=1.0, clip_scale=clip_scale)
            # 将 list 转换为 task_id -> batch_size 的字典，用于 policy._cfg
            allocated_batch_sizes_dict = {cfg.policy.task_id: size for cfg, size in zip(cfgs, allocated_batch_sizes_list)}
            print("分配后的 batch_sizes: ", allocated_batch_sizes_dict)
            # 更新 policy 的内部 batch_size 配置 (假设 policy._cfg.batch_size 是字典)
            policy._cfg.batch_size = allocated_batch_sizes_dict
            # 同时更新每个任务cfg和buffer的batch_size (如果需要单独控制)
            for i, cfg in enumerate(cfgs):
                 cfg.policy.batch_size = allocated_batch_sizes_dict # 更新cfg中的字典
                 # 如果 buffer 需要知道自己的 batch_size
                 # game_buffers[i].batch_size = allocated_batch_sizes_dict.get(cfg.policy.task_id)


        # 对于当前进程的每个任务，进行数据收集和评估
        for idx, (cfg, collector, evaluator, replay_buffer) in enumerate(
                zip(cfgs, collectors, evaluators, game_buffers)):

            current_task_id = cfg.policy.task_id # 获取当前任务ID

            # 记录缓冲区内存使用情况
            log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger, current_task_id)

            # 收集参数 (保持不变)
            policy_config = cfg.policy # 获取当前任务的 policy_config
            collect_kwargs = {
                'temperature': visit_count_temperature(
                    policy_config.manual_temperature_decay,
                    policy_config.fixed_temperature_value,
                    policy_config.threshold_training_steps_for_final_temperature,
                    trained_steps=learner.train_iter
                ),
                'epsilon': 0.0
            }
            update_per_collect = policy_config.update_per_collect
            if update_per_collect is None:
                update_per_collect = 40
            # 判断是否需要进行评估
            # if learner.train_iter == 0 or evaluator.should_eval(learner.train_iter):
            if learner.train_iter == 0 or evaluator.should_eval(learner.train_iter): # only for debug
                print('=' * 20)
                print(f'评估任务_id: {current_task_id}...')
                # 重置评估器策略状态
                evaluator._policy.reset(reset_init_data=True, task_id=current_task_id)

                # 执行安全评估 (使用非 DDP 版本)
                stop, reward = safe_eval(evaluator, learner, collector)
                if stop is None or reward is None:
                    print(f"评估过程中遇到问题或超时，任务ID: {current_task_id}，继续训练...")
                    task_rewards[current_task_id] = float('inf') # 评估失败设为最大难度
                else:
                    try:
                        eval_mean_reward = reward.get('eval_episode_return_mean', float('inf'))
                        print(f"任务 {current_task_id} 的评估奖励: {eval_mean_reward}")
                        task_rewards[current_task_id] = eval_mean_reward
                    except Exception as e:
                        print(f"任务 {current_task_id} 提取评估奖励时发生错误: {e}")
                        task_rewards[current_task_id] = float('inf')

            print('=' * 20)
            print(f'开始收集任务_id: {current_task_id}...')
            print(f'cfg.policy.task_id={current_task_id}')

            # 重置收集器策略状态
            collector._policy.reset(reset_init_data=True, task_id=current_task_id)
            # 收集数据
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)

            # 更新重放缓冲区
            replay_buffer.push_game_segments(new_data)
            replay_buffer.remove_oldest_data_to_fit()

            # 周期性地重新分析缓冲区 (逻辑保持不变)
            if policy_config.buffer_reanalyze_freq >= 1: # 使用当前任务的 policy_config
                # 确保 update_per_collect 是有效的
                if update_per_collect is None or update_per_collect == 0:
                     logging.warning("update_per_collect 未定义或为零，无法计算 reanalyze_interval")
                     reanalyze_interval = float('inf') # 避免触发
                else:
                     reanalyze_interval = update_per_collect // policy_config.buffer_reanalyze_freq

            else: # buffer_reanalyze_freq < 1
                 reanalyze_interval = float('inf') # 这种情况下，reanalyze 基于 train_epoch
                 if train_epoch > 0 and policy_config.buffer_reanalyze_freq > 0 and \
                    train_epoch % int(1 / policy_config.buffer_reanalyze_freq) == 0 and \
                    replay_buffer.get_num_of_transitions() // policy_config.num_unroll_steps > int(reanalyze_batch_size / policy_config.reanalyze_partition):
                     with timer:
                         replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                     buffer_reanalyze_count += 1
                     logging.info(f'缓冲区重新分析次数: {buffer_reanalyze_count}, 耗时: {timer.value}')

            logging.info(f'完成任务 {current_task_id} 的数据收集')

        # 检查是否有足够的数据进行训练 (使用修改后的逻辑)
        # 假设 policy._cfg.batch_size 是 {task_id: batch_size}
        # not_enough_data = any(
        #      game_buffers[idx].get_num_of_transitions() < policy._cfg.batch_size.get(cfg.policy.task_id, 0)
        #      for idx, (cfg, replay_buffer) in enumerate(zip(cfgs, game_buffers))
        # )
        not_enough_data = any(
             game_buffers[idx].get_num_of_transitions() < policy._cfg.batch_size[cfg.policy.task_id]
             for idx, (cfg, replay_buffer) in enumerate(zip(cfgs, game_buffers))
        )

        # 获取当前温度
        current_temperature_task_weight = temperature_scheduler.get_temperature(learner.train_iter)

        # 计算任务权重 (本地计算)
        if task_complexity_weight:
            # 直接使用本地收集的 task_rewards
            task_weights = compute_task_weights(task_rewards, temperature=current_temperature_task_weight)
            print(f"本地计算的任务权重 (按 task_id 排列): {task_weights}")
        else:
            task_weights = None


        # 学习策略
        if not not_enough_data:
            for i in range(update_per_collect): # 使用从主配置获取的 update_per_collect
                train_data_multi_task = []
                envstep_this_epoch = 0 # 当前训练迭代累积的 envstep

                for idx, (cfg, collector, replay_buffer) in enumerate(zip(cfgs, collectors, game_buffers)):
                    current_task_id = cfg.policy.task_id
                    # 获取该任务的批次大小
                    # current_batch_size = policy._cfg.batch_size.get(current_task_id, 0)
                    current_batch_size = policy._cfg.batch_size[current_task_id]
                    
                    if current_batch_size == 0:
                         logging.warning(f"任务 {current_task_id} 的 batch_size 为 0，跳过采样。")
                         continue

                    if replay_buffer.get_num_of_transitions() >= current_batch_size:
                        policy_config = cfg.policy # 获取当前任务的 policy_config
                        # 重新分析逻辑 (放在采样前)
                        if policy_config.buffer_reanalyze_freq >= 1:
                            if update_per_collect is not None and update_per_collect > 0: # 确保 update_per_collect 有效
                                reanalyze_interval = update_per_collect // policy_config.buffer_reanalyze_freq
                                if i % reanalyze_interval == 0 and \
                                        replay_buffer.get_num_of_transitions() // policy_config.num_unroll_steps > int(reanalyze_batch_size / policy_config.reanalyze_partition):
                                    with timer:
                                        replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                                    buffer_reanalyze_count += 1
                                    logging.info(f'缓冲区重新分析次数: {buffer_reanalyze_count}, 耗时: {timer.value}')

                        train_data = replay_buffer.sample(current_batch_size, policy)
                        train_data.append(current_task_id)  # 追加task_id以区分任务
                        train_data_multi_task.append(train_data)
                        envstep_this_epoch += collector.envstep # 累加所有 collector 的 envstep
                    else:
                        logging.warning(
                            f'任务 {current_task_id} 数据不足: '
                            f'batch_size: {current_batch_size}, buffer: {replay_buffer.get_num_of_transitions()}'
                        )
                        # 如果一个任务数据不足，是否要跳过整个训练批次？当前逻辑是继续收集其他任务的数据
                        # break # 如果希望一个不足就中断本次训练，取消注释此行

                if train_data_multi_task:
                    # learn_kwargs 准备 (使用本地计算的 task_weights)
                    # 注意：原代码 task_exploitation_weight 的计算依赖 log_vars，需要在 train 之后计算
                    learn_kwargs = {'task_weights': task_weights} # 先传入基于reward的权重

                    # 执行训练
                    log_vars = learner.train(train_data_multi_task, envstep_this_epoch, policy_kwargs=learn_kwargs)

                    # --- 计算并更新 task_exploitation_weight (本地计算) ---
                    if i == 0 and use_task_exploitation_weight:
                        # 从 log_vars 中提取本地任务的 obs_loss
                        local_obs_loss_task = {}
                        for cfg in cfgs:
                             task_id = cfg.policy.task_id
                             loss_key = f'noreduce_obs_loss_task{task_id}'
                             if log_vars and loss_key in log_vars[0]: # 检查 log_vars 是否有效
                                 local_obs_loss_task[task_id] = log_vars[0][loss_key]

                        # 计算任务利用权重 (本地)
                        if local_obs_loss_task:
                             task_exploitation_weight = compute_task_weights(
                                 local_obs_loss_task,
                                 option="rank",
                                 temperature=1, # 或者使用 current_temperature_task_weight
                                 reverse=True # 通常损失越高，权重越低？或者反过来？这里假设损失高权重高 (探索难度大的任务)
                             )
                             print(f"本地计算的 task_exploitation_weight (按 task_id 排列): {task_exploitation_weight}")
                        else:
                             logging.warning("未能计算本地 task_exploitation_weight，obs_loss 数据为空或无效。")
                             task_exploitation_weight = None

                        # 注意：这里计算出的 exploitation_weight 在下一次迭代 (i=1) 的 learn_kwargs 中才会生效
                        # 如果希望立即生效，需要在 learner.train 内部处理或再次调用 train？
                        # 或者更新 learn_kwargs 供后续使用 (如果 learner.train 支持动态kwargs)
                        learn_kwargs['task_exploitation_weight'] = task_exploitation_weight # 更新kwargs供后续打印或调试


                    # --- 更新优先级 ---
                    if compiled_cfg_first.policy.use_priority: # 使用主配置判断是否用优先级
                         if log_vars: # 确保 log_vars 有效
                             for batch_idx, (cfg, replay_buffer) in enumerate(zip(cfgs, game_buffers)):
                                 task_id = cfg.policy.task_id
                                 priority_key = f'value_priority_task{task_id}'
                                 if priority_key in log_vars[0]:
                                     # 确保 train_data_multi_task 包含对应任务的数据
                                     if batch_idx < len(train_data_multi_task):
                                         try:
                                             replay_buffer.update_priority(
                                                 train_data_multi_task[batch_idx], # 使用对应的训练数据
                                                 log_vars[0][priority_key]
                                             )
                                             # --- 优先级统计与归一化 (本地逻辑) ---
                                             current_priorities = log_vars[0][priority_key]
                                             mean_priority = np.mean(current_priorities)
                                             std_priority = np.std(current_priorities)
                                             alpha = 0.1
                                             running_mean_key = f'running_mean_priority_task{task_id}'
                                             if running_mean_key not in value_priority_tasks:
                                                 value_priority_tasks[running_mean_key] = mean_priority
                                             else:
                                                 value_priority_tasks[running_mean_key] = (
                                                         alpha * mean_priority +
                                                         (1 - alpha) * value_priority_tasks[running_mean_key]
                                                 )
                                             running_mean_priority = value_priority_tasks[running_mean_key]
                                             # normalized_priorities = (current_priorities - running_mean_priority) / (std_priority + 1e-6)
                                             if policy_config.print_task_priority_logs: # 使用当前任务配置
                                                 print(f"任务 {task_id} - 平均优先级: {mean_priority:.8f}, "
                                                       f"运行平均优先级: {running_mean_priority:.8f}, "
                                                       f"标准差: {std_priority:.8f}")
                                         except Exception as e:
                                             logging.error(f"更新任务 {task_id} 优先级时出错: {e}")
                                     else:
                                        logging.warning(f"无法更新任务 {task_id} 的优先级，因为 train_data_multi_task 中缺少对应数据。")

                                 else:
                                     logging.warning(f"无法找到任务 {task_id} 的优先级键 '{priority_key}' in log_vars[0]")
                         else:
                            logging.warning("log_vars 为空，无法更新优先级。")


        train_epoch += 1
        # 单进程不需要 barrier

        # 检查是否需要终止训练 (本地检查)
        # 计算本地最大 envstep
        local_max_envstep = max(collector.envstep for collector in collectors) if collectors else 0
        max_envstep_reached = local_max_envstep >= max_env_step

        # 检查本地 train_iter
        max_train_iter_reached = learner.train_iter >= max_train_iter

        if max_envstep_reached or max_train_iter_reached:
            logging.info(f'达到终止条件: env_step ({local_max_envstep}/{max_env_step}) 或 train_iter ({learner.train_iter}/{max_train_iter})')
            break

        # 重新计算位置嵌入差异并清除缓存 (如果 policy 需要)
        # 假设 policy 有此方法
        if hasattr(policy, 'recompute_pos_emb_diff_and_clear_cache'):
             policy.recompute_pos_emb_diff_and_clear_cache()


    # 调用learner的after_run钩子
    learner.call_hook('after_run')
    return policy