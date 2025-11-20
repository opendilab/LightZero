import logging
import os
from functools import partial
from typing import Tuple, Optional, List, Dict

import torch
import numpy as np
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy, Policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import set_pkg_seed, get_rank, get_world_size
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter

from lzero.entry.utils import log_buffer_memory_usage, TemperatureScheduler
from lzero.policy import visit_count_temperature
# HACK: The following imports are for type hinting purposes.
# The actual GameBuffer is selected dynamically based on the policy type.
from lzero.mcts import UniZeroGameBuffer
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.worker import MuZeroSegmentCollector as Collector
from ding.utils import EasyTimer
import torch.nn.functional as F

import torch.distributed as dist
import concurrent.futures
from collections import defaultdict


# ====================================================================================================================
# Note: The following global benchmark score definitions are for reference.
# The active implementation for score initialization is located within the `train_unizero_multitask_segment_ddp` function
# to ensure scores are correctly set based on the `benchmark_name` argument passed to the function.
# ====================================================================================================================
# global BENCHMARK_NAME
# # BENCHMARK_NAME = "atari"
# BENCHMARK_NAME = "dmc" # TODO
# if BENCHMARK_NAME == "atari":
#     RANDOM_SCORES = np.array([
#         227.8, 5.8, 222.4, 210.0, 14.2, 2360.0, 0.1, 1.7, 811.0, 10780.5,
#         152.1, 0.0, 65.2, 257.6, 1027.0, 29.0, 52.0, 1598.0, 258.5, 307.3,
#         -20.7, 24.9, 163.9, 11.5, 68.4, 533.4
#     ])
#     HUMAN_SCORES = np.array([
#         7127.7, 1719.5, 742.0, 8503.3, 753.1, 37187.5, 12.1, 30.5, 7387.8, 35829.4,
#         1971.0, 29.6, 4334.7, 2412.5, 30826.4, 302.8, 3035.0, 2665.5, 22736.3, 6951.6,
#         14.6, 69571.3, 13455.0, 7845.0, 42054.7, 11693.2
#     ])
# elif BENCHMARK_NAME == "dmc":
#     RANDOM_SCORES = np.array([0]*26)
#     HUMAN_SCORES = np.array([1000]*26)
#
# # New order to original index mapping
# # New order: [Pong, MsPacman, Seaquest, Boxing, Alien, ChopperCommand, Hero, RoadRunner,
# #            Amidar, Assault, Asterix, BankHeist, BattleZone, CrazyClimber, DemonAttack,
# #            Freeway, Frostbite, Gopher, Jamesbond, Kangaroo, Krull, KungFuMaster,
# #            PrivateEye, UpNDown, Qbert, Breakout]
# # Mapping to indices in the original array (0-based)
# new_order = [
#     20, 19, 24, 6, 0, 8, 14, 23, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 15, 16, 17, 18, 21, 25, 22, 7
# ]
#
# # Generate new arrays based on new_order
# new_RANDOM_SCORES = RANDOM_SCORES[new_order]
# new_HUMAN_SCORES = HUMAN_SCORES[new_order]


# ------------------------------------------------------------
# 1. Add a dedicated process-group for the learner.
#    (This should be called once during main/learner initialization)
# ------------------------------------------------------------
def build_learner_group(learner_ranks: list[int]) -> "dist.ProcessGroup":
    """
    Overview:
        Build a new process group for learners that perform backward propagation.
        This is useful in scenarios like MoCo where specific ranks handle the learning process.
    Arguments:
        - learner_ranks (:obj:`list[int]`): A list of ranks that will perform the backward pass.
                                            For example, if CUDA_VISIBLE_DEVICES=0,1, then learner_ranks=[0,1].
    Returns:
        - pg (:obj:`dist.ProcessGroup`): A new process group for the specified learner ranks.
    """
    world_pg = dist.group.WORLD
    pg = dist.new_group(ranks=learner_ranks, backend='nccl')
    if dist.get_rank() in learner_ranks:
        torch.cuda.set_device(learner_ranks.index(dist.get_rank()))
    return pg


# Stores the latest evaluation returns: {task_id: eval_episode_return_mean}
GLOBAL_EVAL_RETURNS: Dict[int, float] = defaultdict(lambda: None)


def compute_unizero_mt_normalized_stats(
        eval_returns: Dict[int, float]
) -> Tuple[Optional[float], Optional[float]]:
    """
    Overview:
        Computes the Human-Normalized Mean and Median from evaluation returns for UniZero-MT.
        If there are no samples, it returns (None, None).
    Arguments:
        - eval_returns (:obj:`Dict[int, float]`): A dictionary of evaluation returns, keyed by task ID.
    Returns:
        - (:obj:`Tuple[Optional[float], Optional[float]]`): A tuple containing the human-normalized mean and median.
                                                            Returns (None, None) if no valid returns are provided.
    """
    normalized = []
    for tid, ret in eval_returns.items():
        if ret is None:
            continue
        # Denominator for normalization
        denom = new_HUMAN_SCORES[tid] - new_RANDOM_SCORES[tid]
        if denom == 0:
            continue
        normalized.append((ret - new_RANDOM_SCORES[tid]) / denom)

    if not normalized:
        return None, None
    arr = np.asarray(normalized, dtype=np.float32)
    return float(arr.mean()), float(np.median(arr))


# Set a timeout for evaluation in seconds
TIMEOUT = 12000  # e.g., 200 minutes

timer = EasyTimer()


def safe_eval(
        evaluator: Evaluator,
        learner: BaseLearner,
        collector: Collector,
        rank: int,
        world_size: int
) -> Tuple[Optional[bool], Optional[float]]:
    """
    Overview:
        Safely executes an evaluation task with a timeout to prevent hangs.
    Arguments:
        - evaluator (:obj:`Evaluator`): The evaluator instance.
        - learner (:obj:`BaseLearner`): The learner instance.
        - collector (:obj:`Collector`): The data collector instance.
        - rank (:obj:`int`): The rank of the current process.
        - world_size (:obj:`int`): The total number of processes.
    Returns:
        - (:obj:`Tuple[Optional[bool], Optional[float]]`): A tuple containing the stop flag and reward if evaluation succeeds,
                                                           otherwise (None, None).
    """
    try:
        print(f"=========评估开始 Rank {rank}/{world_size}===========")
        # Reset the stop_event to ensure it is not set before each evaluation.
        evaluator.stop_event.clear()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit the evaluation task.
            future = executor.submit(evaluator.eval, learner.save_checkpoint, learner.train_iter, collector.envstep)
            try:
                stop, reward = future.result(timeout=TIMEOUT)
            except concurrent.futures.TimeoutError:
                # If a timeout occurs, set the stop_event.
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
        game_buffers: List['UniZeroGameBuffer'],
        alpha: float = 1.0,
        clip_scale: int = 1
) -> List[int]:
    """
    Overview:
        Allocates batch sizes for different tasks inversely proportional to the number of collected episodes.
        It also dynamically adjusts the batch size range to improve training stability and efficiency.
    Arguments:
        - cfgs (:obj:`List[dict]`): A list of configurations for each task.
        - game_buffers (:obj:`List[GameBuffer]`): A list of replay buffer instances for each task.
        - alpha (:obj:`float`): A hyperparameter to control the degree of inverse proportionality. Defaults to 1.0.
        - clip_scale (:obj:`int`): The clipping ratio for dynamic adjustment. Defaults to 1.
    Returns:
        - (:obj:`List[int]`): The list of allocated batch sizes.
    """
    # Extract the number of collected episodes for each task.
    buffer_num_of_collected_episodes = [buffer.num_of_collected_episodes for buffer in game_buffers]

    # Get the current world_size and rank.
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # Gather the lists of collected episodes from all ranks.
    all_task_num_of_collected_episodes = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(all_task_num_of_collected_episodes, buffer_num_of_collected_episodes)

    # Merge the collected episodes from all ranks into a single list.
    all_task_num_of_collected_episodes = [
        episode for sublist in all_task_num_of_collected_episodes for episode in sublist
    ]
    if rank == 0:
        print(f'所有任务的 collected episodes: {all_task_num_of_collected_episodes}')

    # Calculate the inverse proportional weights for each task.
    inv_episodes = np.array([1.0 / (episodes + 1) for episodes in all_task_num_of_collected_episodes])
    inv_sum = np.sum(inv_episodes)

    # Calculate the total batch size (sum of cfg.policy.batch_size for all tasks).
    total_batch_size = cfgs[0].policy.total_batch_size

    # Dynamic adjustment: define the min and max batch size range.
    avg_batch_size = total_batch_size / world_size
    min_batch_size = avg_batch_size / clip_scale
    max_batch_size = avg_batch_size * clip_scale

    # Dynamically adjust alpha to make batch size changes smoother.
    task_weights = (inv_episodes / inv_sum) ** alpha
    batch_sizes = total_batch_size * task_weights

    # Clip the batch sizes to be within the [min_batch_size, max_batch_size] range.
    batch_sizes = np.clip(batch_sizes, min_batch_size, max_batch_size)

    # Ensure batch sizes are integers.
    batch_sizes = [int(size) for size in batch_sizes]

    return batch_sizes


def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Overview:
        Symlog normalization to reduce the magnitude difference of target values.
        symlog(x) = sign(x) * log(|x| + 1)
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def inv_symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Overview:
        Inverse operation of Symlog to restore the original value.
        inv_symlog(x) = sign(x) * (exp(|x|) - 1)
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


# Global max and min for "run-max-min" normalization
GLOBAL_MAX = -float('inf')
GLOBAL_MIN = float('inf')


def compute_task_weights(
        task_returns: Dict[int, float],
        option: str = "symlog",
        epsilon: float = 1e-6,
        temperature: float = 1.0,
        use_softmax: bool = False,
        reverse: bool = False,
        clip_min: float = 1e-2,
        clip_max: float = 1.0,
) -> Dict[int, float]:
    """
    Overview:
        An improved function for calculating task weights, supporting multiple normalization methods,
        Softmax, proportional/inverse weighting, and weight clipping.
    Arguments:
        - task_returns (:obj:`Dict[int, float]`): A dictionary where keys are task_ids and values are evaluation rewards or losses.
        - option (:obj:`str`): Normalization method. Options: "symlog", "max-min", "run-max-min", "rank", "none".
        - epsilon (:obj:`float`): A small value to avoid division by zero.
        - temperature (:obj:`float`): Temperature coefficient to control the weight distribution.
        - use_softmax (:obj:`bool`): Whether to use Softmax for weight distribution.
        - reverse (:obj:`bool`): If True, weights are inversely proportional to values; if False, they are proportional.
        - clip_min (:obj:`float`): The minimum value to clip weights to.
        - clip_max (:obj:`float`): The maximum value to clip weights to.
    Returns:
        - (:obj:`Dict[int, float]`): A dictionary of weights for each task, where keys are task_ids.
    """
    global GLOBAL_MAX, GLOBAL_MIN

    # Return an empty dictionary if the input is empty.
    if not task_returns:
        return {}

    # Step 1: Construct a tensor from the values of task_returns.
    task_ids = list(task_returns.keys())
    returns_tensor = torch.tensor(list(task_returns.values()), dtype=torch.float32)

    if option == "symlog":
        # Use symlog normalization.
        scaled_returns = symlog(returns_tensor)
    elif option == "max-min":
        # Use max-min normalization.
        max_reward = returns_tensor.max().item()
        min_reward = returns_tensor.min().item()
        scaled_returns = (returns_tensor - min_reward) / (max_reward - min_reward + epsilon)
    elif option == "run-max-min":
        # Use global running max-min normalization.
        GLOBAL_MAX = max(GLOBAL_MAX, returns_tensor.max().item())
        GLOBAL_MIN = min(GLOBAL_MIN, returns_tensor.min().item())
        scaled_returns = (returns_tensor - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN + epsilon)
    elif option == "rank":
        # Use rank-based normalization. Rank is based on value size, with 1 for the smallest.
        sorted_indices = torch.argsort(returns_tensor)
        scaled_returns = torch.empty_like(returns_tensor)
        rank_values = torch.arange(1, len(returns_tensor) + 1, dtype=torch.float32)  # Ranks from 1 to N
        scaled_returns[sorted_indices] = rank_values
    elif option == "none":
        # No normalization.
        scaled_returns = returns_tensor
    else:
        raise ValueError(f"Unsupported option: {option}")

    # Step 2: Determine if weights are proportional or inversely proportional based on `reverse`.
    if not reverse:
        # Proportional: weight is positively correlated with the value.
        raw_weights = scaled_returns
    else:
        # Inverse: weight is negatively correlated with the value.
        # Clamp to avoid division by zero or negative numbers.
        scaled_returns = torch.clamp(scaled_returns, min=epsilon)
        raw_weights = 1.0 / scaled_returns

    # Step 3: Calculate weights with or without Softmax.
    if use_softmax:
        # Use Softmax for weight distribution.
        beta = 1.0 / max(temperature, epsilon)  # Ensure temperature is not zero.
        logits = -beta * raw_weights
        softmax_weights = F.softmax(logits, dim=0).numpy()
        weights = dict(zip(task_ids, softmax_weights))
    else:
        # Do not use Softmax, calculate weights directly.
        # Temperature scaling.
        scaled_weights = raw_weights ** (1 / max(temperature, epsilon))  # Ensure temperature is not zero.

        # Normalize weights.
        total_weight = scaled_weights.sum()
        normalized_weights = scaled_weights / total_weight

        # Convert to dictionary.
        weights = dict(zip(task_ids, normalized_weights.numpy()))

    # Step 4: Clip the weight range.
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
        benchmark_name: str = "atari"
) -> 'Policy':
    """
    Overview:
        The training entry point for UniZero, designed to enhance the planning capabilities of reinforcement learning agents
        by addressing the limitations of MuZero-like algorithms in environments requiring long-term dependency capture.
        For more details, please refer to https://arxiv.org/abs/2406.10667.

    Arguments:
        - input_cfg_list (:obj:`List[Tuple[int, Tuple[dict, dict]]]`): A list of configurations for different tasks.
        - seed (:obj:`int`): The random seed.
        - model (:obj:`Optional[torch.nn.Module]`): An instance of torch.nn.Module.
        - model_path (:obj:`Optional[str]`): The path to a pre-trained model checkpoint file.
        - max_train_iter (:obj:`Optional[int]`): The maximum number of policy update iterations during training.
        - max_env_step (:obj:`Optional[int]`): The maximum number of environment interaction steps to collect.
        - benchmark_name (:obj:`str`): The name of the benchmark, e.g., "atari" or "dmc".

    Returns:
        - policy (:obj:`Policy`): The converged policy.
    """
    # ------------------------------------------------------------------------------------
    # ====== UniZero-MT Benchmark Scores (corresponding to 26 Atari100k task IDs) ======
    # Original RANDOM_SCORES and HUMAN_SCORES
    if benchmark_name == "atari":
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
    elif benchmark_name == "dmc":
        RANDOM_SCORES = np.zeros(26)
        HUMAN_SCORES = np.ones(26) * 1000
    else:
        raise ValueError(f"Unsupported BENCHMARK_NAME: {benchmark_name}")

    # New order to original index mapping
    # New order: [Pong, MsPacman, Seaquest, Boxing, Alien, ChopperCommand, Hero, RoadRunner,
    #            Amidar, Assault, Asterix, BankHeist, BattleZone, CrazyClimber, DemonAttack,
    #            Freeway, Frostbite, Gopher, Jamesbond, Kangaroo, Krull, KungFuMaster,
    #            PrivateEye, UpNDown, Qbert, Breakout]
    # Mapping to indices in the original array (0-based)
    new_order = [
        20, 19, 24, 6, 0, 8, 14, 23, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 15, 16, 17, 18, 21, 25, 22, 7
    ]
    global new_RANDOM_SCORES, new_HUMAN_SCORES
    # Generate new arrays based on new_order
    new_RANDOM_SCORES = RANDOM_SCORES[new_order]
    new_HUMAN_SCORES = HUMAN_SCORES[new_order]
    # Log the reordered results
    print("重排后的 RANDOM_SCORES:")
    print(new_RANDOM_SCORES)
    print("\n重排后的 HUMAN_SCORES:")
    print(new_HUMAN_SCORES)
    # ------------------------------------------------------------------------------------

    # Initialize the temperature scheduler for task weighting.
    initial_temperature = 10.0
    final_temperature = 1.0
    threshold_steps = int(1e4)  # Temperature drops to 1.0 after 10k training steps.
    temperature_scheduler = TemperatureScheduler(
        initial_temp=initial_temperature,
        final_temp=final_temperature,
        threshold_steps=threshold_steps,
        mode='linear'  # or 'exponential'
    )

    # Get the current process rank and total world size.
    rank = get_rank()
    world_size = get_world_size()

    # Task partitioning among ranks.
    total_tasks = len(input_cfg_list)
    tasks_per_rank = total_tasks // world_size
    remainder = total_tasks % world_size

    # ==================== START: 关键修复 ====================
    # 1. 精确计算当前Rank负责的任务数量
    if rank < remainder:
        start_idx = rank * (tasks_per_rank + 1)
        end_idx = start_idx + tasks_per_rank + 1
        num_tasks_for_this_rank = tasks_per_rank + 1
    else:
        start_idx = rank * tasks_per_rank + remainder
        end_idx = start_idx + tasks_per_rank
        num_tasks_for_this_rank = tasks_per_rank
    # ==================== END: 关键修复 ====================

    tasks_for_this_rank = input_cfg_list[start_idx:end_idx]

    # Ensure at least one task is assigned.
    if len(tasks_for_this_rank) == 0:
        logging.warning(f"Rank {rank}: No tasks assigned, continuing execution.")
        # Initialize empty lists to avoid errors later.
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
        # Use the config of the first task to create a shared policy.
        task_id, [cfg, create_cfg] = tasks_for_this_rank[0]

        # ==================== START: 关键修复 ====================
        # 2. 将正确的任务数量设置到 *所有* 相关配置中
        #    在创建Policy实例之前，必须确保配置是正确的
        for config_tuple in tasks_for_this_rank:
            # config_tuple is (task_id, [cfg_obj, create_cfg_obj])
            config_tuple[1][0].policy.task_num = num_tasks_for_this_rank
        
        # 3. 确保用于创建Policy的那个cfg对象也拥有正确的task_num
        cfg.policy.task_num = num_tasks_for_this_rank
        # ==================== END: 关键修复 ====================

        # Ensure the specified policy type is supported.
        assert create_cfg.policy.type in ['unizero_multitask', 'sampled_unizero_multitask'], \
            "train_unizero entry currently only supports 'unizero_multitask' or 'sampled_unizero_multitask'"

        if create_cfg.policy.type == 'unizero_multitask':
            from lzero.mcts import UniZeroGameBuffer as GameBuffer
        if create_cfg.policy.type == 'sampled_unizero_multitask':
            from lzero.mcts import SampledUniZeroGameBuffer as GameBuffer

        # Set device based on CUDA availability.
        cfg.policy.device = cfg.policy.model.world_model_cfg.device if torch.cuda.is_available() else 'cpu'
        logging.info(f'配置的设备: {cfg.policy.device}')

        # Compile the configuration.
        cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
        # Create the shared policy.
        policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

        # Load a pre-trained model if a path is provided.
        if model_path is not None:
            logging.info(f'开始加载模型: {model_path}')
            policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))
            logging.info(f'完成加载模型: {model_path}')

        # Create a TensorBoard logger.
        log_dir = os.path.join('./{}/log'.format(cfg.exp_name), f'serial_rank_{rank}')
        tb_logger = SummaryWriter(log_dir)

        # Create the shared learner.
        learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

        policy_config = cfg.policy

        # Process each task assigned to the current rank.
        for local_task_id, (task_id, [cfg, create_cfg]) in enumerate(tasks_for_this_rank):
            # Set a unique random seed for each task.
            cfg.policy.device = 'cuda' if cfg.policy.cuda and torch.cuda.is_available() else 'cpu'
            cfg = compile_config(cfg, seed=seed + task_id, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
            policy_config = cfg.policy
            policy.collect_mode.get_attribute('cfg').n_episode = policy_config.n_episode
            policy.eval_mode.get_attribute('cfg').n_episode = policy_config.n_episode

            # Create environments.
            env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
            collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
            evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
            collector_env.seed(cfg.seed + task_id)
            evaluator_env.seed(cfg.seed + task_id, dynamic_seed=False)
            set_pkg_seed(cfg.seed + task_id, use_cuda=cfg.policy.cuda)

            # Create task-specific game buffers, collectors, and evaluators.
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
            # Handle batch_size robustly - it might be a list or already an integer
            if isinstance(cfg.policy.batch_size, (list, tuple)):
                replay_buffer.batch_size = cfg.policy.batch_size[task_id]
            elif isinstance(cfg.policy.batch_size, dict):
                replay_buffer.batch_size = cfg.policy.batch_size[task_id]
            else:
                replay_buffer.batch_size = cfg.policy.batch_size

            game_buffers.append(replay_buffer)
            collector_envs.append(collector_env)
            evaluator_envs.append(evaluator_env)
            collectors.append(collector)
            evaluators.append(evaluator)

    # Call the learner's before_run hook.
    learner.call_hook('before_run')
    value_priority_tasks = {}

    buffer_reanalyze_count = 0
    train_epoch = 0
    reanalyze_batch_size = cfg.policy.reanalyze_batch_size
    update_per_collect = cfg.policy.update_per_collect

    task_exploitation_weight = None

    # Dictionary to store task rewards.
    task_returns = {}  # {task_id: reward}

    while True:
        # Dynamically adjust batch sizes.
        if cfg.policy.allocated_batch_sizes:
            clip_scale = np.clip(1 + (3 * train_epoch / 1000), 1, 4)
            allocated_batch_sizes = allocate_batch_size(cfgs, game_buffers, alpha=1.0, clip_scale=clip_scale)
            if rank == 0:
                print("分配后的 batch_sizes: ", allocated_batch_sizes)
            # Assign the corresponding batch size to each task config
            for idx, (cfg, collector, evaluator, replay_buffer) in enumerate(
                    zip(cfgs, collectors, evaluators, game_buffers)):
                task_id = cfg.policy.task_id
                if isinstance(allocated_batch_sizes, dict):
                    cfg.policy.batch_size = allocated_batch_sizes.get(task_id, cfg.policy.batch_size)
                elif isinstance(allocated_batch_sizes, list):
                    # Use the index in the list or task_id as fallback
                    cfg.policy.batch_size = allocated_batch_sizes[idx] if idx < len(allocated_batch_sizes) else cfg.policy.batch_size
                else:
                    logging.warning(f"Unexpected type for allocated_batch_sizes: {type(allocated_batch_sizes)}")
            # Also update the policy config (use the full list for compatibility)
            policy._cfg.batch_size = allocated_batch_sizes

        # For each task on the current rank, perform data collection and evaluation.
        for idx, (cfg, collector, evaluator, replay_buffer) in enumerate(
                zip(cfgs, collectors, evaluators, game_buffers)):

            # Log buffer memory usage.
            log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger, cfg.policy.task_id)

            collect_kwargs = {
                'temperature': visit_count_temperature(
                    policy_config.manual_temperature_decay,
                    policy_config.fixed_temperature_value,
                    policy_config.threshold_training_steps_for_final_temperature,
                    trained_steps=learner.train_iter
                ),
                'epsilon': 0.0  # Default epsilon value.
            }

            if policy_config.eps.eps_greedy_exploration_in_collect:
                epsilon_greedy_fn = get_epsilon_greedy_fn(
                    start=policy_config.eps.start,
                    end=policy_config.eps.end,
                    decay=policy_config.eps.decay,
                    type_=policy_config.eps.type
                )
                collect_kwargs['epsilon'] = epsilon_greedy_fn(collector.envstep)

            # Check if it's time for evaluation.
            if learner.train_iter > 10 and learner.train_iter % cfg.policy.eval_freq == 0:
            # if learner.train_iter == 0 or learner.train_iter % cfg.policy.eval_freq == 0: # only for debug TODO
            
                print('=' * 20)
                print(f'Rank {rank} 评估任务_id: {cfg.policy.task_id}...')

                # TODO: Ensure policy reset logic is optimal for multi-task settings.
                evaluator._policy.reset(reset_init_data=True, task_id=cfg.policy.task_id)

                # Perform safe evaluation.
                stop, reward = safe_eval(evaluator, learner, collector, rank, world_size)
                # Check if evaluation was successful.
                if stop is None or reward is None:
                    print(f"Rank {rank} 在评估过程中遇到问题，继续训练...")
                    task_returns[cfg.policy.task_id] = float('inf')  # Set task difficulty to max if evaluation fails.
                else:
                    # Extract 'eval_episode_return_mean' from the reward dictionary.
                    try:
                        eval_mean_reward = reward.get('eval_episode_return_mean', float('inf'))
                        print(f"任务 {cfg.policy.task_id} 的评估奖励: {eval_mean_reward}")
                        task_returns[cfg.policy.task_id] = eval_mean_reward
                    except Exception as e:
                        print(f"提取评估奖励时发生错误: {e}")
                        task_returns[cfg.policy.task_id] = float('inf')  # Set reward to max on error.

            print('=' * 20)
            print(f'开始收集 Rank {rank} 的任务_id: {cfg.policy.task_id}...')
            print(f'Rank {rank}: cfg.policy.task_id={cfg.policy.task_id} ')

            # Reset initial data before each collection, crucial for multi-task settings.
            collector._policy.reset(reset_init_data=True, task_id=cfg.policy.task_id)
            # Collect data.
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)

            # Update the replay buffer.
            replay_buffer.push_game_segments(new_data)
            replay_buffer.remove_oldest_data_to_fit()

            # ===== For debugging purposes only =====
            # if train_epoch > 2:
            #     with timer:
            #         replay_buffer.reanalyze_buffer(2, policy)
            #     buffer_reanalyze_count += 1
            #     logging.info(f'缓冲区重新分析次数: {buffer_reanalyze_count}')
            #     logging.info(f'缓冲区重新分析耗时: {timer.value}')
            # ====================================

            # Periodically reanalyze the buffer.
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

            # Log after data collection.
            logging.info(f'Rank {rank}: 完成任务 {cfg.policy.task_id} 的数据收集')

        # Check if there is enough data for training.
        not_enough_data = any(
            replay_buffer.get_num_of_transitions() < cfgs[0].policy.total_batch_size / world_size
            for replay_buffer in game_buffers
        )

        print(f"not_enough_data:{not_enough_data}")
        # Get the current temperature for task weighting.
        current_temperature_task_weight = temperature_scheduler.get_temperature(learner.train_iter)

        if learner.train_iter > 10 and learner.train_iter % cfg.policy.eval_freq == 0:
            # Calculate task weights.
            try:
                # Gather task rewards.
                dist.barrier()
                all_task_returns = [None for _ in range(world_size)]
                dist.all_gather_object(all_task_returns, task_returns)
                # Merge task rewards.
                merged_task_returns = {}
                for returns in all_task_returns:
                    if returns:
                        merged_task_returns.update(returns)

                logging.warning(f"Rank {rank}: merged_task_returns: {merged_task_returns}")

                # Calculate global task weights.
                task_weights = compute_task_weights(merged_task_returns, temperature=current_temperature_task_weight)

                # ---------- Maintain UniZero-MT global evaluation results ----------
                for tid, ret in merged_task_returns.items():
                    GLOBAL_EVAL_RETURNS[tid] = ret  # Update even for solved tasks.

                # Calculate Human-Normalized Mean / Median.
                uni_mean, uni_median = compute_unizero_mt_normalized_stats(GLOBAL_EVAL_RETURNS)

                if uni_mean is not None:  # At least one task has been evaluated.
                    if rank == 0:  # Only write to TensorBoard on rank 0 to avoid duplication.
                        tb_logger.add_scalar('UniZero-MT/NormalizedMean', uni_mean, global_step=learner.train_iter)
                        tb_logger.add_scalar('UniZero-MT/NormalizedMedian', uni_median, global_step=learner.train_iter)
                    logging.info(f"Rank {rank}: UniZero-MT Norm Mean={uni_mean:.4f}, Median={uni_median:.4f}")
                else:
                    logging.info(f"Rank {rank}: 暂无数据计算 UniZero-MT 归一化指标")

                # Synchronize task weights.
                dist.broadcast_object_list([task_weights], src=0)
            except Exception as e:
                logging.error(f'Rank {rank}: 同步任务权重失败，错误: {e}')
                break

        # ---------------- Sampling done, preparing for backward pass ----------------
        # dist.barrier()  # ★★★ Critical synchronization point ★★★

        # Learn policy.
        if not not_enough_data:
            for i in range(update_per_collect):
                train_data_multi_task = []
                envstep_multi_task = 0
                for idx, (cfg, collector, replay_buffer) in enumerate(zip(cfgs, collectors, game_buffers)):
                    envstep_multi_task += collector.envstep
                    # Handle batch_size robustly - it might be a list or already an integer
                    if isinstance(cfg.policy.batch_size, (list, tuple)):
                        batch_size = cfg.policy.batch_size[cfg.policy.task_id]
                    elif isinstance(cfg.policy.batch_size, dict):
                        batch_size = cfg.policy.batch_size[cfg.policy.task_id]
                    else:
                        batch_size = cfg.policy.batch_size

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
                        train_data.append(cfg.policy.task_id)  # Append task_id to differentiate tasks.
                        train_data_multi_task.append(train_data)
                    else:
                        logging.warning(
                            f'重放缓冲区中的数据不足以采样mini-batch: '
                            f'batch_size: {batch_size}, replay_buffer: {replay_buffer}'
                        )
                        break

                if train_data_multi_task:
                    learn_kwargs = {'task_weights': None,"train_iter":learner.train_iter}
                    
                    # DDP automatically synchronizes gradients and parameters during training.
                    log_vars = learner.train(train_data_multi_task, envstep_multi_task, policy_kwargs=learn_kwargs)

                    # Check if task_exploitation_weight needs to be calculated.
                    if i == 0:
                        # Calculate task weights.
                        try:
                            dist.barrier()  # Wait for all processes to synchronize.
                            if cfg.policy.use_task_exploitation_weight:  # Use obs loss now, new polish.
                                # Gather obs_loss from all tasks.
                                all_obs_loss = [None for _ in range(world_size)]
                                # Build obs_loss data for the current process's tasks.
                                merged_obs_loss_task = {}
                                for cfg, replay_buffer in zip(cfgs, game_buffers):
                                    task_id = cfg.policy.task_id
                                    if f'noreduce_obs_loss_task{task_id}' in log_vars[0]:
                                        merged_obs_loss_task[task_id] = log_vars[0][
                                            f'noreduce_obs_loss_task{task_id}']
                                # Gather obs_loss data from all processes.
                                dist.all_gather_object(all_obs_loss, merged_obs_loss_task)
                                # Merge obs_loss data from all processes.
                                global_obs_loss_task = {}
                                for obs_loss_task in all_obs_loss:
                                    if obs_loss_task:
                                        global_obs_loss_task.update(obs_loss_task)
                                # Calculate global task weights.
                                if global_obs_loss_task:
                                    task_exploitation_weight = compute_task_weights(
                                        global_obs_loss_task,
                                        option="rank",
                                        # TODO: Decide whether to use the temperature scheduler here.
                                        temperature=1,
                                    )
                                    # Broadcast task weights to all processes.
                                    dist.broadcast_object_list([task_exploitation_weight], src=0)
                                    print(
                                        f"rank{rank}, task_exploitation_weight (按 task_id 排列): {task_exploitation_weight}")
                                else:
                                    logging.warning(f"Rank {rank}: 未能计算全局 obs_loss 任务权重，obs_loss 数据为空。")
                                    task_exploitation_weight = None
                            else:
                                task_exploitation_weight = None
                            # Update training parameters to include the calculated task weights.
                            learn_kwargs['task_weight'] = task_exploitation_weight
                        except Exception as e:
                            logging.error(f'Rank {rank}: 同步任务权重失败，错误: {e}')
                            raise e  # Re-raise the exception for external capture and analysis.

                    if cfg.policy.use_priority:
                        for idx, (cfg, replay_buffer) in enumerate(zip(cfgs, game_buffers)):
                            # Update task-specific replay buffer priorities.
                            task_id = cfg.policy.task_id
                            # replay_buffer.update_priority(
                            #     train_data_multi_task[idx],
                            #     log_vars[0][f'value_priority_task{task_id}']
                            # )
                            replay_buffer.update_priority(
                                train_data_multi_task[idx],
                                log_vars[0][f'noreduce_value_priority_task{task_id}']
                            )

                            # current_priorities = log_vars[0][f'value_priority_task{task_id}']
                            # mean_priority = np.mean(current_priorities)
                            # std_priority = np.std(current_priorities)

                            # alpha = 0.1  # Smoothing factor
                            # if f'running_mean_priority_task{task_id}' not in value_priority_tasks:
                            #     value_priority_tasks[f'running_mean_priority_task{task_id}'] = mean_priority
                            # else:
                            #     value_priority_tasks[f'running_mean_priority_task{task_id}'] = (
                            #             alpha * mean_priority +
                            #             (1 - alpha) * value_priority_tasks[f'running_mean_priority_task{task_id}']
                            #     )

                            # # Use running mean to calculate normalized priorities.
                            # running_mean_priority = value_priority_tasks[f'running_mean_priority_task{task_id}']
                            # normalized_priorities = (current_priorities - running_mean_priority) / (
                            #             std_priority + 1e-6)

                            # # If needed, update the replay buffer with normalized priorities.
                            # # replay_buffer.update_priority(train_data_multi_task[idx], normalized_priorities)

                            # # Log priority statistics.
                            # if cfg.policy.print_task_priority_logs:
                            #     print(f"任务 {task_id} - 平均优先级: {mean_priority:.8f}, "
                            #           f"运行平均优先级: {running_mean_priority:.8f}, "
                            #           f"标准差: {std_priority:.8f}")

        train_epoch += 1
        policy.recompute_pos_emb_diff_and_clear_cache()

        # Synchronize all ranks to ensure they have completed training.
        try:
            dist.barrier()
            logging.info(f'Rank {rank}: 通过训练后的同步障碍')
        except Exception as e:
            logging.error(f'Rank {rank}: 同步障碍失败，错误: {e}')
            break

        # Check for termination conditions.
        try:
            local_envsteps = [collector.envstep for collector in collectors]
            total_envsteps = [None for _ in range(world_size)]
            dist.all_gather_object(total_envsteps, local_envsteps)

            all_envsteps = torch.cat([torch.tensor(envsteps, device=cfg.policy.device) for envsteps in total_envsteps])
            max_envstep_reached = torch.all(all_envsteps >= max_env_step)

            # Gather train_iter from all processes.
            global_train_iter = torch.tensor([learner.train_iter], device=cfg.policy.device)
            all_train_iters = [torch.zeros_like(global_train_iter) for _ in range(world_size)]
            dist.all_gather(all_train_iters, global_train_iter)

            max_train_iter_reached = torch.any(torch.stack(all_train_iters) >= max_train_iter)

            if max_envstep_reached.item() or max_train_iter_reached.item():
                logging.info(f'Rank {rank}: 达到终止条件')
                dist.barrier()  # Ensure all processes synchronize before exiting.
                break
        except Exception as e:
            logging.error(f'Rank {rank}: 终止检查失败，错误: {e}')
            break

    # Call the learner's after_run hook.
    learner.call_hook('after_run')
    return policy