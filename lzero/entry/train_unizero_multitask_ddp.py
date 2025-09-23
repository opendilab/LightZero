import logging
import os
from functools import partial
from typing import Tuple, Optional, List
import concurrent.futures
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from tensorboardX import SummaryWriter

from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import set_pkg_seed, get_rank, get_world_size, EasyTimer
from ding.worker import BaseLearner

from lzero.entry.utils import log_buffer_memory_usage, TemperatureScheduler, symlog, inv_symlog
from lzero.policy import visit_count_temperature
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.worker import MuZeroCollector as Collector

# Set timeout (seconds)
TIMEOUT = 12000
timer = EasyTimer()

def safe_eval(
        evaluator: Evaluator,
        learner: BaseLearner,
        collector: Collector,
        rank: int,
        world_size: int
) -> Tuple[Optional[bool], Optional[float]]:
    """
    Safely execute the evaluation task to avoid timeout.
    Args:
        evaluator (Evaluator): The evaluator instance.
        learner (BaseLearner): The learner instance.
        collector (Collector): The data collector instance.
        rank (int): Rank of the current process.
        world_size (int): Total number of processes.
    Returns:
        Tuple[Optional[bool], Optional[float]]: If evaluation succeeds, returns the stop flag and reward; 
        otherwise returns (None, None).
    """
    try:
        print(f"========= Evaluation Started Rank {rank}/{world_size} ==========")
        # Reset stop_event to ensure it is cleared before each evaluation
        evaluator.stop_event.clear()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(evaluator.eval, learner.save_checkpoint, learner.train_iter, collector.envstep)
            try:
                stop, reward = future.result(timeout=TIMEOUT)
            except concurrent.futures.TimeoutError:
                # Timeout occurred, set stop_event
                evaluator.stop_event.set()
                print(f"Evaluation timed out on Rank {rank}/{world_size}, exceeded {TIMEOUT} seconds.")
                return None, None

        print(f"====== Evaluation Finished Rank {rank}/{world_size} ======")
        return stop, reward
    except Exception as e:
        print(f"Error occurred during evaluation on Rank {rank}/{world_size}: {e}")
        return None, None


def allocate_batch_size(
        cfgs: List[dict],
        game_buffers,
        alpha: float = 1.0,
        clip_scale: int = 1
) -> List[int]:
    """
    Allocate batch_size inversely proportional to the number of collected episodes 
    for different tasks, and dynamically adjust the batch_size range to improve 
    training stability and efficiency.
    Args:
        cfgs (List[dict]): Configuration list for each task.
        game_buffers (List[GameBuffer]): Replay buffer instances for each task.
        alpha (float, optional): Hyperparameter controlling the degree of inverse proportionality. Default is 1.0.
        clip_scale (int, optional): Dynamic adjustment scale factor. Default is 1.
    Returns:
        List[int]: The allocated batch_size list.
    """
    # Extract the number of collected episodes for each task (assuming buffer has this attribute)
    buffer_num_of_collected_episodes = [buffer.num_of_collected_episodes for buffer in game_buffers]

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # Gather collected episodes from all ranks
    all_task_num_of_collected_episodes = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(all_task_num_of_collected_episodes, buffer_num_of_collected_episodes)

    # Flatten into a single list
    all_task_num_of_collected_episodes = [
        episode for sublist in all_task_num_of_collected_episodes for episode in sublist
    ]
    if rank == 0:
        print(f'Collected episodes for all tasks: {all_task_num_of_collected_episodes}')

    # Compute inverse weights for each task
    inv_episodes = np.array([1.0 / (episodes + 1) for episodes in all_task_num_of_collected_episodes])
    inv_sum = np.sum(inv_episodes)

    # Compute total batch_size
    total_batch_size = cfgs[0].policy.total_batch_size

    # Dynamic adjustment: min and max batch_size range
    avg_batch_size = total_batch_size / world_size
    min_batch_size = avg_batch_size / clip_scale
    max_batch_size = avg_batch_size * clip_scale

    # Dynamically adjust alpha to smooth batch_size variation
    task_weights = (inv_episodes / inv_sum) ** alpha
    batch_sizes = total_batch_size * task_weights

    # Clip batch_size within [min_batch_size, max_batch_size]
    batch_sizes = np.clip(batch_sizes, min_batch_size, max_batch_size)
    batch_sizes = [int(size) for size in batch_sizes]

    return batch_sizes


# Global max and min (for "run-max-min")
GLOBAL_MAX = -float('inf')
GLOBAL_MIN = float('inf')

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
    Improved task weight computation function. 
    Supports multiple normalization methods, Softmax, proportional/inverse weighting, 
    and adds clipping functionality for weight ranges.
    Args:
        task_rewards (dict): Dictionary of task rewards or losses, 
            with task_id as key and the value as the reward/loss.
        option (str): Normalization method. Options are "symlog", "max-min", "run-max-min", "rank", "none".
        epsilon (float): Small constant to avoid division by zero.
        temperature (float): Temperature parameter controlling weight distribution.
        use_softmax (bool): Whether to use Softmax for weight allocation.
        reverse (bool): If True, weights are inversely proportional to values; 
            if False, weights are directly proportional.
        clip_min (float): Minimum value for clipping weights.
        clip_max (float): Maximum value for clipping weights.
    Returns:
        dict: Normalized weights for each task, with task_id as key and the normalized weight as value.
    """
    global GLOBAL_MAX, GLOBAL_MIN
    if not task_rewards:
        return {}

    # Step 1: Construct tensor from task_rewards values
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
        # Rank normalization
        # Rank is based on sorted order; 1 = smallest, higher rank = larger value
        sorted_indices = torch.argsort(rewards_tensor)
        scaled_rewards = torch.empty_like(rewards_tensor)
        rank_values = torch.arange(1, len(rewards_tensor) + 1, dtype=torch.float32) 
        scaled_rewards[sorted_indices] = rank_values
    elif option == "none":
        scaled_rewards = rewards_tensor
    else:
        raise ValueError(f"Unsupported option: {option}")

    # Step 2: Compute proportional or inverse weights
    if not reverse:
        raw_weights = scaled_rewards
    else:
        scaled_rewards = torch.clamp(scaled_rewards, min=epsilon)
        raw_weights = 1.0 / scaled_rewards

    # Step 3: Apply Softmax or direct normalization
    if use_softmax:
        # Softmax weighting
        beta = 1.0 / max(temperature, epsilon)  # avoid division by zero
        logits = -beta * raw_weights
        softmax_weights = F.softmax(logits, dim=0).numpy()
        weights = dict(zip(task_ids, softmax_weights))
    else:
        scaled_weights = raw_weights ** (1 / max(temperature, epsilon)) 
        total_weight = scaled_weights.sum()
        normalized_weights = scaled_weights / total_weight
        weights = dict(zip(task_ids, normalized_weights.numpy()))

    # Step 4: Clip weights within [clip_min, clip_max]
    for task_id in weights:
        weights[task_id] = max(min(weights[task_id], clip_max), clip_min)

    return weights

def train_unizero_multitask_ddp(
        input_cfg_list: List[Tuple[int, Tuple[dict, dict]]],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':
    """
    Overview:
        Entry point for UniZero training. The goal is to improve the planning ability 
        of reinforcement learning agents by addressing the limitations of MuZero-like 
        algorithms in environments that require capturing long-term dependencies.
        For more details, refer to https://arxiv.org/abs/2406.10667.
    Args:
        - input_cfg_list (:obj:`List[Tuple[int, Tuple[dict, dict]]]`): Configuration list for different tasks.
        - seed (:obj:`int`): Random seed.
        - model (:obj:`Optional[torch.nn.Module]`): An instance of torch.nn.Module.
        - model_path (:obj:`Optional[str]`): Path to the pretrained model checkpoint file.
        - max_train_iter (:obj:`Optional[int]`): Maximum number of policy update iterations during training.
        - max_env_step (:obj:`Optional[int]`): Maximum number of collected environment interaction steps.
    Returns:
        - policy (:obj:`Policy`): The converged policy.
    """
    initial_temperature = 10.0
    final_temperature = 1.0
    threshold_steps = int(1e4)  
    temperature_scheduler = TemperatureScheduler(
        initial_temp=initial_temperature,
        final_temp=final_temperature,
        threshold_steps=threshold_steps,
        mode='linear'
    )

    rank = get_rank()
    world_size = get_world_size()

    # Task partitioning
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

    # Ensure at least one task is assigned
    if len(tasks_for_this_rank) == 0:
        logging.warning(f"Rank {rank}: no tasks assigned, continuing execution.")
        # Initialize empty lists to avoid errors in later code
        cfgs, game_buffers, collector_envs, evaluator_envs, collectors, evaluators = [], [], [], [], [], []
    else:
        print(f"Rank {rank}/{world_size}, handling tasks {start_idx} to {end_idx - 1}")

    cfgs = []
    game_buffers = []
    collector_envs = []
    evaluator_envs = []
    collectors = []
    evaluators = []

    if tasks_for_this_rank:
        # Use the first task’s config to create a shared policy
        task_id, [cfg, create_cfg] = tasks_for_this_rank[0]

        for config in tasks_for_this_rank:
            config[1][0].policy.task_num = tasks_per_rank

        assert create_cfg.policy.type in ['unizero_multitask',
                                        'sampled_unizero_multitask'], "train_unizero entry 目前仅支持 'unizero_multitask'"

        if create_cfg.policy.type == 'unizero_multitask':
            from lzero.mcts import UniZeroGameBuffer as GameBuffer
        if create_cfg.policy.type == 'sampled_unizero_multitask':
            from lzero.mcts import SampledUniZeroGameBuffer as GameBuffer

        cfg.policy.device = cfg.policy.model.world_model_cfg.device if torch.cuda.is_available() else 'cpu'
        logging.info(f'Configured device: {cfg.policy.device}')

        cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
        # Create shared policy
        policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

        if model_path is not None:
            logging.info(f'Loading pretrained model: {model_path}')
            policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))
            logging.info(f'Finished loading pretrained model: {model_path}')

        log_dir = os.path.join('./{}/log'.format(cfg.exp_name), f'serial_rank_{rank}')
        tb_logger = SummaryWriter(log_dir)

        # Create shared learner
        learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
        policy_config = cfg.policy

        # Handle each task assigned to this rank
        for local_task_id, (task_id, [cfg, create_cfg]) in enumerate(tasks_for_this_rank):
            cfg.policy.device = 'cuda' if cfg.policy.cuda and torch.cuda.is_available() else 'cpu'
            cfg = compile_config(cfg, seed=seed + task_id, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
            policy_config = cfg.policy
            policy.collect_mode.get_attribute('cfg').n_episode = policy_config.n_episode
            policy.eval_mode.get_attribute('cfg').n_episode = policy_config.n_episode

            # Create environments
            env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
            collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
            evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
            collector_env.seed(cfg.seed + task_id)
            evaluator_env.seed(cfg.seed + task_id, dynamic_seed=False)
            set_pkg_seed(cfg.seed + task_id, use_cuda=cfg.policy.cuda)

            # Create game buffer, collector, and evaluator
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

    task_complexity_weight = cfg.policy.task_complexity_weight
    use_task_exploitation_weight = cfg.policy.use_task_exploitation_weight
    task_exploitation_weight = None

    # Create task reward dictionary
    task_rewards = {}  # {task_id: reward}
    
    while True:
        # Dynamically adjust batch_size
        if cfg.policy.allocated_batch_sizes:
            clip_scale = np.clip(1 + (3 * train_epoch / 1000), 1, 4)
            allocated_batch_sizes = allocate_batch_size(cfgs, game_buffers, alpha=1.0, clip_scale=clip_scale)
            if rank == 0:
                print("Allocated batch_sizes: ", allocated_batch_sizes)
            for idx, (cfg, collector, evaluator, replay_buffer) in enumerate(
                    zip(cfgs, collectors, evaluators, game_buffers)):
                cfg.policy.batch_size = allocated_batch_sizes
                policy._cfg.batch_size = allocated_batch_sizes

        # Perform data collection and evaluation for each task on this rank
        for idx, (cfg, collector, evaluator, replay_buffer) in enumerate(
                zip(cfgs, collectors, evaluators, game_buffers)):

            # Log buffer memory usage
            log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger, cfg.policy.task_id)

            collect_kwargs = {
                'temperature': visit_count_temperature(
                    policy_config.manual_temperature_decay,
                    policy_config.fixed_temperature_value,
                    policy_config.threshold_training_steps_for_final_temperature,
                    trained_steps=learner.train_iter
                ),
                'epsilon': 0.0
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
                print(f'Rank {rank} evaluating task_id: {cfg.policy.task_id}...')
                evaluator._policy.reset(reset_init_data=True, task_id=cfg.policy.task_id)

                # Perform safe evaluation
                stop, reward = safe_eval(evaluator, learner, collector, rank, world_size)
                if stop is None or reward is None:
                    print(f"Rank {rank} encountered an issue during evaluation, continuing training...")
                    task_rewards[cfg.policy.task_id] = float('inf')  # Assign max difficulty if evaluation fails
                else:
                    try:
                        eval_mean_reward = reward.get('eval_episode_return_mean', float('inf'))
                        print(f"Evaluation reward for task {cfg.policy.task_id}: {eval_mean_reward}")
                        task_rewards[cfg.policy.task_id] = eval_mean_reward
                    except Exception as e:
                        print(f"Error extracting evaluation reward: {e}")
                        task_rewards[cfg.policy.task_id] = float('inf')  # Assign max reward if error occurs


            print('=' * 20)
            print(f'Starting data collection for Rank {rank}, task_id: {cfg.policy.task_id}...')
            print(f'Rank {rank}: cfg.policy.task_id={cfg.policy.task_id} ')

            # Reset policy state before each collection (important for multi-task setups)
            collector._policy.reset(reset_init_data=True, task_id=cfg.policy.task_id)
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)

            replay_buffer.push_game_segments(new_data)
            replay_buffer.remove_oldest_data_to_fit()

            if cfg.policy.buffer_reanalyze_freq >= 1:
                reanalyze_interval = update_per_collect // cfg.policy.buffer_reanalyze_freq
            else:
                if train_epoch > 0 and train_epoch % int(1 / cfg.policy.buffer_reanalyze_freq) == 0 and \
                        replay_buffer.get_num_of_transitions() // cfg.policy.num_unroll_steps > int(
                    reanalyze_batch_size / cfg.policy.reanalyze_partition):
                    with timer:
                        replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                    buffer_reanalyze_count += 1
                    logging.info(f'Buffer reanalyze count: {buffer_reanalyze_count}')
                    logging.info(f'Buffer reanalyze time cost: {timer.value}')

            logging.info(f'Rank {rank}: Finished data collection for task {cfg.policy.task_id}')

        # Check if there is enough data for training
        not_enough_data = any(
            replay_buffer.get_num_of_transitions() < cfgs[0].policy.total_batch_size / world_size
            for replay_buffer in game_buffers
        )

        # Get current temperature
        current_temperature_task_weight = temperature_scheduler.get_temperature(learner.train_iter)

        # Compute task weights
        try:
            dist.barrier()
            if task_complexity_weight:
                all_task_rewards = [None for _ in range(world_size)]
                dist.all_gather_object(all_task_rewards, task_rewards)
                merged_task_rewards = {}
                for rewards in all_task_rewards:
                    if rewards:
                        merged_task_rewards.update(rewards)
                task_weights = compute_task_weights(merged_task_rewards, temperature=current_temperature_task_weight)
                dist.broadcast_object_list([task_weights], src=0)
                print(f"Rank {rank}, global task weights (by task_id): {task_weights}")
            else:
                task_weights = None
        except Exception as e:
            logging.error(f'Rank {rank}: Failed to synchronize task weights, error: {e}')
            break

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
                                logging.info(f'Buffer reanalyze count: {buffer_reanalyze_count}')
                                logging.info(f'Buffer reanalyze time cost: {timer.value}')


                        train_data = replay_buffer.sample(batch_size, policy)
                        train_data.append(cfg.policy.task_id) 
                        train_data_multi_task.append(train_data)
                    else:
                        logging.warning(
                            f'Not enough data in replay buffer to sample a mini-batch: '
                            f'batch_size: {batch_size}, replay_buffer: {replay_buffer}'
                        )
                        break

                if train_data_multi_task:
                    learn_kwargs = {'task_weights':task_exploitation_weight}
                    log_vars = learner.train(train_data_multi_task, envstep_multi_task, policy_kwargs=learn_kwargs)

                    # Compute task_exploitation_weight if needed
                    if i == 0:
                        try:
                            dist.barrier() 
                            if use_task_exploitation_weight:
                                all_obs_loss = [None for _ in range(world_size)]
                                merged_obs_loss_task = {}
                                for cfg, replay_buffer in zip(cfgs, game_buffers):
                                    task_id = cfg.policy.task_id
                                    if f'noreduce_obs_loss_task{task_id}' in log_vars[0]:
                                        merged_obs_loss_task[task_id] = log_vars[0][f'noreduce_obs_loss_task{task_id}']
                                dist.all_gather_object(all_obs_loss, merged_obs_loss_task)
                                global_obs_loss_task = {}
                                for obs_loss_task in all_obs_loss:
                                    if obs_loss_task:
                                        global_obs_loss_task.update(obs_loss_task)
                                if global_obs_loss_task:
                                    task_exploitation_weight = compute_task_weights(
                                        global_obs_loss_task,
                                        option="rank",
                                        temperature=1,
                                    )
                                    dist.broadcast_object_list([task_exploitation_weight], src=0)
                                    print(f"Rank {rank}, task_exploitation_weight (by task_id): {task_exploitation_weight}")
                                else:
                                    logging.warning(f"Rank {rank}: Failed to compute global obs_loss task weights, obs_loss data is empty.")
                                    task_exploitation_weight = None
                            else:
                                task_exploitation_weight = None
                            learn_kwargs['task_weight'] = task_exploitation_weight
                        except Exception as e:
                            logging.error(f'Rank {rank}: Failed to synchronize task weights, error: {e}')
                            raise e

                    if cfg.policy.use_priority:
                        for idx, (cfg, replay_buffer) in enumerate(zip(cfgs, game_buffers)):
                            task_id = cfg.policy.task_id
                            replay_buffer.update_priority(
                                train_data_multi_task[idx],
                                log_vars[0][f'value_priority_task{task_id}']
                            )

                            current_priorities = log_vars[0][f'value_priority_task{task_id}']
                            mean_priority = np.mean(current_priorities)
                            std_priority = np.std(current_priorities)

                            alpha = 0.1  # smoothing factor
                            if f'running_mean_priority_task{task_id}' not in value_priority_tasks:
                                value_priority_tasks[f'running_mean_priority_task{task_id}'] = mean_priority
                            else:
                                value_priority_tasks[f'running_mean_priority_task{task_id}'] = (
                                        alpha * mean_priority +
                                        (1 - alpha) * value_priority_tasks[f'running_mean_priority_task{task_id}']
                                )

                            running_mean_priority = value_priority_tasks[f'running_mean_priority_task{task_id}']
                            normalized_priorities = (current_priorities - running_mean_priority) / (std_priority + 1e-6)

                            if cfg.policy.print_task_priority_logs:
                                print(f"Task {task_id} - Mean priority: {mean_priority:.8f}, "
                                    f"Running mean priority: {running_mean_priority:.8f}, "
                                    f"Std: {std_priority:.8f}")

        train_epoch += 1
        policy.recompute_pos_emb_diff_and_clear_cache()

        # Synchronize all ranks after training
        try:
            dist.barrier()
            logging.info(f'Rank {rank}: passed training synchronization barrier')
        except Exception as e:
            logging.error(f'Rank {rank}: synchronization barrier failed, error: {e}')
            break

        # Check termination conditions
        try:
            local_envsteps = [collector.envstep for collector in collectors]
            total_envsteps = [None for _ in range(world_size)]
            dist.all_gather_object(total_envsteps, local_envsteps)

            all_envsteps = torch.cat([torch.tensor(envsteps, device=cfg.policy.device) for envsteps in total_envsteps])
            max_envstep_reached = torch.all(all_envsteps >= max_env_step)

            global_train_iter = torch.tensor([learner.train_iter], device=cfg.policy.device)
            all_train_iters = [torch.zeros_like(global_train_iter) for _ in range(world_size)]
            dist.all_gather(all_train_iters, global_train_iter)

            max_train_iter_reached = torch.any(torch.stack(all_train_iters) >= max_train_iter)

            if max_envstep_reached.item() or max_train_iter_reached.item():
                logging.info(f'Rank {rank}: termination condition reached')
                dist.barrier()
                break
        except Exception as e:
            logging.error(f'Rank {rank}: termination check failed, error: {e}')
            break

    learner.call_hook('after_run')
    return policy