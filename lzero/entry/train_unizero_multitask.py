import logging
import os
from functools import partial
from typing import Tuple, Optional, List, Dict, Any
import concurrent.futures
import torch
import numpy as np
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy, Policy
from ding.utils import set_pkg_seed, EasyTimer
from ding.worker import BaseLearner

from lzero.entry.utils import log_buffer_memory_usage, TemperatureScheduler
from lzero.policy import visit_count_temperature
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.worker import MuZeroCollector as Collector

# Set timeout (seconds)
TIMEOUT = 12000
timer = EasyTimer()

def safe_eval(
        evaluator: Evaluator,
        learner: BaseLearner,
        collector: Collector
) -> Tuple[Optional[bool], Optional[float]]:
    """
    Safely execute the evaluation task to avoid timeout (non-DDP version).
    Args:
        evaluator (Evaluator): The evaluator instance.
        learner (BaseLearner): The learner instance.
        collector (Collector): The data collector instance.
    Returns:
        Tuple[Optional[bool], Optional[float]]: If evaluation succeeds, returns the stop flag and reward; 
        otherwise returns (None, None).
    """
    try:
        print(f"========= Evaluation Started =========")
        # Reset stop_event to ensure it is unset before each evaluation
        evaluator.stop_event.clear()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit the evaluation task
            future = executor.submit(evaluator.eval, learner.save_checkpoint, learner.train_iter, collector.envstep)
            try:
                stop, reward = future.result(timeout=TIMEOUT)
            except concurrent.futures.TimeoutError:
                # Timeout occurred, set stop_event
                evaluator.stop_event.set()
                print(f"Evaluation operation timed out after {TIMEOUT} seconds.")
                return None, None

        print(f"====== Evaluation Finished ======")
        return stop, reward
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        return None, None

def allocate_batch_size_local(
        cfgs: List[Dict[str, Any]],
        game_buffers,
        alpha: float = 1.0,
        clip_scale: int = 1
) -> List[int]:
    """
    Allocate batch_size inversely proportional to the number of collected episodes 
    for different tasks (non-DDP version).
    Args:
        cfgs (List[Dict[str, Any]]): Configuration list for each task.
        game_buffers (List[Any]): Replay buffer instances for each task (use Any to avoid specific type dependency).
        alpha (float, optional): Hyperparameter controlling the degree of inverse proportionality. Default is 1.0.
        clip_scale (int, optional): Dynamic adjustment scale factor. Default is 1.
    Returns:
        List[int]: The allocated batch_size list.
    """
    # Extract the number of collected episodes for each task (assuming buffer has this attribute)
    buffer_num_of_collected_episodes = [buffer.num_of_collected_episodes for buffer in game_buffers]
    print(f'Collected episodes for all local tasks: {buffer_num_of_collected_episodes}')

    # Compute the inverse weights for each task
    inv_episodes = np.array([1.0 / (episodes + 1) for episodes in buffer_num_of_collected_episodes])
    inv_sum = np.sum(inv_episodes)

    # Compute the total batch_size (taken from the first task's configuration)
    # Assume total_batch_size refers to the total batch size required by the current process
    total_batch_size = cfgs[0].policy.total_batch_size

    # Dynamic adjustment: minimum and maximum batch_size range
    num_local_tasks = len(cfgs)
    avg_batch_size = total_batch_size / max(num_local_tasks, 1) # Avoid division by zero
    min_batch_size = avg_batch_size / clip_scale
    max_batch_size = avg_batch_size * clip_scale

    # Dynamically adjust alpha to make batch_size changes smoother
    task_weights = (inv_episodes / inv_sum) ** alpha
    batch_sizes = total_batch_size * task_weights

    # Clip batch_size within [min_batch_size, max_batch_size]
    batch_sizes = np.clip(batch_sizes, min_batch_size, max_batch_size)

    # Ensure batch_size is an integer
    batch_sizes = [int(size) for size in batch_sizes]

    return batch_sizes

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
        logits = -beta * raw_weights 
        softmax_weights = F.softmax(logits, dim=0).numpy()
        weights = dict(zip(task_ids, softmax_weights))
    else:
        scaled_weights = raw_weights ** (1 / max(temperature, epsilon))
        total_weight = scaled_weights.sum()
        normalized_weights = scaled_weights / max(total_weight, epsilon) # Avoid division by zero
        weights = dict(zip(task_ids, normalized_weights.numpy()))

    for task_id in weights:
        weights[task_id] = max(min(weights[task_id], clip_max), clip_min)

    return weights

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
        Entry point for UniZero multi-task training (non-DDP version).
    Args:
        - input_cfg_list (:obj:`List[Tuple[int, Tuple[dict, dict]]]`): Configuration list for different tasks.
        - seed (:obj:`int`): Random seed.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - model_path (:obj:`Optional[str]`): Path to the pre-trained model.
        - max_train_iter (:obj:`Optional[int]`): Maximum number of policy update iterations.
        - max_env_step (:obj:`Optional[int]`): Maximum number of collected environment interaction steps.
    Returns:
        - policy (:obj:`Policy`): The converged policy.
    """
    # Initialize temperature scheduler (unchanged)
    initial_temperature = 10.0
    final_temperature = 1.0
    threshold_steps = int(1e4)
    temperature_scheduler = TemperatureScheduler(
        initial_temp=initial_temperature,
        final_temp=final_temperature,
        threshold_steps=threshold_steps,
        mode='linear'
    )

    # Handle all tasks in a single process
    tasks = input_cfg_list
    total_tasks = len(tasks)
    print(f"Handling all {total_tasks} tasks in a single process.")

    cfgs = []
    game_buffers = []
    collector_envs = []
    evaluator_envs = []
    collectors = []
    evaluators = []

    # Ensure at least one task is provided
    if not tasks:
        logging.error("No task configurations provided. Training cannot proceed.")
        return None

    # Use the first task's configuration to create the shared policy and learner
    task_id_first, [cfg_first, create_cfg_first] = tasks[0]

    assert create_cfg_first.policy.type in ['unizero_multitask', 'sampled_unizero_multitask'], "train_unizero_multitask entry currently only supports 'unizero_multitask' or 'sampled_unizero_multitask'"


    GameBuffer = None
    if create_cfg_first.policy.type == 'unizero_multitask':
        from lzero.mcts import UniZeroGameBuffer as GB
        GameBuffer = GB
    elif create_cfg_first.policy.type == 'sampled_unizero_multitask':
        from lzero.mcts import SampledUniZeroGameBuffer as SGB
        GameBuffer = SGB
    else:
        raise NotImplementedError(f"Policy type {create_cfg_first.policy.type} not fully supported for GameBuffer import.")

    cfg_first.policy.device = 'cuda' if cfg_first.policy.cuda and torch.cuda.is_available() else 'cpu'
    logging.info(f'Using device: {cfg_first.policy.device}')

    # Compile the main config (only for creating policy and learner)
    # Note: we compile once here, but later re-compile per-task configs
    compiled_cfg_first = compile_config(cfg_first, seed=seed, env=None, auto=True, create_cfg=create_cfg_first, save_cfg=True)

    # Create shared policy
    policy = create_policy(compiled_cfg_first.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    if model_path is not None:
        logging.info(f'Loading pretrained model: {model_path}')
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=compiled_cfg_first.policy.device))
        logging.info(f'Finished loading model: {model_path}')

    log_dir = os.path.join('./{}/log/'.format(compiled_cfg_first.exp_name), 'serial')
    tb_logger = SummaryWriter(log_dir)

    # Create shared learner
    learner = BaseLearner(compiled_cfg_first.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=compiled_cfg_first.exp_name)

    # Process each task
    for local_task_id, (task_id, [cfg, create_cfg]) in enumerate(tasks):
        # Set random seed per task
        current_seed = seed + task_id
        cfg.policy.device = 'cuda' if cfg.policy.cuda and torch.cuda.is_available() else 'cpu'
        # Compile per-task config
        cfg = compile_config(cfg, seed=current_seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
        # Get policy config
        policy_config = cfg.policy
        policy_config.task_id = task_id # explicitly set task_id
        policy.collect_mode.get_attribute('cfg').n_episode = policy_config.n_episode
        policy.eval_mode.get_attribute('cfg').n_episode = policy_config.n_episode

        # Create environments
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
        collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
        evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
        collector_env.seed(current_seed)
        evaluator_env.seed(current_seed, dynamic_seed=False)
        set_pkg_seed(current_seed, use_cuda=cfg.policy.cuda)

        # Create buffer, collector, and evaluator
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

    learner.call_hook('before_run')
    value_priority_tasks = {}

    buffer_reanalyze_count = 0
    train_epoch = 0
    
    reanalyze_batch_size = compiled_cfg_first.policy.reanalyze_batch_size
    update_per_collect = compiled_cfg_first.policy.update_per_collect
    task_complexity_weight = compiled_cfg_first.policy.task_complexity_weight
    use_task_exploitation_weight = compiled_cfg_first.policy.use_task_exploitation_weight
    task_exploitation_weight = None

    task_rewards = {}  
    while True:
        # Dynamically allocate batch_size 
        if compiled_cfg_first.policy.allocated_batch_sizes:
            clip_scale = np.clip(1 + (3 * train_epoch / 1000), 1, 4)
            allocated_batch_sizes_list = allocate_batch_size_local(cfgs, game_buffers, alpha=1.0, clip_scale=clip_scale)
            # Convert list to {task_id: batch_size}
            allocated_batch_sizes_dict = {cfg.policy.task_id: size for cfg, size in zip(cfgs, allocated_batch_sizes_list)}
            print("Allocated batch_sizes: ", allocated_batch_sizes_dict)
            policy._cfg.batch_size = allocated_batch_sizes_dict
            for i, cfg in enumerate(cfgs):
                cfg.policy.batch_size = allocated_batch_sizes_dict

        # Iterate over tasks for data collection and evaluation
        for idx, (cfg, collector, evaluator, replay_buffer) in enumerate(
                zip(cfgs, collectors, evaluators, game_buffers)):

            current_task_id = cfg.policy.task_id

            # Log buffer memory usage
            log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger, current_task_id)

            policy_config = cfg.policy
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

            if learner.train_iter == 0 or evaluator.should_eval(learner.train_iter): # only for debug
                print('=' * 20)
                print(f'Evaluating task_id: {current_task_id}...')
                # Reset evaluator policy state
                evaluator._policy.reset(reset_init_data=True, task_id=current_task_id)

                # Perform safe evaluation (non-DDP version)
                stop, reward = safe_eval(evaluator, learner, collector)
                if stop is None or reward is None:
                    print(f"Evaluation failed or timed out, task_id: {current_task_id}, continuing training...")
                    task_rewards[current_task_id] = float('inf')
                else:
                    try:
                        eval_mean_reward = reward.get('eval_episode_return_mean', float('inf'))
                        print(f"Evaluation reward for task {current_task_id}: {eval_mean_reward}")
                        task_rewards[current_task_id] = eval_mean_reward
                    except Exception as e:
                        print(f"Error extracting reward for task {current_task_id}: {e}")
                        task_rewards[current_task_id] = float('inf')

            print('=' * 20)
            print(f'Starting data collection for task_id: {current_task_id}...')
            print(f'cfg.policy.task_id={current_task_id}')

            # Reset collector policy state
            collector._policy.reset(reset_init_data=True, task_id=current_task_id)
            
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)

            replay_buffer.push_game_segments(new_data)
            replay_buffer.remove_oldest_data_to_fit()

            if policy_config.buffer_reanalyze_freq >= 1:
                if update_per_collect is None or update_per_collect == 0:
                    logging.warning("update_per_collect undefined or zero, cannot compute reanalyze_interval")
                    reanalyze_interval = float('inf')
            
                else:
                    reanalyze_interval = update_per_collect // policy_config.buffer_reanalyze_freq
            else: 
                reanalyze_interval = float('inf') 
                if train_epoch > 0 and policy_config.buffer_reanalyze_freq > 0 and \
                    train_epoch % int(1 / policy_config.buffer_reanalyze_freq) == 0 and \
                    replay_buffer.get_num_of_transitions() // policy_config.num_unroll_steps > int(reanalyze_batch_size / policy_config.reanalyze_partition):
                    with timer:
                        replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                    buffer_reanalyze_count += 1
                    logging.info(f'Buffer reanalyze count: {buffer_reanalyze_count}, time cost: {timer.value}')

            logging.info(f'Finished data collection for task {current_task_id}')

        not_enough_data = any(
            game_buffers[idx].get_num_of_transitions() < policy._cfg.batch_size[cfg.policy.task_id]
            for idx, (cfg, replay_buffer) in enumerate(zip(cfgs, game_buffers))
        )

        current_temperature_task_weight = temperature_scheduler.get_temperature(learner.train_iter)

        # Compute task weights
        if task_complexity_weight:
            task_weights = compute_task_weights(task_rewards, temperature=current_temperature_task_weight)
        else:
            task_weights = None

        if not not_enough_data:
            for i in range(update_per_collect): 
                train_data_multi_task = []
                envstep_this_epoch = 0 

                for idx, (cfg, collector, replay_buffer) in enumerate(zip(cfgs, collectors, game_buffers)):
                    current_task_id = cfg.policy.task_id
                    current_batch_size = policy._cfg.batch_size[current_task_id]
                    
                    if current_batch_size == 0:
                        logging.warning(f"Task {current_task_id} batch_size is 0, skipping sampling.")
                        continue

                    if replay_buffer.get_num_of_transitions() >= current_batch_size:
                        policy_config = cfg.policy
                        if policy_config.buffer_reanalyze_freq >= 1:
                            if update_per_collect is not None and update_per_collect > 0: 
                                reanalyze_interval = update_per_collect // policy_config.buffer_reanalyze_freq
                                if i % reanalyze_interval == 0 and \
                                        replay_buffer.get_num_of_transitions() // policy_config.num_unroll_steps > int(reanalyze_batch_size / policy_config.reanalyze_partition):
                                    with timer:
                                        replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                                    buffer_reanalyze_count += 1
                                    logging.info(f'Buffer reanalyze count: {buffer_reanalyze_count}, time cost: {timer.value}')

                        train_data = replay_buffer.sample(current_batch_size, policy)
                        train_data.append(current_task_id)  
                        train_data_multi_task.append(train_data)
                        envstep_this_epoch += collector.envstep 
                    else:
                        logging.warning(
                            f'Not enough data for task {current_task_id}: '
                            f'batch_size: {current_batch_size}, buffer: {replay_buffer.get_num_of_transitions()}'
                        )

                if train_data_multi_task:
                    learn_kwargs = {'task_weights': task_weights} 
                    log_vars = learner.train(train_data_multi_task, envstep_this_epoch, policy_kwargs=learn_kwargs)

                    # --- Compute and update task_exploitation_weight ---
                    if i == 0 and use_task_exploitation_weight:
                        local_obs_loss_task = {}
                        for cfg in cfgs:
                            task_id = cfg.policy.task_id
                            loss_key = f'noreduce_obs_loss_task{task_id}'
                            if log_vars and loss_key in log_vars[0]:
                                local_obs_loss_task[task_id] = log_vars[0][loss_key]

                        if local_obs_loss_task:
                            task_exploitation_weight = compute_task_weights(
                                local_obs_loss_task,
                                option="rank",
                                temperature=1, 
                                reverse=True
                            )
                            print(f"Locally computed task_exploitation_weight (by task_id): {task_exploitation_weight}")
                        
                        else:
                            logging.warning("Unable to compute local task_exploitation_weight, obs_loss is empty or invalid.")
                            task_exploitation_weight = None

                        learn_kwargs['task_exploitation_weight'] = task_exploitation_weight

                    if compiled_cfg_first.policy.use_priority:
                        if log_vars: 
                            for batch_idx, (cfg, replay_buffer) in enumerate(zip(cfgs, game_buffers)):
                                task_id = cfg.policy.task_id
                                priority_key = f'value_priority_task{task_id}'
                                if priority_key in log_vars[0]:
                                    if batch_idx < len(train_data_multi_task):
                                        try:
                                            replay_buffer.update_priority(
                                                train_data_multi_task[batch_idx], 
                                                log_vars[0][priority_key]
                                            )
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
                                            if policy_config.print_task_priority_logs:
                                                print(f"Task {task_id} - Mean priority: {mean_priority:.8f}, "
                                                    f"Running mean priority: {running_mean_priority:.8f}, "
                                                    f"Std: {std_priority:.8f}")
                                        except Exception as e:
                                            logging.error(f"Error updating priority for task {task_id}: {e}")
                                    else:
                                        logging.warning(f"Cannot update priority for task {task_id}, missing data in train_data_multi_task.")
                                else:
                                    logging.warning(f"Priority key '{priority_key}' not found for task {task_id} in log_vars[0]")
                        else:
                            logging.warning("log_vars is empty, cannot update priorities.")
        train_epoch += 1
        # Check termination conditions
        local_max_envstep = max(collector.envstep for collector in collectors) if collectors else 0
        max_envstep_reached = local_max_envstep >= max_env_step
        max_train_iter_reached = learner.train_iter >= max_train_iter

        if max_envstep_reached or max_train_iter_reached:
            logging.info(f'Termination condition reached: env_step ({local_max_envstep}/{max_env_step}) or train_iter ({learner.train_iter}/{max_train_iter})')
            break

        if hasattr(policy, 'recompute_pos_emb_diff_and_clear_cache'):
            policy.recompute_pos_emb_diff_and_clear_cache()

    learner.call_hook('after_run')
    return policy