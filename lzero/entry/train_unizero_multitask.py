
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
from lzero.entry.utils import (
    EVALUATION_TIMEOUT,
    TemperatureScheduler,
    allocate_batch_size,
    compute_task_weights,
    compute_unizero_mt_normalized_stats,
    log_buffer_memory_usage,
    safe_eval,
    symlog,
    inv_symlog,
)

from lzero.entry.utils import log_buffer_memory_usage, TemperatureScheduler
from lzero.policy import visit_count_temperature
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.worker import MuZeroCollector as Collector

# Set timeout (seconds)
timer = EasyTimer()

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

    task_exploitation_weight = None
    task_rewards = {}  
    
    while True:
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

            if learner.train_iter > 0 and evaluator.should_eval(learner.train_iter): # only for debug
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
            logging.info(f'Finished data collection for task {cfg.policy.task_id}, collected {len(new_data[0]) if new_data else 0} segments')

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
                    learn_kwargs = {'task_weights': task_weights, "train_iter": learner.train_iter} 
                    log_vars = learner.train(train_data_multi_task, envstep_this_epoch, policy_kwargs=learn_kwargs)


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
