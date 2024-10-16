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
from ding.utils import set_pkg_seed, get_rank
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter

from lzero.entry.utils import log_buffer_memory_usage
from lzero.policy import visit_count_temperature
from lzero.worker import MuZeroCollector as Collector, MuZeroEvaluator as Evaluator
from lzero.mcts import UniZeroGameBuffer as GameBuffer

from line_profiler import line_profiler

#@profile
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
    cfgs = []
    game_buffers = []
    collector_envs = []
    evaluator_envs = []
    collectors = []
    evaluators = []

    task_id, [cfg, create_cfg] = input_cfg_list[0]

    # Ensure the specified policy type is supported
    assert create_cfg.policy.type in ['unizero_multitask'], "train_unizero entry now only supports 'unizero'"

    # Set device based on CUDA availability
    cfg.policy.device = cfg.policy.model.world_model_cfg.device if torch.cuda.is_available() else 'cpu'
    logging.info(f'cfg.policy.device: {cfg.policy.device}')

    # Compile the configuration
    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
    # Create shared policy for all tasks
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    # Load pretrained model if specified
    if model_path is not None:
        logging.info(f'Loading model from {model_path} begin...')
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))
        logging.info(f'Loading model from {model_path} end!')

    # Create SummaryWriter for TensorBoard logging
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')) if get_rank() == 0 else None
    # Create shared learner for all tasks
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

    # TODO task_id = 0:
    policy_config = cfg.policy
    batch_size = policy_config.batch_size[0]

    for task_id, input_cfg in input_cfg_list:
        if task_id > 0:
            # Get the configuration for each task
            cfg, create_cfg = input_cfg
            cfg.policy.device = 'cuda' if cfg.policy.cuda and torch.cuda.is_available() else 'cpu'
            cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
            policy_config = cfg.policy
            policy.collect_mode.get_attribute('cfg').n_episode = policy_config.n_episode
            policy.eval_mode.get_attribute('cfg').n_episode = policy_config.n_episode

        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
        collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
        evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
        collector_env.seed(cfg.seed + task_id)
        evaluator_env.seed(cfg.seed + task_id, dynamic_seed=False)
        set_pkg_seed(cfg.seed + task_id, use_cuda=cfg.policy.cuda)

        # ===== NOTE: Create different game buffer, collector, evaluator for each task ====
        # TODO: share replay buffer for all tasks
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

    while True:
        # Precompute positional embedding matrices for collect/eval (not training)
        policy._collect_model.world_model.precompute_pos_emb_diff_kv()
        policy._target_model.world_model.precompute_pos_emb_diff_kv()

        # Collect data for each task
        for task_id, (cfg, collector, evaluator, replay_buffer) in enumerate(
                zip(cfgs, collectors, evaluators, game_buffers)):
            log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger)

            collect_kwargs = {
                'temperature': visit_count_temperature(
                    policy_config.manual_temperature_decay,
                    policy_config.fixed_temperature_value,
                    policy_config.threshold_training_steps_for_final_temperature,
                    trained_steps=learner.train_iter
                ),
                'epsilon': 0.0  # Default epsilon value
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
                print(f'evaluate task_id: {task_id}...')
                stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
                if stop:
                    break

            print('=' * 20)
            print(f'collect task_id: {task_id}...')

            # Reset initial data before each collection
            collector._policy.reset(reset_init_data=True)
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)

            # Determine updates per collection
            update_per_collect = cfg.policy.update_per_collect
            if update_per_collect is None:
                collected_transitions_num = sum(len(game_segment) for game_segment in new_data[0])
                update_per_collect = int(collected_transitions_num * cfg.policy.replay_ratio)

            # Update replay buffer
            replay_buffer.push_game_segments(new_data)
            replay_buffer.remove_oldest_data_to_fit()

        not_enough_data = any(replay_buffer.get_num_of_transitions() < batch_size for replay_buffer in game_buffers)

        # Learn policy from collected data.
        if not not_enough_data:
            # Learner will train ``update_per_collect`` times in one iteration.
            for i in range(update_per_collect):
                train_data_multi_task = []
                envstep_multi_task = 0
                for task_id, (cfg, collector, replay_buffer) in enumerate(zip(cfgs, collectors, game_buffers)):
                    envstep_multi_task += collector.envstep
                    if replay_buffer.get_num_of_transitions() > batch_size:
                        batch_size = cfg.policy.batch_size[task_id]
                        train_data = replay_buffer.sample(batch_size, policy)
                        if cfg.policy.reanalyze_ratio > 0 and i % 20 == 0:
                            policy.recompute_pos_emb_diff_and_clear_cache()
                        # Append task_id to train_data
                        train_data.append(task_id)
                        train_data_multi_task.append(train_data)
                    else:
                        logging.warning(
                            f'The data in replay_buffer is not sufficient to sample a mini-batch: '
                            f'batch_size: {batch_size}, replay_buffer: {replay_buffer}'
                        )
                        break

                if train_data_multi_task:
                    log_vars = learner.train(train_data_multi_task, envstep_multi_task)
                
                if cfg.policy.use_priority:
                    for task_id, replay_buffer in enumerate(game_buffers):
                        #  Update the priority for the task-specific replay buffer.
                        replay_buffer.update_priority(train_data_multi_task[task_id], log_vars[0][f'value_priority_task{task_id}'])
                        
                        # Retrieve the updated priorities for the current task.
                        current_priorities = log_vars[0][f'value_priority_task{task_id}']
                        
                        # Calculate statistics: mean, running mean, standard deviation for the priorities.
                        mean_priority = np.mean(current_priorities)
                        std_priority = np.std(current_priorities)
                        
                        # Using exponential moving average for running mean (alpha is the smoothing factor).
                        alpha = 0.1  # You can adjust this smoothing factor as needed.
                        if f'running_mean_priority_task{task_id}' not in value_priority_tasks:
                            # Initialize running mean if it does not exist.
                            value_priority_tasks[f'running_mean_priority_task{task_id}'] = mean_priority
                        else:
                            # Update running mean.
                            value_priority_tasks[f'running_mean_priority_task{task_id}'] = (
                                alpha * mean_priority + (1 - alpha) * value_priority_tasks[f'running_mean_priority_task{task_id}']
                            )
                        
                        # Calculate the normalized priority using the running mean.
                        running_mean_priority = value_priority_tasks[f'running_mean_priority_task{task_id}']
                        normalized_priorities = (current_priorities - running_mean_priority) / (std_priority + 1e-6)
                        
                        # Store the normalized priorities back to the replay buffer (if needed).
                        # replay_buffer.update_priority(train_data_multi_task[task_id], normalized_priorities)
                        
                        # Log the statistics if the print_task_priority_logs flag is set.
                        if cfg.policy.print_task_priority_logs:
                            print(f"Task {task_id} - Mean Priority: {mean_priority:.8f}, "
                                f"Running Mean Priority: {running_mean_priority:.8f}, "
                                f"Standard Deviation: {std_priority:.8f}")


        if all(collector.envstep >= max_env_step for collector in collectors) or learner.train_iter >= max_train_iter:
            break

    learner.call_hook('after_run')
    return policy