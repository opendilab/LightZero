import logging
import os
from functools import partial
from typing import Optional, Tuple

import torch
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import set_pkg_seed, get_rank
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter

from lzero.entry.utils import log_buffer_memory_usage, log_buffer_run_time
from lzero.policy import visit_count_temperature
from lzero.policy.random_policy import LightZeroRandomPolicy
from lzero.worker import MuZeroCollector as Collector
from lzero.worker import MuZeroEvaluator as Evaluator
from .utils import random_collect


def train_rezero(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':
    """
    Train entry for ReZero algorithms (ReZero-MuZero, ReZero-EfficientZero). More details can be found in the ReZero paper: https://arxiv.org/pdf/2404.16364.

    Args:
        - input_cfg (:obj:`Tuple[dict, dict]`): Configuration dictionaries (user_config, create_cfg).
        - seed (:obj:`int`): Random seed for reproducibility.
        - model (:obj:`Optional[torch.nn.Module]`): Pre-initialized model instance.
        - model_path (:obj:`Optional[str]`): Path to pretrained model checkpoint.
        - max_train_iter (:obj:`Optional[int]`): Maximum number of training iterations.
        - max_env_step (:obj:`Optional[int]`): Maximum number of environment steps.

    Returns:
        - Policy: Trained policy object.
    """
    cfg, create_cfg = input_cfg
    assert create_cfg.policy.type in ['efficientzero', 'muzero'], \
        "train_rezero entry only supports 'efficientzero' and 'muzero' algorithms"

    # Import appropriate GameBuffer based on policy type
    if create_cfg.policy.type == 'muzero':
        from lzero.mcts import ReZeroMZGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'efficientzero':
        from lzero.mcts import ReZeroEZGameBuffer as GameBuffer

    # Set device (CUDA if available and enabled, otherwise CPU)
    cfg.policy.device = 'cuda' if cfg.policy.cuda and torch.cuda.is_available() else 'cpu'

    # Compile and finalize configuration
    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)

    # Create environment, policy, and core components
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    # Set seeds for reproducibility
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    # Adjust checkpoint saving frequency for offline evaluation
    if cfg.policy.eval_offline:
        cfg.policy.learn.learner.hook.save_ckpt_after_iter = cfg.policy.eval_freq

    # Create and initialize policy
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])
    if model_path:
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))

    # Initialize worker components
    tb_logger = SummaryWriter(os.path.join(f'./{cfg.exp_name}/log/', 'serial')) if get_rank() == 0 else None
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    replay_buffer = GameBuffer(cfg.policy)
    collector = Collector(
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=cfg.policy
    )
    evaluator = Evaluator(
        eval_freq=cfg.policy.eval_freq,
        n_evaluator_episode=cfg.env.n_evaluator_episode,
        stop_value=cfg.env.stop_value,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=cfg.policy
    )

    # Main training loop
    learner.call_hook('before_run')
    update_per_collect = cfg.policy.update_per_collect

    # Perform initial random data collection if specified
    if cfg.policy.random_collect_episode_num > 0:
        random_collect(cfg.policy, policy, LightZeroRandomPolicy, collector, collector_env, replay_buffer)

    # Initialize offline evaluation tracking if enabled
    if cfg.policy.eval_offline:
        eval_train_iter_list, eval_train_envstep_list = [], []

    # Evaluate initial random agent
    stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)

    buffer_reanalyze_count = 0
    train_epoch = 0
    while True:
        # Log buffer metrics
        log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger)
        log_buffer_run_time(learner.train_iter, replay_buffer, tb_logger)

        # Prepare collection parameters
        collect_kwargs = {
            'temperature': visit_count_temperature(
                cfg.policy.manual_temperature_decay,
                cfg.policy.fixed_temperature_value,
                cfg.policy.threshold_training_steps_for_final_temperature,
                trained_steps=learner.train_iter
            ),
            'epsilon': get_epsilon_greedy_fn(
                cfg.policy.eps.start, cfg.policy.eps.end,
                cfg.policy.eps.decay, cfg.policy.eps.type
            )(collector.envstep) if cfg.policy.eps.eps_greedy_exploration_in_collect else 0.0
        }

        # Periodic evaluation
        if evaluator.should_eval(learner.train_iter):
            if cfg.policy.eval_offline:
                eval_train_iter_list.append(learner.train_iter)
                eval_train_envstep_list.append(collector.envstep)
            else:
                stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
                if stop:
                    break

        # Collect new data
        new_data = collector.collect(
            train_iter=learner.train_iter,
            policy_kwargs=collect_kwargs,
            collect_with_pure_policy=cfg.policy.collect_with_pure_policy
        )

        # Update collection frequency if not specified
        if update_per_collect is None:
            collected_transitions = sum(len(segment) for segment in new_data[0])
            update_per_collect = int(collected_transitions * cfg.policy.replay_ratio)

        # Update replay buffer
        replay_buffer.push_game_segments(new_data)
        replay_buffer.remove_oldest_data_to_fit()

        # Periodically reanalyze buffer
        if cfg.policy.buffer_reanalyze_freq >= 1:
            # Reanalyze buffer <buffer_reanalyze_freq> times in one train_epoch
            reanalyze_interval = update_per_collect // cfg.policy.buffer_reanalyze_freq
        else:
            # Reanalyze buffer each <1/buffer_reanalyze_freq> train_epoch
            if train_epoch % (1//cfg.policy.buffer_reanalyze_freq) == 0 and replay_buffer.get_num_of_transitions() > 2000:
                # When reanalyzing the buffer, the samples in the entire buffer are processed in mini-batches with a batch size of 2000.
                # This is an empirically selected value for optimal efficiency.
                replay_buffer.reanalyze_buffer(2000, policy)
                buffer_reanalyze_count += 1
                logging.info(f'Buffer reanalyze count: {buffer_reanalyze_count}')

        # Training loop
        for i in range(update_per_collect):
            if cfg.policy.buffer_reanalyze_freq >= 1:
                # Reanalyze buffer <buffer_reanalyze_freq> times in one train_epoch
                if i % reanalyze_interval == 0 and replay_buffer.get_num_of_transitions() > 2000:
                    # When reanalyzing the buffer, the samples in the entire buffer are processed in mini-batches with a batch size of 2000.
                    # This is an empirically selected value for optimal efficiency.
                    replay_buffer.reanalyze_buffer(2000, policy)
                    buffer_reanalyze_count += 1
                    logging.info(f'Buffer reanalyze count: {buffer_reanalyze_count}')

            # Sample and train on mini-batch
            if replay_buffer.get_num_of_transitions() > cfg.policy.batch_size:
                train_data = replay_buffer.sample(cfg.policy.batch_size, policy)
                log_vars = learner.train(train_data, collector.envstep)

                if cfg.policy.use_priority:
                    replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])
            else:
                logging.warning('Insufficient data in replay buffer for sampling. Continuing collection...')
                break

        train_epoch += 1
        # Check termination conditions
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            if cfg.policy.eval_offline:
                perform_offline_evaluation(cfg, learner, policy, evaluator, eval_train_iter_list,
                                           eval_train_envstep_list)
            break

    learner.call_hook('after_run')
    return policy


def perform_offline_evaluation(cfg, learner, policy, evaluator, eval_train_iter_list, eval_train_envstep_list):
    """
    Perform offline evaluation of the trained model.

    Args:
        cfg (dict): Configuration dictionary.
        learner (BaseLearner): Learner object.
        policy (Policy): Policy object.
        evaluator (Evaluator): Evaluator object.
        eval_train_iter_list (list): List of training iterations for evaluation.
        eval_train_envstep_list (list): List of environment steps for evaluation.
    """
    logging.info('Starting offline evaluation...')
    ckpt_dirname = f'./{learner.exp_name}/ckpt'

    for train_iter, collector_envstep in zip(eval_train_iter_list, eval_train_envstep_list):
        ckpt_path = os.path.join(ckpt_dirname, f'iteration_{train_iter}.pth.tar')
        policy.learn_mode.load_state_dict(torch.load(ckpt_path, map_location=cfg.policy.device))
        stop, reward = evaluator.eval(learner.save_checkpoint, train_iter, collector_envstep)
        logging.info(f'Offline eval at iter: {train_iter}, steps: {collector_envstep}, reward: {reward}')

    logging.info('Offline evaluation completed')