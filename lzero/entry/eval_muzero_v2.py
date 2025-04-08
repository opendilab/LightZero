import os
from functools import partial
from typing import Optional, Tuple
import logging

import numpy as np
import torch
from tensorboardX import SummaryWriter

from ding.config import compile_config
from ding.envs import create_env_manager
from ding.envs import get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.worker import BaseLearner
from lzero.worker import MuZeroEvaluator
from lzero.entry.utils import initialize_zeros_batch
import logging
import os
from functools import partial
from typing import Tuple, Optional

import torch
import wandb
from ding.config import compile_config
from ding.envs import create_env_manager
from ding.envs import get_vec_env_setting
from ding.policy import create_policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import set_pkg_seed, get_rank
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from lzero.entry.utils import log_buffer_memory_usage
from lzero.policy import visit_count_temperature
from lzero.policy.random_policy import LightZeroRandomPolicy
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.worker import MuZeroCollector as Collector
from .utils import random_collect, calculate_update_per_collect
import torch.distributed as dist
from ding.utils import set_pkg_seed, get_rank, get_world_size

def eval_muzero_v2(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        num_episodes_each_seed: int = 1,
        print_seed_details: int = False,
) -> 'Policy':  # noqa
    """
    Overview:
        The eval entry for MCTS+RL algorithms, including MuZero, EfficientZero, Sampled EfficientZero, StochasticMuZero, GumbelMuZero, UniZero, etc.
    Arguments:
        - input_cfg (:obj:`Tuple[dict, dict]`): Config in dict type.
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - model_path (:obj:`Optional[str]`): The pretrained model path, which should
            point to the ckpt file of the pretrained model, and an absolute path is recommended.
            In LightZero, the path is usually something like ``exp_name/ckpt/ckpt_best.pth.tar``.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    cfg, create_cfg = input_cfg

    # Ensure the specified policy type is supported
    assert create_cfg.policy.type in ['unizero', 'sampled_unizero'], "train_unizero only supports the following algorithms: 'unizero', 'sampled_unizero'"
    logging.info(f"Using policy type: {create_cfg.policy.type}")

    # Import the appropriate GameBuffer class based on the policy type
    game_buffer_classes = {'unizero': 'UniZeroGameBuffer', 'sampled_unizero': 'SampledUniZeroGameBuffer'}
    GameBuffer = getattr(__import__('lzero.mcts', fromlist=[game_buffer_classes[create_cfg.policy.type]]),
                         game_buffer_classes[create_cfg.policy.type])

    # Check for GPU availability and set the device accordingly
    cfg.policy.device = cfg.policy.model.world_model_cfg.device if torch.cuda.is_available() else 'cpu'
    logging.info(f"Device set to: {cfg.policy.device}")

    # Compile the configuration file
    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)

    # Create environment manager
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    # Initialize environment and random seed
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=torch.cuda.is_available())

    # Initialize wandb if specified
    if cfg.policy.use_wandb:
        logging.info("Initializing wandb...")
        wandb.init(
            project="LightZero",
            config=cfg,
            sync_tensorboard=False,
            monitor_gym=False,
            save_code=True,
        )
        logging.info("wandb initialization completed!")

    # Create policy
    logging.info("Creating policy...")
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])
    logging.info("Policy created successfully!")

    # Load pretrained model if specified
    if model_path is not None:
        logging.info(f"Loading pretrained model from {model_path}...")
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))
        logging.info("Pretrained model loaded successfully!")

    # Create core components for training
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')) if get_rank() == 0 else None
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    replay_buffer = GameBuffer(cfg.policy)
    collector = Collector(env=collector_env, policy=policy.collect_mode, tb_logger=tb_logger, exp_name=cfg.exp_name,
                          policy_config=cfg.policy)
    evaluator = Evaluator(eval_freq=cfg.policy.eval_freq, n_evaluator_episode=cfg.env.n_evaluator_episode,
                          stop_value=cfg.env.stop_value, env=evaluator_env, policy=policy.eval_mode,
                          tb_logger=tb_logger, exp_name=cfg.exp_name, policy_config=cfg.policy)

    # Execute the learner's before_run hook
    learner.call_hook('before_run')

    if cfg.policy.use_wandb:
        policy.set_train_iter_env_step(learner.train_iter, collector.envstep)

    # Randomly collect data if specified
    if cfg.policy.random_collect_episode_num > 0:
        logging.info("Collecting random data...")
        random_collect(cfg.policy, policy, LightZeroRandomPolicy, collector, collector_env, replay_buffer)
        logging.info("Random data collection completed!")

    batch_size = policy._cfg.batch_size

    if cfg.policy.multi_gpu:
        # Get current world size and rank
        world_size = get_world_size()
        rank = get_rank()
    else:
        world_size = 1
        rank = 0

    while True:
        # Log memory usage of the replay buffer
        log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger)

        # Set temperature parameter for data collection
        collect_kwargs = {
            'temperature': visit_count_temperature(
                cfg.policy.manual_temperature_decay,
                cfg.policy.fixed_temperature_value,
                cfg.policy.threshold_training_steps_for_final_temperature,
                trained_steps=learner.train_iter
            ),
            'epsilon': 0.0  # Default epsilon value
        }

        # Configure epsilon-greedy exploration
        if cfg.policy.eps.eps_greedy_exploration_in_collect:
            epsilon_greedy_fn = get_epsilon_greedy_fn(
                start=cfg.policy.eps.start,
                end=cfg.policy.eps.end,
                decay=cfg.policy.eps.decay,
                type_=cfg.policy.eps.type
            )
            collect_kwargs['epsilon'] = epsilon_greedy_fn(collector.envstep)

        # Evaluate policy performance
        # logging.info(f"Training iteration {learner.train_iter}: Starting evaluation...")
        # stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
        # logging.info(f"Training iteration {learner.train_iter}: Evaluation completed, stop condition: {stop}, current reward: {reward}")
        # if stop:
        #     logging.info("Stopping condition met, training ends!")
        #     break

        # Collect new data
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        logging.info(f"Rank {rank}, Training iteration {learner.train_iter}: New data collection completed!")

        if world_size > 1:
            # Synchronize all ranks before training
            try:
                dist.barrier()
            except Exception as e:
                logging.error(f'Rank {rank}: Synchronization barrier failed, error: {e}')
                break

        
        policy.recompute_pos_emb_diff_and_clear_cache()



    learner.call_hook('after_run')
    if cfg.policy.use_wandb:
        wandb.finish()
    logging.info("===== Training Completed =====")
    return policy
