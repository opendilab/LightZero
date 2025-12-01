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

from lzero.entry.utils import log_buffer_memory_usage, convert_to_batch_for_gpt, create_unizero_loss_metrics, UniZeroDataLoader
from lzero.policy import visit_count_temperature
from lzero.policy.random_policy import LightZeroRandomPolicy
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.worker import MuZeroCollector as Collector
from .utils import random_collect, calculate_update_per_collect
import torch.distributed as dist
from ding.utils import set_pkg_seed, get_rank, get_world_size


def train_unizero_with_loss_landscape(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':
    """
    Overview:
        This function serves as the entry point for computing the loss landscape visualization of UniZero.
        It loads a pre-trained checkpoint specified by model_path, collects data using that checkpoint,
        and then visualizes the loss landscape. Unlike training, this function does NOT perform any policy
        updates on the checkpoint - it only loads the model, collects data, and visualizes the landscape.

    Arguments:
        - input_cfg (:obj:`Tuple[dict, dict]`): Configuration in dictionary format.
            ``Tuple[dict, dict]`` indicates [user_config, create_cfg].
        - seed (:obj:`int`): Random seed for reproducibility.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of a PyTorch model.
        - model_path (:obj:`Optional[str]`): Path to the checkpoint file to load for loss landscape visualization.
            This is the default checkpoint for loss landscape computation. An absolute path is recommended.
            The checkpoint will be loaded but NOT updated during this process.
        - max_train_iter (:obj:`Optional[int]`): Maximum number of policy update iterations during training.
        - max_env_step (:obj:`Optional[int]`): Maximum number of environment interaction steps to collect.

    Returns:
        - policy (:obj:`Policy`): The policy with the loaded checkpoint after loss landscape computation.
    """
    cfg, create_cfg = input_cfg

    # Ensure the specified policy type is supported for UniZero with loss landscape
    assert create_cfg.policy.type in ['unizero', 'sampled_unizero'], "train_unizero_with_loss_landscape only supports the following algorithms: 'unizero', 'sampled_unizero'"
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

    # Create environment managers for data collection
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    # Set random seeds for reproducibility
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

    # Create core components for data collection and loss landscape computation
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')) if get_rank() == 0 else None
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    replay_buffer = GameBuffer(cfg.policy)  # Buffer to store collected game segments
    collector = Collector(env=collector_env, policy=policy.collect_mode, tb_logger=tb_logger, exp_name=cfg.exp_name,
                          policy_config=cfg.policy)
    # Note: Evaluator is created but not used in loss landscape mode
    evaluator = Evaluator(eval_freq=cfg.policy.eval_freq, n_evaluator_episode=cfg.env.n_evaluator_episode,
                          stop_value=cfg.env.stop_value, env=evaluator_env, policy=policy.eval_mode,
                          tb_logger=tb_logger, exp_name=cfg.exp_name, policy_config=cfg.policy)

    # Execute the learner's initialization hook
    learner.call_hook('before_run')

    if cfg.policy.use_wandb:
        policy.set_train_iter_env_step(learner.train_iter, collector.envstep)

    # Perform initial random data collection to seed the replay buffer
    if cfg.policy.random_collect_episode_num > 0:
        logging.info("Performing random data collection to initialize replay buffer...")
        random_collect(cfg.policy, policy, LightZeroRandomPolicy, collector, collector_env, replay_buffer)
        logging.info("Random data collection completed!")

    batch_size = policy._cfg.batch_size

    # Setup multi-GPU configuration
    if cfg.policy.multi_gpu:
        world_size = get_world_size()
        rank = get_rank()
    else:
        world_size = 1
        rank = 0

    # Main training loop: Collect data once, then compute loss landscape
    while True:
        # Log memory usage of the replay buffer
        log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger)

        # Set temperature parameter for data collection during MCTS
        collect_kwargs = {
            'temperature': visit_count_temperature(
                cfg.policy.manual_temperature_decay,
                cfg.policy.fixed_temperature_value,
                cfg.policy.threshold_training_steps_for_final_temperature,
                trained_steps=learner.train_iter
            ),
            'epsilon': 0.0  # Default epsilon value
        }

        # Configure epsilon-greedy exploration if specified
        if cfg.policy.eps.eps_greedy_exploration_in_collect:
            epsilon_greedy_fn = get_epsilon_greedy_fn(
                start=cfg.policy.eps.start,
                end=cfg.policy.eps.end,
                decay=cfg.policy.eps.decay,
                type_=cfg.policy.eps.type
            )
            collect_kwargs['epsilon'] = epsilon_greedy_fn(collector.envstep)

        # Collect data for loss landscape computation (single iteration)
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        logging.info(f"Rank {rank}, Data collection iteration {learner.train_iter}: Completed!")

        # Note: In loss landscape mode, we don't perform policy updates
        # We only collect data to populate the replay buffer for landscape computation

        # Update replay buffer with newly collected data
        replay_buffer.push_game_segments(new_data)
        replay_buffer.remove_oldest_data_to_fit()

        # Check if sufficient data has been collected for loss landscape computation
        if collector.envstep > cfg.policy.train_start_after_envsteps:
            if cfg.policy.sample_type == 'episode':
                data_sufficient = replay_buffer.get_num_of_game_segments() > batch_size
            else:
                data_sufficient = replay_buffer.get_num_of_transitions() > batch_size

            if not data_sufficient:
                logging.warning(
                    f'Rank {rank}: Insufficient data in replay_buffer for landscape computation: '
                    f'batch_size: {batch_size}, buffer segments/transitions: {replay_buffer}. Continuing to collect data....'
                )
                # Continue collecting until enough data is available
                continue

        # Data collection complete - trigger loss landscape computation
        # ========== START: Loss Landscape Computation ==========
        logging.info("=" * 80)
        logging.info("Data collection complete! Starting Loss Landscape Computation")
        logging.info("=" * 80)

        # Import loss landscape module
        from lzero.loss_landscape import LossLandscape
        
        # ========== Loss Landscape Configuration ==========
        # Grid resolution for 2D loss landscape
        grid_size = 21  # 21x21 = 441 evaluation points
        # Number of batches for averaging loss estimates
        num_batches = 100  # Use multiple batches for stable loss estimation
        use_cuda = torch.cuda.is_available()

        # Create data loader from collected replay buffer
        dataloader = UniZeroDataLoader(replay_buffer, policy, batch_size, num_batches)

        # Create loss and metrics computation function for UniZero
        metrics_fn = create_unizero_loss_metrics(policy)

        # Setup output directory for loss landscape visualizations
        output_dir = os.path.join(cfg.exp_name, 'loss_landscape')
        os.makedirs(output_dir, exist_ok=True)

        # Generate HDF5 file path for storing computed loss landscape
        env_name = cfg.env.env_id.split('-')[0] if '-' in cfg.env.env_id else cfg.env.env_id
        landscape_file = os.path.join(output_dir, f'loss_landscape_{env_name}_{grid_size}x{grid_size}.h5')

        # Initialize LossLandscape with trained model and data
        logging.info("Initializing LossLandscape for UniZero model")
        landscape = LossLandscape(
            net=policy._model,
            dataloader=dataloader,
            criterion=metrics_fn,
            use_cuda=use_cuda,
            surf_file=landscape_file
        )

        # Compute 2D loss landscape over the model's weight space
        logging.info(f"Computing 2D loss landscape with {grid_size}x{grid_size} grid ({grid_size**2} evaluations)")
        result = landscape.compute_2d(
            xrange=(-1, 1, grid_size),           # X-axis: -1 to 1 normalized range
            yrange=(-1, 1, grid_size),           # Y-axis: -1 to 1 normalized range
            dir_type='weights',                  # Perturb weights, not states
            normalize='filter',                  # Normalize by filter (layer-wise)
            ignore='biasbn',                     # Ignore bias and batch norm parameters
            save=True                            # Save to HDF5 file
        )

        logging.info(f"Loss landscape computed. Data saved to: {landscape_file}")

        # Generate visualization plots from computed landscape
        try:
            landscape.plot_2d_contour(surf_name='auto', vmin=None, vmax=None)
            logging.info("2D contour plot (loss levels) generated successfully")
        except Exception as e:
            logging.warning(f"Failed to generate contour plot: {e}")

        try:
            landscape.plot_2d_surface(surf_name='auto')
            logging.info("3D surface plot generated successfully")
        except Exception as e:
            logging.warning(f"Failed to generate surface plot: {e}")

        logging.info("=" * 80)
        logging.info("Loss landscape visualization complete!")
        logging.info(f"Results saved to: {output_dir}")
        logging.info("=" * 80)
        # ========== END: Loss Landscape Computation ==========

        # Exit training loop after loss landscape computation
        logging.info("Exiting training loop - loss landscape computation finished")
        break

    # Cleanup and finalization
    learner.call_hook('after_run')
    if cfg.policy.use_wandb:
        wandb.finish()
    logging.info("===== Training with Loss Landscape Visualization Completed =====")
    return policy