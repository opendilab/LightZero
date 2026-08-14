import os
from functools import partial
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import wandb
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import get_rank, get_world_size, set_pkg_seed
from ding.worker import BaseLearner
from ditk import logging
from lzero.entry.utils import log_buffer_memory_usage
from lzero.policy import visit_count_temperature
from lzero.policy.random_policy import LightZeroRandomPolicy
from lzero.worker import MuZeroCollector as Collector
from lzero.worker import MuZeroEvaluator as Evaluator
from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from .utils import calculate_update_per_collect, random_collect


def train_unizero(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':
    """
    Overview:
        This function serves as the training entry point for UniZero, as proposed in our paper "UniZero: Generalized and Efficient Planning with Scalable Latent World Models".
        UniZero aims to enhance the planning capabilities of reinforcement learning agents by addressing the limitations found in MuZero-style algorithms,
        particularly in environments that require capturing long-term dependencies. More details can be found in https://arxiv.org/abs/2406.10667.
    
    Arguments:
        - input_cfg (:obj:`Tuple[dict, dict]`): Configuration in dictionary format.
            ``Tuple[dict, dict]`` indicates [user_config, create_cfg].
        - seed (:obj:`int`): Random seed for reproducibility.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of a PyTorch model.
        - model_path (:obj:`Optional[str]`): Path to the pretrained model, which should
            point to the checkpoint file of the pretrained model. An absolute path is recommended.
            In LightZero, the path typically resembles ``exp_name/ckpt/ckpt_best.pth.tar``.
        - max_train_iter (:obj:`Optional[int]`): Maximum number of policy update iterations during training.
        - max_env_step (:obj:`Optional[int]`): Maximum number of environment interaction steps to collect.
    
    Returns:
        - policy (:obj:`Policy`): The converged policy after training.
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
    # Policy parameters may remain unchanged during replay warm-up, so learner
    # train_iter is not a unique rollout identifier.  PPO freshness needs a
    # monotonically increasing collection version of its own.
    collection_version = 0

    if cfg.policy.use_wandb:
        policy.set_train_iter_env_step(learner.train_iter, collector.envstep)

    # Randomly collect data if specified
    if cfg.policy.policy_improvement == 'ppo' and cfg.policy.random_collect_episode_num > 0:
        raise ValueError(
            'UniZero+PPO does not accept random-policy warmup rollouts. Set '
            'random_collect_episode_num=0; world-model replay warmup can be added '
            'as an explicit pretraining phase without treating it as PPO data.'
        )
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
        if learner.train_iter == 0 or evaluator.should_eval(learner.train_iter):
            logging.info(f"Training iteration {learner.train_iter}: Starting evaluation...")
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            logging.info(f"Training iteration {learner.train_iter}: Evaluation completed, stop condition: {stop}, current reward: {reward}")
            if stop:
                logging.info("Stopping condition met, training ends!")
                break

        # Collect new data
        collection_train_iter = collection_version
        collection_version += 1
        new_data = collector.collect(
            train_iter=collection_train_iter,
            policy_kwargs=collect_kwargs,
            collect_with_pure_policy=True if cfg.policy.policy_improvement == 'ppo' else None,
        )
        logging.info(f"Rank {rank}, Training iteration {learner.train_iter}: New data collection completed!")

        # Determine updates per collection
        update_per_collect = cfg.policy.update_per_collect
        if update_per_collect is None:
            update_per_collect = calculate_update_per_collect(cfg, new_data, world_size)

        # Update replay buffer
        replay_buffer.push_game_segments(new_data)
        replay_buffer.remove_oldest_data_to_fit()
        on_policy_indices = (
            replay_buffer.get_on_policy_indices(collection_train_iter)
            if cfg.policy.policy_improvement == 'ppo'
            else None
        )

        if world_size > 1:
            # Synchronize all ranks before training
            try:
                dist.barrier()
            except Exception as e:
                logging.error(f'Rank {rank}: Synchronization barrier failed, error: {e}')
                break

        # Check if there is sufficient data for training
        if collector.envstep > cfg.policy.train_start_after_envsteps:
            if cfg.policy.policy_improvement == 'ppo':
                data_sufficient = len(on_policy_indices) > 0
            elif cfg.policy.sample_type == 'episode':
                data_sufficient = replay_buffer.get_num_of_game_segments() > batch_size
            else:
                data_sufficient = replay_buffer.get_num_of_transitions() > batch_size
            
            if not data_sufficient:
                logging.warning(
                    f'Rank {rank}: The data in replay_buffer is not sufficient to sample a mini-batch: '
                    f'batch_size: {batch_size}, replay_buffer: {replay_buffer}. Continue to collect now ....'
                )
                if cfg.policy.policy_improvement == 'ppo':
                    # The transition/reward data remains available for replay;
                    # discard only bulky rollout tensors that will never be
                    # consumed after this policy version is skipped.
                    replay_buffer.release_on_policy_data(collection_train_iter)
                continue

            if cfg.policy.policy_improvement == 'ppo':
                ppo_cfg = cfg.policy.ppo
                minibatch_size = min(int(ppo_cfg.minibatch_size), len(on_policy_indices))
                target_kl = float(ppo_cfg.target_kl)
                stop_ppo = False
                for epoch in range(int(ppo_cfg.epochs)):
                    shuffled_indices = np.random.permutation(on_policy_indices)
                    for start in range(0, len(shuffled_indices), minibatch_size):
                        minibatch_indices = shuffled_indices[start:start + minibatch_size]
                        train_data = replay_buffer.sample_on_policy(
                            minibatch_indices, policy, collection_train_iter
                        )
                        if cfg.policy.use_wandb:
                            policy.set_train_iter_env_step(learner.train_iter, collector.envstep)
                        train_data.extend([
                            learner.train_iter,
                            {
                                'type': 'ppo',
                                'collection_train_iter': collection_train_iter,
                                'epoch': epoch,
                            },
                        ])
                        log_vars = learner.train(train_data, collector.envstep)
                        approx_kl = float(log_vars[0].get('ppo/approx_kl', 0.0))
                        if epoch == 0 and start == 0:
                            initial_ratio_mean = float(log_vars[0].get('ppo/ratio_mean', 1.0))
                            initial_ratio_min = float(log_vars[0].get('ppo/ratio_min', 1.0))
                            initial_ratio_max = float(log_vars[0].get('ppo/ratio_max', 1.0))
                            tolerance = float(ppo_cfg.fresh_ratio_tolerance)
                            max_ratio_error = max(
                                abs(initial_ratio_min - 1.0), abs(initial_ratio_max - 1.0)
                            )
                            if max_ratio_error > tolerance:
                                raise RuntimeError(
                                    'Fresh PPO rollout failed the ratio=1 invariant: '
                                    f'mean/min/max={initial_ratio_mean:.8f}/'
                                    f'{initial_ratio_min:.8f}/{initial_ratio_max:.8f}, '
                                    f'tolerance={tolerance:.2e}'
                                )
                        if target_kl > 0 and approx_kl > target_kl:
                            logging.info(
                                'Stopping PPO epochs early at epoch %d: approx_kl %.6f > target_kl %.6f',
                                epoch, approx_kl, target_kl,
                            )
                            stop_ppo = True
                            break
                    if stop_ppo:
                        break

                replay_buffer.release_on_policy_data(collection_train_iter)

                # Replay-based latent model updates happen only after PPO has consumed
                # the behavior-policy rollout, so the first PPO ratio starts at one.
                wm_updates = ppo_cfg.world_model_update_per_collect
                wm_updates = update_per_collect if wm_updates is None else int(wm_updates)
                for _ in range(wm_updates):
                    wm_batch_size = min(batch_size, replay_buffer.get_num_of_transitions())
                    train_data = replay_buffer.sample_world_model(wm_batch_size)
                    train_data.extend([learner.train_iter, {'type': 'world_model'}])
                    learner.train(train_data, collector.envstep)
            else:
                # Original UniZero MCTS policy-improvement path.
                for i in range(update_per_collect):
                    train_data = replay_buffer.sample(batch_size, policy)
                    if replay_buffer._cfg.reanalyze_ratio > 0 and i % 20 == 0:
                        policy.recompute_pos_emb_diff_and_clear_cache()

                    if cfg.policy.use_wandb:
                        policy.set_train_iter_env_step(learner.train_iter, collector.envstep)

                    train_data.append(learner.train_iter)

                    if os.environ.get('DEBUG', '').lower() == 'true':
                        import pudb; pudb.set_trace()

                    log_vars = learner.train(train_data, collector.envstep)

                    if cfg.policy.use_priority:
                        replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])

        elif cfg.policy.policy_improvement == 'ppo':
            # Rollouts collected during replay warm-up are intentionally not
            # used by a later actor version.
            replay_buffer.release_on_policy_data(collection_train_iter)

        policy.recompute_pos_emb_diff_and_clear_cache()

        # Check stopping criteria
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            logging.info("Stopping condition met, training ends!")
            break

    learner.call_hook('after_run')
    if cfg.policy.use_wandb:
        wandb.finish()
    logging.info("===== Training Completed =====")
    return policy
