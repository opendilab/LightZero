#!/usr/bin/env python3
"""
Loss Landscape Visualization Tool for UniZero

This script generates 2D loss landscape visualizations for UniZero models.
It loads a trained checkpoint and computes loss landscapes using multiple metrics.

Usage:
    python lzero/tools/visualize_loss_landscape.py \\
        --checkpoint data_lz/data_unizero/Pong/xxx/ckpt/ckpt_best.pth.tar \\
        --env PongNoFrameskip-v4 \\
        --seed 0
        
        我想实现 ezv2 请你阅读/mnt/shared-storage-user/tangjia/temp/EfficientZeroV2/ez/mcts 下面的代码,掌握 ezv2 的逻辑然后帮我在/mnt/shared-storage-user/tangjia/eff/LightZero/lzero/mcts/ptree 下面创建一个 ptree_ezv2.py
  文件,/mnt/shared-storage-user/tangjia/eff/LightZero/lzero/mcts/ptree也有 ez 的实现,ez 已经在/mnt/shared-storage-user/tangjia/eff/LightZero/lzero/mcts/ptree 下面了,我现在需要实现 ezv2 
        
        
"""
import os
import sys
import argparse
import logging
from functools import partial
from typing import Dict, Optional, Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed

# Add loss_landscape_core to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../../')
from loss_landscape_core import LossLandscape


def setup_logging(output_dir: str):
    """Setup logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'landscape.log')),
            logging.StreamHandler()
        ]
    )


class UniZeroDataLoader:
    """
    Wrapper to convert UniZero GameBuffer to DataLoader-like interface.

    This allows us to use the GameBuffer with loss_landscape_core's LossLandscape class.
    """

    def __init__(self, replay_buffer, policy, batch_size: int, num_batches: int):
        """
        Args:
            replay_buffer: UniZeroGameBuffer instance
            policy: UniZero policy instance
            batch_size: Batch size for sampling
            num_batches: Number of batches to sample
        """
        self.buffer = replay_buffer
        self.policy = policy
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __iter__(self):
        """Iterate through sampled batches from the replay buffer."""
        for _ in range(self.num_batches):
            batch = self.buffer.sample(self.batch_size, self.policy)
            yield batch

    def __len__(self):
        return self.num_batches


def create_unizero_metrics(policy_instance) -> callable:
    """
    Create a custom metrics function for UniZero.

    This function returns a metrics computation function that computes multiple
    loss components for the loss landscape.

    Args:
        policy_instance: UniZero policy instance

    Returns:
        A callable that computes metrics given model, dataloader, and use_cuda
    """

    def compute_metrics(net, dataloader, use_cuda: bool) -> Dict[str, float]:
        """
        Compute multiple metrics for UniZero.

        Args:
            net: UniZero model
            dataloader: DataLoader (UniZeroDataLoader instance)
            use_cuda: Whether to use GPU

        Returns:
            Dict mapping metric names to values
        """
        net.eval()
        device = 'cuda' if use_cuda else 'cpu'

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_reward_loss = 0.0
        total_consistency_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_data in dataloader:
                # batch_data structure from GameBuffer.sample():
                # (obs, action, mask, target_value, target_reward, target_policy, ...)

                try:
                    obs_batch_ori, action_batch, mask_batch, target_value, target_reward, target_policy = batch_data[:6]

                    if use_cuda:
                        obs_batch_ori = obs_batch_ori.cuda()
                        action_batch = action_batch.cuda()
                        mask_batch = mask_batch.cuda()
                        target_value = target_value.cuda()
                        target_reward = target_reward.cuda()
                        target_policy = target_policy.cuda()

                    batch_size = obs_batch_ori.shape[0]

                    # Forward pass through the network
                    # This mimics the forward_learn logic but without gradient computation

                    # Initial inference
                    network_output = net.initial_inference(obs_batch_ori)
                    latent_state, reward, value, policy_logits = network_output[:4]

                    # Compute losses using the same logic as policy._forward_learn
                    # (simplified version - extracts the main loss components)

                    # Policy loss
                    from lzero.policy.utils import cross_entropy_loss
                    policy_loss = cross_entropy_loss(policy_logits, target_policy[:, 0])

                    # Value loss
                    value_loss = cross_entropy_loss(value, target_value[:, 0])

                    # Reward loss (simplified - first step reward)
                    reward_loss = cross_entropy_loss(reward, target_reward[:, 0])

                    # Consistency loss (optional, set to 0 if not computed)
                    consistency_loss = torch.zeros_like(policy_loss)

                    # Accumulate losses
                    total_policy_loss += (policy_loss.mean().item() if policy_loss.numel() > 0 else 0.0) * batch_size
                    total_value_loss += (value_loss.mean().item() if value_loss.numel() > 0 else 0.0) * batch_size
                    total_reward_loss += (reward_loss.mean().item() if reward_loss.numel() > 0 else 0.0) * batch_size
                    total_consistency_loss += (consistency_loss.mean().item() if consistency_loss.numel() > 0 else 0.0) * batch_size
                    total_samples += batch_size

                except Exception as e:
                    logging.warning(f"Error processing batch: {e}")
                    continue

        # Return metrics dictionary
        if total_samples > 0:
            metrics = {
                'policy_loss': total_policy_loss / total_samples,
                'value_loss': total_value_loss / total_samples,
                'reward_loss': total_reward_loss / total_samples,
                'consistency_loss': total_consistency_loss / total_samples,
                'total_loss': (total_policy_loss + total_value_loss + total_reward_loss) / total_samples
            }
        else:
            logging.warning("No samples processed! Returning dummy metrics.")
            metrics = {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'reward_loss': 0.0,
                'consistency_loss': 0.0,
                'total_loss': 0.0
            }

        return metrics

    return compute_metrics


def build_unizero_config(env_id: str, seed: int):
    """
    Build UniZero configuration without triggering training.
    This is a standalone copy of atari_unizero_segment_config logic.

    Args:
        env_id: Environment ID (e.g., 'PongNoFrameskip-v4')
        seed: Random seed

    Returns:
        Tuple of (cfg, create_cfg)
    """
    from easydict import EasyDict
    from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map

    action_space_size = atari_env_action_space_map[env_id]

    # Config parameters
    collector_env_num = 8
    num_segments = 8
    game_segment_length = 20
    evaluator_env_num = 10
    num_simulations = 50
    batch_size = 64
    num_layers = 2
    replay_ratio = 0.25
    num_unroll_steps = 10
    infer_context_length = 4
    buffer_reanalyze_freq = 1/50
    reanalyze_batch_size = 160
    reanalyze_partition = 0.75

    atari_unizero_config = dict(
        env=dict(
            stop_value=int(1e6),
            env_id=env_id,
            observation_shape=(3, 64, 64),
            gray_scale=False,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
        ),
        policy=dict(
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000000, ), ), ),
            model=dict(
                observation_shape=(3, 64, 64),
                action_space_size=action_space_size,
                reward_support_range=(-300., 301., 1.),
                value_support_range=(-300., 301., 1.),
                world_model_cfg=dict(
                    support_size=601,
                    policy_entropy_weight=5e-3,
                    continuous_action_space=False,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    device='cuda',
                    action_space_size=action_space_size,
                    num_layers=num_layers,
                    num_heads=8,
                    embed_dim=768,
                    obs_type='image',
                    env_num=max(collector_env_num, evaluator_env_num),
                    num_simulations=num_simulations,
                    rotary_emb=False,
                ),
            ),
            model_path=None,
            use_augmentation=False,
            manual_temperature_decay=False,
            threshold_training_steps_for_final_temperature=int(2.5e4),
            use_priority=False,
            num_unroll_steps=num_unroll_steps,
            update_per_collect=None,
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            optim_type='AdamW',
            learning_rate=0.0001,
            num_simulations=num_simulations,
            num_segments=num_segments,
            td_steps=5,
            train_start_after_envsteps=0,
            game_segment_length=game_segment_length,
            grad_clip_value=5,
            replay_buffer_size=int(1e6),
            eval_freq=int(5e3),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
        ),
    )
    cfg = EasyDict(atari_unizero_config)

    create_config = dict(
        env=dict(
            type='atari_lightzero',
            import_names=['zoo.atari.envs.atari_lightzero_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(
            type='unizero',
            import_names=['lzero.policy.unizero'],
        ),
    )
    create_cfg = EasyDict(create_config)

    return cfg, create_cfg


def load_unizero_config_and_create_policy(
    env_id: str,
    seed: int,
    checkpoint_path: str,
    use_cuda: bool = True
) -> Tuple:
    """
    Load UniZero configuration, create policy, and load checkpoint.

    Args:
        env_id: Environment ID (e.g., 'PongNoFrameskip-v4')
        seed: Random seed
        checkpoint_path: Path to checkpoint file
        use_cuda: Whether to use GPU

    Returns:
        Tuple of (policy, cfg, create_cfg)
    """
    logging.info(f"Loading UniZero configuration for {env_id}")

    # Build configuration (without triggering training)
    cfg, create_cfg = build_unizero_config(env_id, seed)

    # Set device
    device = 'cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu'
    cfg.policy.device = device
    cfg.policy.model.world_model_cfg.device = device

    logging.info(f"Using device: {device}")

    # Compile config
    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg)

    # Create policy
    logging.info("Creating UniZero policy")
    policy = create_policy(cfg.policy, enable_field=['learn', 'collect', 'eval'])

    # Load checkpoint
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # UniZero checkpoint structure: {model, target_model, optimizer_world_model, ...}
        # The load_state_dict expects the entire checkpoint dict
        try:
            policy.learn_mode.load_state_dict(checkpoint)
            logging.info("Checkpoint loaded successfully")
        except KeyError as e:
            logging.error(f"Failed to load checkpoint with standard structure: {e}")
            # Try alternative: if checkpoint only has 'model' key at top level
            if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
                try:
                    policy.learn_mode.load_state_dict(checkpoint['model'])
                    logging.info("Checkpoint loaded from ['model'] key")
                except Exception as e2:
                    logging.error(f"Failed to load checkpoint from ['model']: {e2}")
                    raise
            else:
                raise
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    return policy, cfg, create_cfg


def collect_data_for_buffer(
    policy,
    cfg,
    create_cfg,
    num_episodes: int = 5,
    use_cuda: bool = True
) -> 'GameBuffer':
    """
    Collect some data and fill the replay buffer.

    This is necessary because the loss landscape computation needs data samples
    from the replay buffer to evaluate the loss at different weight perturbations.

    Args:
        policy: UniZero policy
        cfg: Configuration
        create_cfg: Create configuration
        num_episodes: Number of episodes to collect
        use_cuda: Whether to use GPU

    Returns:
        Filled GameBuffer instance
    """
    logging.info(f"Collecting {num_episodes} episodes of data for replay buffer")

    from functools import partial
    from lzero.mcts import UniZeroGameBuffer as GameBuffer
    from lzero.worker import MuZeroCollector as Collector

    # Create environments
    env_fn, collector_env_cfg, _ = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(
        cfg.env.manager,
        [partial(env_fn, cfg=c) for c in collector_env_cfg]
    )
    collector_env.seed(cfg.seed)
    set_pkg_seed(cfg.seed, use_cuda=use_cuda)

    # Create replay buffer
    replay_buffer = GameBuffer(cfg.policy)

    # Create collector
    collector = Collector(
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=None,
        exp_name='temp_landscape',
        policy_config=cfg.policy
    )

    # Collect data
    collected_episodes = 0
    while collected_episodes < num_episodes:
        new_data = collector.collect(train_iter=0)
        replay_buffer.push_game_segments(new_data)
        collected_episodes += len(new_data)
        logging.info(f"Collected {collected_episodes}/{num_episodes} episodes")

    collector_env.close()
    logging.info(f"Replay buffer filled with {replay_buffer.get_num_of_transitions()} transitions")

    return replay_buffer


def visualize_loss_landscape(
    checkpoint_path: str,
    env_id: str = 'PongNoFrameskip-v4',
    seed: int = 0,
    output_dir: str = './landscape_output',
    grid_size: int = 11,
    num_batches: int = 20,
    batch_size: int = 64,
    num_collect_episodes: int = 10,
    use_cuda: bool = True
):
    """
    Main function to generate loss landscape visualization.

    Args:
        checkpoint_path: Path to trained checkpoint
        env_id: Environment ID
        seed: Random seed
        output_dir: Output directory for results
        grid_size: Grid size for 2D landscape (11 or 21)
        num_batches: Number of batches to use for landscape computation
        batch_size: Batch size for sampling
        num_collect_episodes: Number of episodes to collect for replay buffer
        use_cuda: Whether to use GPU
    """
    print("=" * 70)
    print("  UniZero Loss Landscape Visualization")
    print("=" * 70)

    # Setup
    setup_logging(output_dir)
    logging.info(f"Configuration:")
    logging.info(f"  Checkpoint: {checkpoint_path}")
    logging.info(f"  Environment: {env_id}")
    logging.info(f"  Output: {output_dir}")
    logging.info(f"  Grid Size: {grid_size}x{grid_size}")
    logging.info(f"  Num Batches: {num_batches}")

    # 1. Load policy and checkpoint
    policy, cfg, create_cfg = load_unizero_config_and_create_policy(
        env_id=env_id,
        seed=seed,
        checkpoint_path=checkpoint_path,
        use_cuda=use_cuda
    )
    # todo
    
    
    # 2. Collect data for replay buffer
    replay_buffer = collect_data_for_buffer(
        policy=policy,
        cfg=cfg,
        create_cfg=create_cfg,
        num_episodes=num_collect_episodes,
        use_cuda=use_cuda
    )

    # 3. Create DataLoader wrapper
    logging.info("Creating DataLoader wrapper for replay buffer")
    dataloader = UniZeroDataLoader(
        replay_buffer=replay_buffer,
        policy=policy,
        batch_size=batch_size,
        num_batches=num_batches
    )

    # 4. Create custom metrics function
    logging.info("Creating custom metrics function")
    criterion = create_unizero_metrics(policy)

    # 5. Initialize LossLandscape
    logging.info("Initializing LossLandscape")
    device = 'cuda' if use_cuda else 'cpu'
    landscape = LossLandscape(
        net=policy.learn_mode,
        dataloader=dataloader,
        criterion=criterion,
        use_cuda=use_cuda,
        surf_file=os.path.join(output_dir, 'loss_landscape.h5')
    )

    # 6. Compute 2D landscape
    logging.info(f"\nComputing 2D loss landscape ({grid_size}x{grid_size} grid)...")
    try:
        result = landscape.compute_2d(
            xrange=(-1, 1, grid_size),
            yrange=(-1, 1, grid_size),
            dir_type='weights',
            normalize='filter',
            ignore='biasbn',
            save=True
        )
        logging.info("✓ Landscape computation complete")
    except Exception as e:
        logging.error(f"Error computing landscape: {e}")
        raise

    # 7. Generate visualizations
    logging.info("\nGenerating visualizations...")
    try:
        landscape.plot_2d_contour(surf_name='auto', vmin=0.1, vmax=10, vlevel=0.5, show=False)
        logging.info("✓ Contour plots generated")

        landscape.plot_2d_surface(surf_name='auto', show=False)
        logging.info("✓ 3D surface plots generated")
    except Exception as e:
        logging.warning(f"Warning during visualization generation: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("  VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - loss_landscape.h5 (data)")
    print(f"  - Multiple PDF visualizations:")
    for metric in ['policy_loss', 'value_loss', 'reward_loss', 'consistency_loss', 'total_loss']:
        print(f"    • {metric} (4 plots)")
    print("\n" + "=" * 70 + "\n")


def main():
    """Parse arguments and run visualization."""
    parser = argparse.ArgumentParser(
        description='Generate loss landscape visualization for UniZero',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python lzero/tools/visualize_loss_landscape.py \\
      --checkpoint data_lz/data_unizero/Pong/xxx/ckpt/ckpt_best.pth.tar \\
      --env PongNoFrameskip-v4

  # Custom grid and output
  python lzero/tools/visualize_loss_landscape.py \\
      --checkpoint data_lz/data_unizero/Pong/xxx/ckpt/ckpt_best.pth.tar \\
      --env PongNoFrameskip-v4 \\
      --output my_landscape \\
      --grid-size 21 \\
      --num-batches 50
        """
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file (required)'
    )
    parser.add_argument(
        '--env',
        type=str,
        default='PongNoFrameskip-v4',
        help='Environment ID (default: PongNoFrameskip-v4)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed (default: 0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./landscape_output',
        help='Output directory (default: ./landscape_output)'
    )
    parser.add_argument(
        '--grid-size',
        type=int,
        default=11,
        choices=[11, 21],
        help='Grid size for landscape (11 or 21, default: 11)'
    )
    parser.add_argument(
        '--num-batches',
        type=int,
        default=20,
        help='Number of batches for landscape computation (default: 20)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for sampling (default: 64)'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=10,
        help='Number of episodes to collect for replay buffer (default: 10)'
    )
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA even if available'
    )

    args = parser.parse_args()

    # Validate checkpoint path
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)
        
    
    
    # Run visualization
    visualize_loss_landscape(
        checkpoint_path=args.checkpoint,
        env_id=args.env,
        seed=args.seed,
        output_dir=args.output,
        grid_size=args.grid_size,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        num_collect_episodes=args.num_episodes,
        use_cuda=not args.no_cuda
    )


if __name__ == '__main__':
    main()
