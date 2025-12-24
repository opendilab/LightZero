"""
Overview:
    This module provides configuration and entry point for UniZero loss landscape visualization on Atari environments.
    It serves as a tool to analyze and visualize the loss landscape of a pretrained UniZero model by:

    1. Loading a pretrained checkpoint (without any training updates)
    2. Using the loaded model to collect game data from the environment
    3. Computing and visualizing the loss landscape based on the collected data

    The loss landscape visualization helps understand the model's weight space geometry and identify
    convergence properties, critical points, and the relationship between different loss metrics.

Usage:
    python zoo/atari/config/atari_unizero_loss_landscape.py --env <env_id> --seed <seed> --ckpt <checkpoint_path> --log_dir <output_dir>

    Examples:
        # Visualize loss landscape for PongNoFrameskip-v4
        python atari_unizero_loss_landscape.py --env PongNoFrameskip-v4 --seed 0 --ckpt model.pth.tar

        # Custom output directory
        python atari_unizero_loss_landscape.py --env Breakout --ckpt /path/to/checkpoint.pth.tar --log_dir ./results
        
Process Flow:
    1. Parse command-line arguments and load configuration
    2. Load the pretrained checkpoint into the model
    3. Initialize data collectors using the loaded model
    4. Collect game data by running episodes with the checkpoint model
    5. Compute loss landscape on multiple points in weight space
    6. Generate visualizations (contour plots, 3D surfaces, heatmaps)
    7. Export results to HDF5 and PDF formats

Output Files:
    - loss_landscape_*.h5           : Loss landscape data in HDF5 format
    - *_2dcontour.pdf               : Contour line plots
    - *_2dcontourf.pdf              : Filled contour plots
    - *_2dheat.pdf                  : Heatmap visualizations
    - *_3dsurface.pdf               : 3D surface plot
    - *.vtp                         : ParaView format for professional rendering

Related Papers:
    - UniZero: "Generalized and Efficient Planning with Scalable Latent World Models" (https://arxiv.org/abs/2406.10667)
    - Loss Landscape: "Visualizing the Loss Landscape of Neural Nets" (https://arxiv.org/abs/1712.09913)

"""

from easydict import EasyDict

from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map


def main(env_id='PongNoFrameskip-v4', seed=0, ckpt=None, log_dir=None):
    action_space_size = atari_env_action_space_map[env_id]

    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    collector_env_num = 8
    num_segments = 8
    game_segment_length = 20
    evaluator_env_num = 10
    num_simulations = 50
    max_env_step = int(5e5)
    batch_size = 64
    num_layers = 2
    replay_ratio = 0.25
    num_unroll_steps = 10
    infer_context_length = 4

    # Defines the frequency of reanalysis. E.g., 1 means reanalyze once per epoch, 2 means reanalyze once every two epochs.
    buffer_reanalyze_freq = 1/50
    # Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
    reanalyze_batch_size = 160
    # The partition of reanalyze. E.g., 1 means reanalyze_batch samples from the whole buffer, 0.5 means samples from the first half of the buffer.
    reanalyze_partition = 0.75
    # ==============================================================
    # end of the most frequently changed config specified by the user
    # ==============================================================
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
            # TODO: only for debug
            # collect_max_episode_steps=int(50),
            # eval_max_episode_steps=int(50),
        ),
        policy=dict(
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000000, ), ), ),  # default is 10000
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
                    max_tokens=2 * num_unroll_steps,  # NOTE: each timestep has 2 tokens: obs and action
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
            # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
            model_path=ckpt,
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
            # ============= The key different params for reanalyze =============
            # Defines the frequency of reanalysis. E.g., 1 means reanalyze once per epoch, 2 means reanalyze once every two epochs.
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            # Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
            reanalyze_batch_size=reanalyze_batch_size,
            # The partition of reanalyze. E.g., 1 means reanalyze_batch samples from the whole buffer, 0.5 means samples from the first half of the buffer.
            reanalyze_partition=reanalyze_partition,
        ),
    )
    atari_unizero_config = EasyDict(atari_unizero_config)
    main_config = atari_unizero_config

    atari_unizero_create_config = dict(
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
    atari_unizero_create_config = EasyDict(atari_unizero_create_config)
    create_config = atari_unizero_create_config

    # Set exp_name based on log_dir if provided, otherwise use default
    if log_dir is not None:
        main_config.exp_name = log_dir
    else:
        main_config.exp_name = f'data_lz/data_unizero/{env_id[:-14]}/{env_id[:-14]}_uz_nlayer{num_layers}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'

    from lzero.entry import train_unizero_with_loss_landscape
    train_unizero_with_loss_landscape([main_config, create_config], seed=seed, model_path=main_config.policy.model_path, max_env_step=max_env_step)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some environment.')
    parser.add_argument('--env', type=str, help='The environment to use', default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    parser.add_argument('--ckpt', type=str, help='Path to pretrained checkpoint. The checkpoint will be loaded (NOT trained) to collect data and visualize the loss landscape.', default=None)
    parser.add_argument('--log_dir', type=str, help='Log directory for output', default=None)
    args = parser.parse_args()

    main(args.env, args.seed, args.ckpt, args.log_dir)

