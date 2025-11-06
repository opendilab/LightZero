import os
import argparse
from typing import Any, Dict

from easydict import EasyDict


def main(env_id: str = 'detective.z5', seed: int = 0, max_env_step: int = int(1e4)) -> None:
    """
    Debug version of Jericho UniZero configuration.

    Optimized for quick testing and debugging with:
    - Minimal environments (1 collector, 1 evaluator)
    - Smaller model (1 layer, 256 embedding)
    - Fewer simulations (5)
    - Smaller batch size (16)
    - Short episode length (20 steps)
    """
    env_id = 'zork1.z5'

    # === DEBUG: Minimal settings ===
    collector_env_num: int = 1       # Minimal collector environments
    evaluator_env_num: int = 1       # Minimal evaluator environments
    n_episode = int(collector_env_num)
    batch_size = 16                  # Small batch size for debugging

    # ------------------------------------------------------------------
    # Base environment parameters
    # ------------------------------------------------------------------
    env_configurations = {
        'detective.z5': (12, 20),    # Reduced max_steps for quick testing
        'omniquest.z5': (25, 20),
        'acorncourt.z5': (45, 20),
        'zork1.z5': (55, 20),
    }

    action_space_size, max_steps = env_configurations.get(env_id, (10, 20))

    # ------------------------------------------------------------------
    # DEBUG: Simplified training parameters
    # ------------------------------------------------------------------
    num_simulations: int = 5         # Minimal MCTS simulations
    num_unroll_steps: int = 5        # Increased from 2 to avoid empty sequences
    infer_context_length: int = 2    # Increased from 1

    num_layers: int = 1              # Single layer model
    replay_ratio: float = 0.05       # Smaller replay ratio
    embed_dim: int = 256             # Smaller embedding (divisible by 8)

    # Reanalysis parameters - disabled for debugging
    buffer_reanalyze_freq: float = 0  # No reanalysis
    reanalyze_batch_size: int = 32
    reanalyze_partition: float = 0.75

    # Model selection
    encoder_option = 'legacy'

    if encoder_option == 'qwen':
        model_name: str = '/mnt/shared-storage-user/tangjia/pr/LightZero/model/models--Qwen--Qwen3-0.6B/snapshots/ec0dcdfc641f7a94f8e969d459caa772b9c01706'
    elif encoder_option == 'legacy':
        model_name: str = '/mnt/shared-storage-user/tangjia/pr/LightZero/model/BAAI--bge-base-en-v1.5'
    else:
        raise ValueError(f"Unsupported encoder option: {encoder_option}")

    # ------------------------------------------------------------------
    # Configuration dictionary
    # ------------------------------------------------------------------
    jericho_unizero_config: Dict[str, Any] = dict(
        env=dict(
            stop_value=int(1e6),
            observation_shape=512,
            max_steps=max_steps,
            max_action_num=action_space_size,
            tokenizer_path=model_name,
            max_seq_len=512,
            game_path=f"./zoo/jericho/envs/z-machine-games-master/jericho-game-suite/{env_id}",
            for_unizero=True,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False),
        ),
        policy=dict(
            multi_gpu=False,
            use_wandb=False,  # Disable wandb for debugging
            learn=dict(
                learner=dict(
                    hook=dict(
                        save_ckpt_after_iter=int(1e6),  # Don't save checkpoints in debug
                    ),
                ),
            ),
            accumulation_steps=1,
            model=dict(
                observation_shape=512,
                action_space_size=action_space_size,
                encoder_option=encoder_option,
                encoder_url=model_name,
                model_type="mlp",
                continuous_action_space=False,
                world_model_cfg=dict(
                    final_norm_option_in_obs_head='LayerNorm',
                    final_norm_option_in_encoder='LayerNorm',
                    predict_latent_loss_type='mse',
                    policy_entropy_weight=5e-2,
                    continuous_action_space=False,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    device="cuda",
                    action_space_size=action_space_size,
                    num_layers=num_layers,
                    num_heads=8,      # 256 / 8 = 32 per head
                    embed_dim=embed_dim,
                    obs_type="text",
                    env_num=max(collector_env_num, evaluator_env_num),

                    task_embed_option=None,
                    use_task_embed=False,
                    use_normal_head=True,
                    use_softmoe_head=False,
                    use_moe_head=False,
                    num_experts_in_moe_head=4,

                    moe_in_transformer=False,
                    multiplication_moe_in_transformer=False,
                    n_shared_experts=1,
                    num_experts_per_tok=1,
                    num_experts_of_moe_in_transformer=8,
                    lora_r=0,
                    lora_alpha=1,
                    lora_dropout=0.0,

                    game_segment_length=64,  # Reduced for debugging
                    decode_loss_mode=None,
                    latent_recon_loss_weight=0.1
                ),
            ),
            update_per_collect=int(collector_env_num*max_steps*replay_ratio),
            action_type="varied_action_space",
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            reanalyze_ratio=0,
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            learning_rate=0.0001,
            cos_lr_scheduler=False,
            fixed_temperature_value=0.25,
            manual_temperature_decay=False,
            num_simulations=num_simulations,
            n_episode=n_episode,
            train_start_after_envsteps=0,
            replay_buffer_size=int(1e4),  # Smaller buffer for debugging
            eval_freq=int(100),            # Evaluate more frequently for debugging
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
            # Label smoothing for training stability
            label_smoothing_eps=0.01,
            policy_ls_eps_start=0.05,
            policy_label_smoothing_eps_end=0.01,

            # Monitor and logging parameters
            monitor_norm_freq=0,  # Disable norm monitoring (0 = disabled)
            grad_clip_value=10.0,  # Gradient clipping

            # Optimizer parameters
            optim_type='Adam',
            weight_decay=1e-4,
            momentum=0.99,

            # Learning rate scheduler parameters
            piecewise_decay_lr_scheduler=False,
            threshold_training_steps_for_final_lr=int(1e6),

            # MCTS and exploration parameters
            collect_num_simulations=num_simulations,
            eval_num_simulations=num_simulations,
            root_dirichlet_alpha=0.25,
            root_noise_weight=0.25,
            target_update_theta=0.005,

            # Data augmentation
            use_augmentation=False,
            augmentation=None,

            # Other parameters
            eps=0.01,
            analysis_sim_norm=False,
            mcts_ctree=False,
            device='cuda',
        ),
    )
    jericho_unizero_config = EasyDict(jericho_unizero_config)

    # ------------------------------------------------------------------
    # Create configuration for importing modules
    # ------------------------------------------------------------------
    jericho_unizero_create_config: Dict[str, Any] = dict(
        env=dict(
            type="jericho",
            import_names=["zoo.jericho.envs.jericho_env"],
        ),
        env_manager=dict(type="base"),
        policy=dict(
            type="unizero",
            import_names=["lzero.policy.unizero"],
        ),
    )
    jericho_unizero_create_config = EasyDict(jericho_unizero_create_config)

    # ------------------------------------------------------------------
    # Combine configurations and construct experiment name
    # ------------------------------------------------------------------
    main_config: EasyDict = jericho_unizero_config
    create_config: EasyDict = jericho_unizero_create_config

    main_config.exp_name = (
        f"debug/unizero_jericho/{env_id}/"
        f"c{collector_env_num}_e{evaluator_env_num}_bs{batch_size}_"
        f"layer{num_layers}_embed{embed_dim}_sim{num_simulations}_seed{seed}"
    )

    from lzero.entry import train_unizero
    # Launch the training process
    train_unizero(
        [main_config, create_config],
        seed=seed,
        model_path=main_config.policy.model_path,
        max_env_step=max_env_step,
    )


if __name__ == "__main__":
    """
    Debug configuration for Jericho UniZero.

    Usage:
        # Run with default settings
        python ./zoo/jericho/configs/jericho_unizero_config_debug.py

        # Run with custom environment
        python ./zoo/jericho/configs/jericho_unizero_config_debug.py --env zork1.z5 --seed 42
    """

    parser = argparse.ArgumentParser(description='Debug Jericho UniZero configuration.')
    parser.add_argument(
        '--env',
        type=str,
        help='Environment identifier (detective.z5, omniquest.z5, etc.)',
        default='detective.z5'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility',
        default=0
    )
    args = parser.parse_args()

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main(args.env, args.seed)
