import os
import argparse

from easydict import EasyDict


def main(env_id: str = 'detective.z5', seed: int = 0) -> None:
    # ------------------------------------------------------------------
    # Base configurations
    # ------------------------------------------------------------------
    env_configurations = {
        'detective.z5': (12, 100),
        'omniquest.z5': (25, 100),
        'acorncourt.z5': (45, 50),
        'zork1.z5': (55, 500),
    }

    # Set action_space_size and max_steps based on env_id
    action_space_size, max_steps = env_configurations.get(env_id, (10, 50))  # Default values if env_id not found

    # ==============================================================
    # Frequently changed configurations (user-specified)
    # ==============================================================
    # Model name or path - configurable according to the predefined model paths or names
    encoder_option = 'legacy'        # ['qwen', 'legacy']. Legacy uses the bge encoder

    if encoder_option == 'qwen':
        model_name: str = 'Qwen/Qwen3-0.6B'
    elif encoder_option == 'legacy':
        model_name: str = 'BAAI/bge-base-en-v1.5'
    else:
        raise ValueError(f"Unsupported encoder option: {encoder_option}")  


    collector_env_num = 8
    game_segment_length = 20
    evaluator_env_num = 5
    num_segments = 8
    num_simulations = 50
    max_env_step = int(5e5)
    batch_size = 64
    num_unroll_steps = 10
    infer_context_length = 4
    num_layers = 2
    replay_ratio = 0.25
    embed_dim = 768

    # Reanalysis parameters
    buffer_reanalyze_freq = 1 / 100000
    reanalyze_batch_size = 160
    reanalyze_partition = 0.75

    # =========== Debug configurations ===========
    # collector_env_num = 2
    # num_segments = 2
    # max_steps = 20
    # game_segment_length = 20
    # evaluator_env_num = 2
    # num_simulations = 5
    # max_env_step = int(5e5)
    # batch_size = 10
    # num_unroll_steps = 5
    # infer_context_length = 2
    # num_layers = 1
    # replay_ratio = 0.05
    # embed_dim = 24

    # ------------------------------------------------------------------
    # Construct Jericho Unizero Segment configuration dictionary
    # ------------------------------------------------------------------
    jericho_unizero_config = dict(
        env=dict(
            stop_value=int(1e6),
            max_steps=max_steps,
            observation_shape=512,
            max_action_num=action_space_size,
            tokenizer_path=model_name,
            max_seq_len=512,
            game_path=f"./zoo/jericho/envs/z-machine-games-master/jericho-game-suite/{env_id}",
            for_unizero=True,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False),
            use_cache=True,
            cache_size=100000,
        ),
        policy=dict(
            learn=dict(
                learner=dict(
                    hook=dict(
                        save_ckpt_after_iter=1000000,
                    ),
                ),
            ),
            model=dict(
                observation_shape=512,
                action_space_size=action_space_size,
                encoder_option=encoder_option,
                encoder_url=model_name,
                model_type="mlp",
                world_model_cfg=dict(
                    final_norm_option_in_obs_head='LayerNorm',
                    final_norm_option_in_encoder='LayerNorm',
                    predict_latent_loss_type='mse',
                    policy_entropy_weight=5e-3,
                    continuous_action_space=False,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,  # each timestep represents 2 tokens: observation and action
                    context_length=2 * infer_context_length,
                    device="cuda",
                    action_space_size=action_space_size,
                    num_layers=num_layers,
                    num_heads=24,
                    embed_dim=embed_dim,
                    obs_type="text",
                    env_num=max(collector_env_num, evaluator_env_num),
                    decode_loss_mode='None', # Controls where to compute reconstruction loss: after_backbone, before_backbone, or None.
                    latent_recon_loss_weight=0.1 
                ),
            ),
            action_type="varied_action_space",
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            reanalyze_ratio=0,
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            learning_rate=0.0001,
            fixed_temperature_value=0.25,
            manual_temperature_decay=False,
            num_simulations=num_simulations,
            num_segments=num_segments,
            train_start_after_envsteps=0,
            game_segment_length=game_segment_length,
            replay_buffer_size=int(1e6),
            eval_freq=int(5e3),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
        ),
    )
    jericho_unizero_config = EasyDict(jericho_unizero_config)

    # ------------------------------------------------------------------
    # Create configuration for module import
    # ------------------------------------------------------------------
    jericho_unizero_create_config = dict(
        env=dict(
            type="jericho",
            import_names=["zoo.jericho.envs.jericho_env"],
        ),
        env_manager=dict(type="base"),  # Use base env manager to avoid subprocess bugs.
        policy=dict(
            type="unizero",
            import_names=["lzero.policy.unizero"],
        ),
    )
    jericho_unizero_create_config = EasyDict(jericho_unizero_create_config)

    main_config = jericho_unizero_config
    create_config = jericho_unizero_create_config

    # Construct experiment name using key parameters
    main_config.exp_name = (
        f"data_lz/data_unizero/{env_id[:-14]}/{env_id[:-14]}_uz_nlayer{num_layers}_gsl{game_segment_length}_"
        f"rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}"
    )

    from lzero.entry import train_unizero_segment

    # Launch the segment training process
    train_unizero_segment(
        [main_config, create_config],
        seed=seed,
        model_path=main_config.policy.model_path,
        max_env_step=max_env_step,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process environment configuration for Unizero Segment training."
    )
    parser.add_argument(
        '--env',
        type=str,
        help="The environment to use, e.g., detective.z5",
        default='detective.z5'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help="The seed to use",
        default=0
    )
    args = parser.parse_args()

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main(args.env, args.seed)