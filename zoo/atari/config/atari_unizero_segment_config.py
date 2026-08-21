from easydict import EasyDict

from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map


def build_config(env_id='ALE/Pong-v5', seed=0, max_env_step_override=None):
    """Build the stable UniZero Atari segment baseline.

    Keep this entry point intentionally small and close to
    ``atari_unizero_config.py``. Performance experiments and operational
    overrides belong in ``atari_unizero_segment_experimental_config.py``.
    """
    action_space_size = atari_env_action_space_map[env_id]

    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    collector_env_num = 8
    evaluator_env_num = 3
    num_segments = 8
    game_segment_length = 20
    num_simulations = 50
    max_env_step = int(5e5)
    batch_size = 64
    num_unroll_steps = 10
    infer_context_length = 4
    num_layers = 2
    replay_ratio = 0.25
    buffer_reanalyze_freq = 1 / 100000
    reanalyze_batch_size = 160
    reanalyze_partition = 0.75
    # ==============================================================
    # end of the most frequently changed config specified by the user
    # ==============================================================
    if max_env_step_override is not None:
        max_env_step = int(max_env_step_override)
        if max_env_step <= 0:
            raise ValueError(f'max_env_step must be positive, got {max_env_step}')

    main_config = EasyDict(
        dict(
            env=dict(
                stop_value=int(1e6),
                env_id=env_id,
                observation_shape=(3, 64, 64),
                gray_scale=False,
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
                n_evaluator_episode=evaluator_env_num,
                manager=dict(shared_memory=False),
            ),
            policy=dict(
                learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000000))),
                model=dict(
                    observation_shape=(3, 64, 64),
                    action_space_size=action_space_size,
                    world_model_cfg=dict(
                        policy_entropy_weight=1e-4,
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
                        encoder_type='resnet',
                        env_num=max(collector_env_num, evaluator_env_num),
                        rotary_emb=False,
                    ),
                ),
                model_path=None,
                num_unroll_steps=num_unroll_steps,
                num_segments=num_segments,
                replay_ratio=replay_ratio,
                batch_size=batch_size,
                learning_rate=0.0001,
                num_simulations=num_simulations,
                train_start_after_envsteps=2000,
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
    )

    create_config = EasyDict(
        dict(
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
    )

    game_name = env_id.split('/')[-1].split('-')[0]
    main_config.exp_name = (
        f'data_lz/data_unizero_segment/{game_name}/'
        f'{game_name}_uz_nlayer{num_layers}_gsl{game_segment_length}'
        f'_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}'
        f'_bs{batch_size}_seed{seed}'
    )
    return main_config, create_config, max_env_step


def main(env_id='ALE/Pong-v5', seed=0, max_env_step_override=None):
    main_config, create_config, max_env_step = build_config(
        env_id, seed, max_env_step_override=max_env_step_override
    )

    from lzero.entry import train_unizero_segment
    train_unizero_segment(
        [main_config, create_config],
        seed=seed,
        model_path=main_config.policy.model_path,
        max_env_step=max_env_step,
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train the stable UniZero Atari segment baseline.')
    parser.add_argument('--env', type=str, default='ALE/Pong-v5', help='Atari environment id.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument(
        '--max-env-step',
        type=int,
        default=None,
        help='Override the default training budget of 500000 environment steps.',
    )
    args = parser.parse_args()
    main(args.env, args.seed, max_env_step_override=args.max_env_step)
