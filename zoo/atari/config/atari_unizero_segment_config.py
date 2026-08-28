"""Stable UniZero Atari config: the best recipe from the 2026-08 3M matrix.

This mirrors the winning ``v3`` arm of the preregistered MsPacman 3M experiment
matrix (the historical best recipe with ``value_loss_weight=0.5``) with the
experimental mechanisms (raw-token KV rebuild, open-loop consistency loss,
contextual value bootstrap) turned off. Those and other ablation/diagnostic
knobs live in ``atari_unizero_segment_experimental_config.py``, whose defaults
reproduce the full v3 recipe.
"""

from easydict import EasyDict

from zoo.atari.config._atari_unizero_segment_utils import _resolve_grad_clip_mode, _safe_run_name
from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map


def build_config(
        env_id='ALE/MsPacman-v5',
        seed=0,
        max_env_step_override=None,
        use_augmentation=False,
        grad_clip_mode_override=None,
        run_name=None,
):
    action_space_size = atari_env_action_space_map[env_id]
    use_augmentation = bool(use_augmentation)
    grad_clip_mode = _resolve_grad_clip_mode(use_augmentation, grad_clip_mode_override)

    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    collector_env_num = 8
    evaluator_env_num = 8
    num_segments = 8
    game_segment_length = 200
    num_simulations = 50
    collect_num_simulations = 25
    max_env_step = int(3e6)
    batch_size = 256
    num_unroll_steps = 10
    infer_context_length = 5
    num_layers = 2
    replay_ratio = 0.1
    buffer_reanalyze_freq = 2e-10
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
                learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=50000))),
                model=dict(
                    observation_shape=(3, 64, 64),
                    action_space_size=action_space_size,
                    reward_support_range=(-300., 301., 1.),
                    value_support_range=(-300., 301., 1.),
                    norm_type='LN',
                    num_res_blocks=1,
                    num_channels=64,
                    world_model_cfg=dict(
                        latent_recon_loss_weight=0.0,
                        perceptual_loss_weight=0.0,
                        norm_type='LN',
                        final_norm_option_in_obs_head='LayerNorm',
                        final_norm_option_in_encoder='LayerNorm',
                        predict_latent_loss_type='mse',
                        support_size=601,
                        # Fixed entropy coefficient; adaptive alpha is disabled below.
                        policy_entropy_weight=0.005,
                        max_blocks=num_unroll_steps,
                        max_tokens=2 * num_unroll_steps,
                        context_length=2 * infer_context_length,
                        action_space_size=action_space_size,
                        num_layers=num_layers,
                        num_heads=8,
                        embed_dim=768,
                        env_num=collector_env_num,
                        num_simulations=num_simulations,
                        game_segment_length=game_segment_length,
                        device='cuda',
                        # Model-side value_priority diagnostic stays on even though the
                        # buffer samples uniformly (policy.use_priority=False below).
                        use_priority=True,
                        encoder_type='resnet',
                        use_normal_head=True,
                        optim_type='AdamW_mix_lr_wdecay',
                        root_cache_key_round_decimals=0,
                        # v3 rebuilds the KV window exactly from retained raw tokens; the
                        # stable config keeps the legacy update path (experimental-only).
                        # rebuild_kv_window_from_tokens=True,
                        rebuild_kv_window_from_tokens=False,
                        # v3 enables the short differentiable MCTS-style latent rollout
                        # consistency loss; disabled here (experimental-only).
                        # open_loop_consistency_loss_weight=1.0,
                        open_loop_consistency_loss_weight=0,
                        open_loop_consistency_batch_size=8,
                        open_loop_consistency_horizon=4,
                        open_loop_prefix_transitions=3,
                        # Policy-stability protections (soft_tanh policy-logits clip +-10).
                        use_policy_logits_clip=True,
                        policy_logits_clip_method='soft_tanh',
                        policy_logits_clip_min=-10.0,
                        policy_logits_clip_max=10.0,
                    ),
                ),
                model_path=None,
                # Learning settings
                optim_type='AdamW_mix_lr_wdecay',
                learning_rate=0.0001,
                weight_decay=1e-2,
                batch_size=batch_size,
                replay_ratio=replay_ratio,
                num_unroll_steps=num_unroll_steps,
                num_segments=num_segments,
                game_segment_length=game_segment_length,
                # KV caches are cleared once per env per this many env steps.
                kv_cache_clear_interval=2000,
                empty_cuda_cache_on_cache_reset=True,
                num_simulations=num_simulations,
                collect_num_simulations=collect_num_simulations,
                fixed_temperature_value=0.25,
                obs_loss_weight=10.0,
                value_loss_weight=0.5,
                grad_clip_value=5.0,
                grad_clip_mode=grad_clip_mode,
                use_augmentation=use_augmentation,
                augmentation=['shift', 'intensity'],
                use_adaptive_entropy_weight=False,
                # Policy label smoothing disabled; value/reward smoothing is unchanged.
                policy_ls_eps_start=0.0,
                policy_ls_eps_end=0.0,
                label_smoothing_eps=0.1,
                use_continuous_label_smoothing=True,
                continuous_ls_eps=0.05,
                monitor_norm_freq=10000,
                use_enhanced_policy_monitoring=True,
                # use_priority=False,
                use_priority=True,
                priority_prob_alpha=0.6,
                priority_prob_beta=0.4,
                # Reanalyze settings
                buffer_reanalyze_freq=buffer_reanalyze_freq,
                reanalyze_search_chunk_size=collector_env_num,
                reanalyze_batch_size=reanalyze_batch_size,
                reanalyze_partition=reanalyze_partition,
                # v3 computes TD bootstrap values from the rolling online-planning
                # history; the stable config uses the training-only sequence context.
                # bootstrap_value_context=True,
                bootstrap_value_context=False,
                # Environment settings
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
                eval_freq=int(5e3),
                replay_buffer_size=int(5e5),
                # Policy checkpoints omit the replay buffer. Refill a small diverse
                # on-policy window before updating a resumed mature model.
                resume_buffer_min_transitions=100000,
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
    if run_name is None:
        augmentation_tag = (
            'fixed-aug' if use_augmentation and grad_clip_mode == 'separate_encoder'
            else 'aug-globalclip' if use_augmentation
            else 'baseline'
        )
        run_name = (
            f'{game_name}_uz_nlayer{num_layers}_gsl{game_segment_length}'
            f'_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}'
            f'_bs{batch_size}_seed{seed}_per_{augmentation_tag}'
        )
    else:
        run_name = _safe_run_name(run_name)
    main_config.exp_name = (
        f'data_unizero_segment/{game_name}/{run_name}'
    )
    return main_config, create_config, max_env_step


def main(
        env_id='ALE/MsPacman-v5',
        seed=0,
        max_env_step_override=None,
        use_augmentation=False,
        grad_clip_mode_override=None,
        run_name=None,
):
    main_config, create_config, max_env_step = build_config(
        env_id,
        seed,
        max_env_step_override=max_env_step_override,
        use_augmentation=use_augmentation,
        grad_clip_mode_override=grad_clip_mode_override,
        run_name=run_name,
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

    parser = argparse.ArgumentParser(description='Train the stable UniZero Atari baseline or fixed-augmentation recipe.')
    parser.add_argument('--env', type=str, default='ALE/MsPacman-v5', help='Atari environment id.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    augmentation_group = parser.add_mutually_exclusive_group()
    augmentation_group.add_argument(
        '--use-augmentation', dest='use_augmentation', action='store_true',
        help='Enable coherent shift/intensity augmentation with separate encoder clipping by default.',
    )
    augmentation_group.add_argument(
        '--no-augmentation', dest='use_augmentation', action='store_false',
        help='Disable augmentation and use global clipping by default (baseline).',
    )
    parser.set_defaults(use_augmentation=False)
    parser.add_argument(
        '--grad-clip-mode', choices=('global', 'separate_encoder'), default=None,
        help='Override clipping topology; normally inferred from the augmentation setting.',
    )
    parser.add_argument(
        '--run-name', type=str, default=None,
        help='Optional unique run-directory basename under data_unizero_segment/<game>/.',
    )
    parser.add_argument(
        '--max-env-step',
        type=int,
        default=None,
        help='Override the default training budget of 3000000 environment steps.',
    )
    args = parser.parse_args()
    main(
        args.env,
        args.seed,
        max_env_step_override=args.max_env_step,
        use_augmentation=args.use_augmentation,
        grad_clip_mode_override=args.grad_clip_mode,
        run_name=args.run_name,
    )
