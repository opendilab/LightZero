import json
import os
import re
import socket
import sys
import traceback
from datetime import datetime

from easydict import EasyDict
from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map


class _Tee:
    """Mirror all output to both the terminal and console.log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return any(getattr(s, 'isatty', lambda: False)() for s in self.streams)

    def fileno(self):
        return self.streams[0].fileno()

    @property
    def encoding(self):
        return getattr(self.streams[0], 'encoding', 'utf-8')

def _atari_game_name(env_id):
    return env_id.split('/')[-1].split('-')[0]


def _safe_run_name(value):
    value = re.sub(r'[^A-Za-z0-9_.-]+', '-', value).strip('-_.')
    if not value:
        raise ValueError('run_name must contain at least one letter or number')
    return value


def main(
        env_id,
        seed,
        output_root='data_unizero',
        run_name=None,
        use_new_cache_manager=False,
        disable_adaptive_alpha=True,
        fixed_alpha=5e-3,
        disable_policy_label_smoothing=True,
        resume_from=None,
        max_env_step_override=None,
        use_priority=None,
        stab_fix=False,
        game_segment_length_override=None,
        save_ckpt_after_iter_override=None,
        legacy_resume_alpha=None,
):
    action_space_size = atari_env_action_space_map[env_id]

    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    collector_env_num = 8
    num_segments = 8
    evaluator_env_num = 8  # 3->8: reduce eval reward_mean noise for attribution (takes effect on next run restart)

    # game_segment_length=20 makes only 20-(num_unroll_steps+td_steps)=5 of the 20 positions in each
    # non-terminal segment eligible as sampling roots (valid_len in game_buffer._push_game_segment),
    # i.e. 75% of the collected transitions can never be trained on. 200 restores ~92.5% coverage.
    game_segment_length = 200 if game_segment_length_override is None else int(game_segment_length_override)
    save_ckpt_after_iter = (
        50000 if save_ckpt_after_iter_override is None else int(save_ckpt_after_iter_override)
    )
    if save_ckpt_after_iter <= 0:
        raise ValueError(f'save_ckpt_after_iter must be positive, got {save_ckpt_after_iter}')
    if legacy_resume_alpha is not None and legacy_resume_alpha <= 0:
        raise ValueError(f'legacy_resume_alpha must be positive, got {legacy_resume_alpha}')
    num_unroll_steps = 10
    infer_context_length = 4

    num_simulations = 50
    batch_size = 256
    replay_ratio = 0.1

    num_layers = 2
    norm_type = "LN"

    if env_id == 'ALE/Pong-v5':
        max_env_step = int(1e6)
    else:
        max_env_step = int(10e6)
    if max_env_step_override is not None:
        max_env_step = int(max_env_step_override)

    # Reanalyze settings
    buffer_reanalyze_freq = 1/5000000000
    reanalyze_batch_size = 160
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
        ),
        policy=dict(
            model=dict(
                observation_shape=(3, 64, 64),
                action_space_size=action_space_size,
                reward_support_range=(-300., 301., 1.),
                value_support_range=(-300., 301., 1.),
                norm_type=norm_type,
                # num_res_blocks=2,
                # num_channels=128,
                num_res_blocks=1,
                num_channels=64,
                world_model_cfg=dict(
                    latent_recon_loss_weight=0.0,
                    perceptual_loss_weight=0.0,
                    norm_type=norm_type,
                    final_norm_option_in_obs_head='LayerNorm',
                    final_norm_option_in_encoder='LayerNorm',
                    predict_latent_loss_type='mse',
                    support_size=601,
                    # Used as the fixed entropy coefficient when adaptive alpha is disabled.
                    policy_entropy_weight=fixed_alpha,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    action_space_size=action_space_size,
                    num_layers=num_layers,
                    num_heads=8,
                    embed_dim=768,
                    env_num=max(collector_env_num, evaluator_env_num),
                    num_simulations=num_simulations,
                    game_segment_length=game_segment_length,
                    device='cuda',
                    use_priority=True,
                    encoder_type='resnet',
                    use_normal_head=True,
                    optim_type='AdamW_mix_lr_wdecay',
                    use_new_cache_manager=use_new_cache_manager,
                    # Policy-stability protections (500K crash chain: extreme target_policy ->
                    # logits explosion -> x_token collapse). Off by default; --stab-fix enables.
                    use_policy_logits_clip=stab_fix,
                    policy_logits_clip_method='soft_tanh',
                    policy_logits_clip_min=-10.0,
                    policy_logits_clip_max=10.0,
                ),
            ),
            # Learning settings
            optim_type='AdamW_mix_lr_wdecay',
            learning_rate=0.0001,
            weight_decay=1e-2,
            batch_size=batch_size,
            replay_ratio=replay_ratio,
            num_unroll_steps=num_unroll_steps,
            num_segments=num_segments,
            game_segment_length=game_segment_length,
            # Full learner checkpoints are ~530MB each; save every 50k train iters instead of
            # the default 10k to bound disk usage. ckpt_best (on new best eval) is unaffected.
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=save_ckpt_after_iter))),
            # KV caches are cleared once per env per this many env steps. Was hardcoded to
            # game_segment_length, which wiped all MCTS kv caches after every single segment.
            kv_cache_clear_interval=2000,
            num_simulations=num_simulations,
            grad_clip_value=5,
            use_augmentation=False,

            # Adaptive target entropy settings from the 2025 Pong run.
            use_adaptive_entropy_weight=not disable_adaptive_alpha,
            adaptive_entropy_alpha_lr=1e-4,
            legacy_resume_adaptive_alpha=legacy_resume_alpha,
            target_entropy_start_ratio=0.98,
            target_entropy_end_ratio=0.7,
            target_entropy_decay_steps=100000,

            # Encoder latent norm clipping — matches the 2025-10-10 Pong run that
            # converged to reward_mean=20 at ~200k env steps.
            # The successful run used encoder-clip 30→10 over 100k steps (cosine).
            # With use_encoder_clip_annealing=False the clip code was unreachable
            # (bug fixed in unizero.py); keeping annealing=True mirrors the known-good
            # baseline and gradually tightens the clip as the model stabilises.
            use_encoder_clip_annealing=True,
            encoder_clip_anneal_type='cosine',
            encoder_clip_start_value=30.0,
            encoder_clip_end_value=10.0,
            encoder_clip_anneal_steps=100000,
            latent_norm_clip_threshold=10.0,  # fallback fixed threshold once annealing completes

            # Policy smoothing decays 0.05->0.01; value/reward use 0.1.
            policy_ls_eps_start=0.0 if disable_policy_label_smoothing else 0.05,
            policy_ls_eps_end=0.0 if disable_policy_label_smoothing else 0.01,
            policy_ls_eps_decay_steps=50000,
            label_smoothing_eps=0.1,
            use_continuous_label_smoothing=stab_fix,
            continuous_ls_eps=0.05,
            monitor_norm_freq=10000,
            # Always-on enhanced policy monitoring: without this flag the whitelisted
            # policy_logits/* and target_policy_entropy/{mean,min,max,std} log keys print
            # constant 0.0 (empty-record averaging), hiding real logits/entropy stats.
            use_enhanced_policy_monitoring=True,

            # Priority settings.
            # Default ON. A 2026-08-06 Pong A/B suggested PER-off learned faster in the first
            # ~25k train iters, but the advantage did not reproduce at later iterations (PER-on
            # matched or exceeded PER-off by iter 35k-40k), so PER remains enabled by default.
            # NOTE: model.world_model_cfg.use_priority=True is kept in sync so the [B]
            # value_priority tensor is computed and update_priority() stays shape-compatible;
            # setting use_priority=False here switches the buffer to uniform sampling and
            # priority write-backs are then discarded.
            use_priority=True if use_priority is None else use_priority,
            priority_prob_alpha=0.6,
            priority_prob_beta=0.4,

            # Reanalyze settings
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,

            # Environment settings
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            eval_freq=int(5e3),
            replay_buffer_size=int(5e5),
            # Policy checkpoints omit the ~25GB full Atari replay buffer. Refill a small diverse
            # on-policy window before updating a mature model after preemption.
            resume_buffer_min_transitions=10000,
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

    # ============ use muzero_segment_collector instead of muzero_collector =============
    from lzero.entry import train_unizero_segment
    game_name = _atari_game_name(env_id)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if run_name is None:
        run_name = f'{game_name.lower()}/{game_name.lower()}_sync_seed{seed}_{timestamp}'
    run_name = _safe_run_name(run_name)

    # LightZero internally prefixes exp_name with "./", so keep it relative to
    # the current working directory even when an absolute output root is given.
    run_dir = os.path.relpath(os.path.abspath(os.path.join(output_root, run_name)), os.getcwd())
    if os.path.exists(run_dir):
        raise FileExistsError(f'Run directory already exists: {os.path.abspath(run_dir)}')
    os.makedirs(run_dir)
    main_config.exp_name = run_dir
    with open(os.path.join(run_dir, 'pid'), 'w', encoding='utf-8') as file:
        file.write(f'{os.getpid()}\n')

    original_stdout, original_stderr = sys.stdout, sys.stderr
    console_path = os.path.join(run_dir, 'console.log')
    with open(console_path, 'a', encoding='utf-8', buffering=1) as console:
        sys.stdout = _Tee(original_stdout, console)
        sys.stderr = _Tee(original_stderr, console)
        try:
            print(f'Run directory: {os.path.abspath(run_dir)}')
            train_unizero_segment(
                [main_config, create_config], seed=seed, model_path=resume_from, max_env_step=max_env_step
            )
        except BaseException:
            traceback.print_exc(file=sys.stderr)
            raise
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = original_stdout, original_stderr


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process different environments and seeds.')
    parser.add_argument('--env', type=str, help='The environment to use', default='ALE/Pong-v5')
    parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    parser.add_argument(
        '--output-root', type=str, default='data_unizero',
        help='Root directory containing one self-contained folder per run.'
    )
    parser.add_argument(
        '--run-name', type=str, default=None,
        help='Optional unique run folder name; defaults to env/variant/seed/timestamp.'
    )
    parser.add_argument(
        '--use-new-cache-manager', action='store_true',
        help='Enable the new UniZero KV cache manager; disabled by default for baseline compatibility.'
    )
    adaptive_alpha_group = parser.add_mutually_exclusive_group()
    adaptive_alpha_group.add_argument(
        '--use-adaptive-alpha', dest='disable_adaptive_alpha', action='store_false',
        help='Enable adaptive entropy alpha instead of the fixed coefficient.'
    )
    adaptive_alpha_group.add_argument(
        '--disable-adaptive-alpha', dest='disable_adaptive_alpha', action='store_true',
        help='Disable adaptive entropy alpha and use --fixed-alpha instead (default).'
    )
    parser.add_argument(
        '--fixed-alpha', type=float, default=5e-3,
        help='Fixed policy entropy coefficient used when adaptive alpha is disabled.'
    )
    policy_smoothing_group = parser.add_mutually_exclusive_group()
    policy_smoothing_group.add_argument(
        '--enable-policy-label-smoothing', dest='disable_policy_label_smoothing', action='store_false',
        help='Enable the 0.05 to 0.01 policy label-smoothing schedule.'
    )
    policy_smoothing_group.add_argument(
        '--disable-policy-label-smoothing', dest='disable_policy_label_smoothing', action='store_true',
        help='Set policy label-smoothing epsilon to zero (default); value/reward smoothing is unchanged.'
    )
    parser.set_defaults(disable_adaptive_alpha=True, disable_policy_label_smoothing=True)
    parser.add_argument(
        '--resume-from', dest='resume_from', type=str, default=None,
        help='Optional learner checkpoint path to resume weights/optimizer/train_iter/envstep from.'
    )
    parser.add_argument(
        '--max-env-step', dest='max_env_step', type=int, default=None,
        help='Override the default max env-step budget (e.g. for continuing a run past its cap).'
    )
    priority_group = parser.add_mutually_exclusive_group()
    priority_group.add_argument(
        '--use-priority', dest='use_priority', action='store_true',
        help='Force prioritized replay on (this is the default).'
    )
    priority_group.add_argument(
        '--no-priority', dest='use_priority', action='store_false',
        help='Disable prioritized replay (uniform sampling) for ablations.'
    )
    parser.set_defaults(use_priority=None)
    parser.add_argument(
        '--stab-fix', dest='stab_fix', action='store_true',
        help='Enable policy-stability protections (soft_tanh policy-logits clip +-10 + '
             'continuous label smoothing eps=0.05) from the 500K-crash fix bundle.'
    )
    parser.add_argument(
        '--game-segment-length', dest='game_segment_length', type=int, default=None,
        help='Override game_segment_length (default 200; the 2025-10 known-good Pong run used 20).'
    )
    parser.add_argument(
        '--save-ckpt-after-iter', dest='save_ckpt_after_iter', type=int, default=None,
        help='Override periodic learner-checkpoint interval; useful on preemptible clusters.'
    )
    parser.add_argument(
        '--legacy-resume-alpha', dest='legacy_resume_alpha', type=float, default=None,
        help='Adaptive alpha recorded externally for a legacy checkpoint without log_alpha.'
    )
    args = parser.parse_args()

    main(
        args.env,
        args.seed,
        output_root=args.output_root,
        run_name=args.run_name,
        use_new_cache_manager=args.use_new_cache_manager,
        disable_adaptive_alpha=args.disable_adaptive_alpha,
        fixed_alpha=args.fixed_alpha,
        disable_policy_label_smoothing=args.disable_policy_label_smoothing,
        resume_from=args.resume_from,
        max_env_step_override=args.max_env_step,
        use_priority=args.use_priority,
        stab_fix=args.stab_fix,
        game_segment_length_override=args.game_segment_length,
        save_ckpt_after_iter_override=args.save_ckpt_after_iter,
        legacy_resume_alpha=args.legacy_resume_alpha,
    )
