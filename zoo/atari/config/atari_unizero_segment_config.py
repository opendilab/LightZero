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


def _prepare_run_directory(run_dir, resume_from=None, resume_in_place=False):
    """Create a run directory, or explicitly reopen it for checkpoint recovery."""
    if resume_in_place and resume_from is None:
        raise ValueError('resume_in_place requires resume_from')
    if os.path.exists(run_dir):
        if not resume_in_place:
            raise FileExistsError(f'Run directory already exists: {os.path.abspath(run_dir)}')
        if not os.path.isdir(run_dir):
            raise NotADirectoryError(f'Run path is not a directory: {os.path.abspath(run_dir)}')
        return
    os.makedirs(run_dir)


def _experimental_config_overrides(
        infer_context_length=None,
        bootstrap_value_context=False,
        open_loop_consistency_weight=None,
        open_loop_recurrent_weight=None,
        open_loop_consistency_batch_size=None,
        open_loop_consistency_horizon=None,
        open_loop_prefix_transitions=None,
        encoder_clip_enabled=None,
):
    """Build only explicitly requested UniZero experimental config overrides.

    The stable disabled defaults live in ``UniZeroPolicy.config``.  Keeping this helper sparse
    prevents Atari launch configs from copying policy defaults and makes every behavioral change
    visible at the command line.
    """
    policy_overrides = {}
    world_model_overrides = {}

    if infer_context_length is not None:
        infer_context_length = int(infer_context_length)
        if infer_context_length < 1:
            raise ValueError(
                f'infer_context_length must be positive, got {infer_context_length}'
            )
        world_model_overrides['context_length'] = 2 * infer_context_length

    if bootstrap_value_context:
        policy_overrides['bootstrap_value_context'] = True

    consistency_weight = (
        0. if open_loop_consistency_weight is None
        else float(open_loop_consistency_weight)
    )
    recurrent_weight = (
        0. if open_loop_recurrent_weight is None
        else float(open_loop_recurrent_weight)
    )
    if consistency_weight < 0:
        raise ValueError(
            f'open_loop_consistency_weight must be non-negative, got {consistency_weight}'
        )
    if recurrent_weight < 0:
        raise ValueError(
            f'open_loop_recurrent_weight must be non-negative, got {recurrent_weight}'
        )
    if consistency_weight > 0 and recurrent_weight > 0:
        raise ValueError(
            'open_loop_consistency_weight and open_loop_recurrent_weight are mutually exclusive'
        )
    if open_loop_consistency_weight is not None:
        world_model_overrides['open_loop_consistency_loss_weight'] = consistency_weight
    if open_loop_recurrent_weight is not None:
        world_model_overrides['open_loop_recurrent_loss_weight'] = recurrent_weight

    if open_loop_consistency_batch_size is not None:
        batch_size = int(open_loop_consistency_batch_size)
        if batch_size <= 0:
            raise ValueError('open_loop_consistency_batch_size must be positive')
        world_model_overrides['open_loop_consistency_batch_size'] = batch_size
    if open_loop_consistency_horizon is not None:
        horizon = int(open_loop_consistency_horizon)
        if horizon <= 0:
            raise ValueError('open_loop_consistency_horizon must be positive')
        world_model_overrides['open_loop_consistency_horizon'] = horizon
    if open_loop_prefix_transitions is not None:
        prefix_transitions = int(open_loop_prefix_transitions)
        if prefix_transitions < 0:
            raise ValueError('open_loop_prefix_transitions must be non-negative')
        world_model_overrides['open_loop_prefix_transitions'] = prefix_transitions

    if encoder_clip_enabled is not None:
        encoder_clip_enabled = bool(encoder_clip_enabled)
        policy_overrides.update(
            use_encoder_clip_annealing=encoder_clip_enabled,
            latent_norm_clip_threshold=10.0 if encoder_clip_enabled else 0.0,
        )

    return policy_overrides, world_model_overrides


def main(
        env_id,
        seed,
        output_root='data_unizero',
        run_name=None,
        use_new_cache_manager=False,
        evaluator_env_num_override=None,
        collect_num_simulations_override=None,
        disable_adaptive_alpha=True,
        fixed_alpha=5e-3,
        disable_policy_label_smoothing=True,
        encoder_clip_enabled=None,
        resume_from=None,
        resume_in_place=False,
        max_env_step_override=None,
        use_priority=None,
        stab_fix=False,
        game_segment_length_override=None,
        infer_context_length_override=None,
        exact_kv_window_reset=False,
        rebuild_kv_window_from_tokens=False,
        contextual_reanalysis=False,
        bootstrap_value_context=False,
        resume_buffer_min_transitions_override=None,
        buffer_reanalyze_freq_override=None,
        save_ckpt_after_iter_override=None,
        periodic_ckpt_keep_last_override=None,
        open_loop_diagnostic_freq_override=None,
        open_loop_consistency_weight_override=None,
        open_loop_recurrent_weight_override=None,
        open_loop_consistency_batch_size_override=None,
        open_loop_consistency_horizon_override=None,
        open_loop_prefix_transitions_override=None,
        legacy_resume_alpha=None,
):
    action_space_size = atari_env_action_space_map[env_id]
    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    collector_env_num = 8
    num_segments = 8
    evaluator_env_num = (
        3 if evaluator_env_num_override is None else int(evaluator_env_num_override)
    )
    if evaluator_env_num <= 0:
        raise ValueError(f'evaluator_env_num must be positive, got {evaluator_env_num}')
    collect_num_simulations = (
        None
        if collect_num_simulations_override is None
        else int(collect_num_simulations_override)
    )
    if collect_num_simulations is not None and collect_num_simulations <= 0:
        raise ValueError(
            f'collect_num_simulations must be positive, got {collect_num_simulations}'
        )

    # game_segment_length=20 makes only 20-(num_unroll_steps+td_steps)=5 of the 20 positions in each
    # non-terminal segment eligible as sampling roots (valid_len in game_buffer._push_game_segment),
    # i.e. 75% of the collected transitions can never be trained on. 200 restores ~92.5% coverage.
    game_segment_length = 200 if game_segment_length_override is None else int(game_segment_length_override)
    save_ckpt_after_iter = (
        50000 if save_ckpt_after_iter_override is None else int(save_ckpt_after_iter_override)
    )
    if save_ckpt_after_iter <= 0:
        raise ValueError(f'save_ckpt_after_iter must be positive, got {save_ckpt_after_iter}')
    periodic_ckpt_keep_last = (
        0 if periodic_ckpt_keep_last_override is None else int(periodic_ckpt_keep_last_override)
    )
    if periodic_ckpt_keep_last < 0:
        raise ValueError(
            f'periodic_ckpt_keep_last must be non-negative, got {periodic_ckpt_keep_last}'
        )
    resume_buffer_min_transitions = (
        10000 if resume_buffer_min_transitions_override is None
        else int(resume_buffer_min_transitions_override)
    )
    if resume_buffer_min_transitions < 0:
        raise ValueError(
            'resume_buffer_min_transitions must be non-negative, got '
            f'{resume_buffer_min_transitions}'
        )
    open_loop_diagnostic_freq = (
        0 if open_loop_diagnostic_freq_override is None
        else int(open_loop_diagnostic_freq_override)
    )
    if open_loop_diagnostic_freq < 0:
        raise ValueError(
            f'open_loop_diagnostic_freq must be non-negative, got {open_loop_diagnostic_freq}'
        )
    if legacy_resume_alpha is not None and legacy_resume_alpha <= 0:
        raise ValueError(f'legacy_resume_alpha must be positive, got {legacy_resume_alpha}')
    num_unroll_steps = 10

    policy_experiment_overrides, world_model_experiment_overrides = (
        _experimental_config_overrides(
            infer_context_length=infer_context_length_override,
            bootstrap_value_context=bootstrap_value_context,
            open_loop_consistency_weight=open_loop_consistency_weight_override,
            open_loop_recurrent_weight=open_loop_recurrent_weight_override,
            open_loop_consistency_batch_size=open_loop_consistency_batch_size_override,
            open_loop_consistency_horizon=open_loop_consistency_horizon_override,
            open_loop_prefix_transitions=open_loop_prefix_transitions_override,
            encoder_clip_enabled=encoder_clip_enabled,
        )
    )

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
    buffer_reanalyze_freq = (
        1 / 5000000000
        if buffer_reanalyze_freq_override is None
        else float(buffer_reanalyze_freq_override)
    )
    if buffer_reanalyze_freq <= 0:
        raise ValueError(
            f'buffer_reanalyze_freq must be positive, got {buffer_reanalyze_freq}'
        )
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
                    exact_kv_window_reset=exact_kv_window_reset,
                    rebuild_kv_window_from_tokens=rebuild_kv_window_from_tokens,
                    open_loop_diagnostic_freq=open_loop_diagnostic_freq,
                    open_loop_diagnostic_batch_size=collector_env_num,
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
            contextual_reanalysis=contextual_reanalysis,
            # Full learner checkpoints are ~530MB each; save every 50k train iters instead of
            # the default 10k to bound disk usage. ckpt_best (on new best eval) is unaffected.
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=save_ckpt_after_iter))),
            periodic_ckpt_keep_last=periodic_ckpt_keep_last,
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
            # NOTE: model.world_model_cfg.use_priority intentionally remains True so the model
            # always returns a shape-[B] value_priority diagnostic. Setting policy.use_priority
            # False here still gives uniform replay/IS weights; the buffer discards priority
            # write-backs. This is deliberate model-vs-buffer asymmetry, not a synchronized flag.
            use_priority=True if use_priority is None else use_priority,
            priority_prob_alpha=0.6,
            priority_prob_beta=0.4,

            # Reanalyze settings
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            # A refresh expands each sequence into H+1 roots. Keep every MCTS group within the
            # same recurrent-KV capacity used by the online collector (8 envs x 50 simulations).
            reanalyze_search_chunk_size=collector_env_num,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,

            # Environment settings
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            eval_freq=int(5e3),
            replay_buffer_size=int(5e5),
            # Policy checkpoints omit the ~25GB full Atari replay buffer. Refill a small diverse
            # on-policy window before updating a mature model after preemption.
            resume_buffer_min_transitions=resume_buffer_min_transitions,
        ),
    )
    atari_unizero_config['policy'].update(policy_experiment_overrides)
    atari_unizero_config['policy']['model']['world_model_cfg'].update(
        world_model_experiment_overrides
    )
    if collect_num_simulations is not None:
        atari_unizero_config['policy']['collect_num_simulations'] = collect_num_simulations
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
    _prepare_run_directory(run_dir, resume_from=resume_from, resume_in_place=resume_in_place)
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
    encoder_clip_group = parser.add_mutually_exclusive_group()
    encoder_clip_group.add_argument(
        '--enable-encoder-clip', dest='encoder_clip_enabled', action='store_true',
        help='Opt in to the encoder latent-norm projection with the policy default 30->10 schedule.'
    )
    encoder_clip_group.add_argument(
        '--disable-encoder-clip', dest='encoder_clip_enabled', action='store_false',
        help='Explicitly disable encoder latent-norm projection (also the policy default).'
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
    parser.set_defaults(
        disable_adaptive_alpha=True,
        disable_policy_label_smoothing=True,
        encoder_clip_enabled=None,
    )
    parser.add_argument(
        '--resume-from', dest='resume_from', type=str, default=None,
        help='Optional learner checkpoint path to resume weights/optimizer/train_iter/envstep from.'
    )
    parser.add_argument(
        '--resume-in-place', action='store_true',
        help='Reopen an existing run directory when resuming from a checkpoint. This is intended '
             'for infrastructure-level automatic restarts.'
    )
    parser.add_argument(
        '--max-env-step', dest='max_env_step', type=int, default=None,
        help='Override the default max env-step budget (e.g. for continuing a run past its cap).'
    )
    parser.add_argument(
        '--evaluator-env-num', dest='evaluator_env_num', type=int, default=None,
        help='Number of deterministic evaluation environments/episodes (default 3).'
    )
    parser.add_argument(
        '--collect-num-simulations', dest='collect_num_simulations', type=int, default=None,
        help='Override MCTS simulations per action during collection (policy default 25).'
    )
    parser.add_argument(
        '--resume-buffer-min-transitions', dest='resume_buffer_min_transitions',
        type=int, default=None,
        help='Collect at least this many replay transitions before updating a resumed mature '
             'checkpoint; default 10000. Fresh runs still use the normal one-batch warmup.'
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
        '--infer-context-length', dest='infer_context_length', type=int, default=None,
        help='Override the number of observation/action blocks retained by online KV inference (default 4).'
    )
    kv_window_group = parser.add_mutually_exclusive_group()
    kv_window_group.add_argument(
        '--exact-kv-window-reset', action='store_true',
        help='Hard-reset online KV context to its latest latent observation instead of applying an '
             'inexact algebraic positional shift (diagnostic mode).'
    )
    parser.add_argument(
        '--bootstrap-value-context', action='store_true',
        help='Compute TD bootstrap values from the same exact rolling replay history available '
             'to online UniZero planning instead of the longer training-only sequence context.'
    )
    parser.add_argument(
        '--contextual-reanalysis', action='store_true',
        help='Opt in to rebuilding replay MCTS root priors and KV caches from the same short '
             'observation/action history used by online planning. Legacy reanalysis remains the default.'
    )
    kv_window_group.add_argument(
        '--rebuild-kv-window-from-tokens', action='store_true',
        help=(
            'Exactly rebuild a full learned-absolute-position KV window from retained raw '
            'observation/action embeddings whenever the window advances.'
        ),
    )
    parser.add_argument(
        '--buffer-reanalyze-freq', type=float, default=None,
        help='Override periodic replay-buffer policy-target reanalysis frequency '
             '(e.g. 0.02 means once every 50 collect/train epochs).'
    )
    parser.add_argument(
        '--save-ckpt-after-iter', dest='save_ckpt_after_iter', type=int, default=None,
        help='Override periodic learner-checkpoint interval; useful on preemptible clusters.'
    )
    parser.add_argument(
        '--periodic-ckpt-keep-last', dest='periodic_ckpt_keep_last', type=int, default=None,
        help='Keep iteration_0 plus this many newest periodic learner checkpoints; '
             '0/default disables pruning and never affects ckpt_best.'
    )
    parser.add_argument(
        '--open-loop-diagnostic-freq', dest='open_loop_diagnostic_freq', type=int, default=None,
        help='Measure detached MCTS-style autoregressive latent rollout error every N learner '
             'iterations; 0/default disables the diagnostic.'
    )
    parser.add_argument(
        '--open-loop-consistency-weight', dest='open_loop_consistency_weight', type=float, default=None,
        help='Weight for the short differentiable MCTS-style latent rollout consistency loss; '
             '0/default disables it.'
    )
    parser.add_argument(
        '--open-loop-recurrent-weight', dest='open_loop_recurrent_weight', type=float, default=None,
        help='Weight for MuZero-style latent/reward/value/policy supervision on recursively '
             'predicted states; 0/default disables it and it is mutually exclusive with '
             '--open-loop-consistency-weight.'
    )
    parser.add_argument(
        '--open-loop-consistency-batch-size', dest='open_loop_consistency_batch_size', type=int, default=None,
        help='Number of replay samples used by the optional open-loop consistency loss.'
    )
    parser.add_argument(
        '--open-loop-consistency-horizon', dest='open_loop_consistency_horizon', type=int, default=None,
        help='Number of predicted-latent transitions in the optional consistency rollout.'
    )
    parser.add_argument(
        '--open-loop-prefix-transitions', dest='open_loop_prefix_transitions', type=int, default=None,
        help='Number of real replay transitions used as a teacher prefix before the optional '
             'open-loop rollout; 0/default starts from a single root observation.'
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
        evaluator_env_num_override=args.evaluator_env_num,
        collect_num_simulations_override=args.collect_num_simulations,
        disable_adaptive_alpha=args.disable_adaptive_alpha,
        fixed_alpha=args.fixed_alpha,
        disable_policy_label_smoothing=args.disable_policy_label_smoothing,
        encoder_clip_enabled=args.encoder_clip_enabled,
        resume_from=args.resume_from,
        resume_in_place=args.resume_in_place,
        max_env_step_override=args.max_env_step,
        use_priority=args.use_priority,
        stab_fix=args.stab_fix,
        game_segment_length_override=args.game_segment_length,
        infer_context_length_override=args.infer_context_length,
        exact_kv_window_reset=args.exact_kv_window_reset,
        rebuild_kv_window_from_tokens=args.rebuild_kv_window_from_tokens,
        contextual_reanalysis=args.contextual_reanalysis,
        bootstrap_value_context=args.bootstrap_value_context,
        resume_buffer_min_transitions_override=args.resume_buffer_min_transitions,
        buffer_reanalyze_freq_override=args.buffer_reanalyze_freq,
        save_ckpt_after_iter_override=args.save_ckpt_after_iter,
        periodic_ckpt_keep_last_override=args.periodic_ckpt_keep_last,
        open_loop_diagnostic_freq_override=args.open_loop_diagnostic_freq,
        open_loop_consistency_weight_override=args.open_loop_consistency_weight,
        open_loop_recurrent_weight_override=args.open_loop_recurrent_weight,
        open_loop_consistency_batch_size_override=args.open_loop_consistency_batch_size,
        open_loop_consistency_horizon_override=args.open_loop_consistency_horizon,
        open_loop_prefix_transitions_override=args.open_loop_prefix_transitions,
        legacy_resume_alpha=args.legacy_resume_alpha,
    )
