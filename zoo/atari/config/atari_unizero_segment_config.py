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
    """Mirror console output to both the terminal and the run directory."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, 'isatty', lambda: False)() for stream in self.streams)

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


def main(env_id, seed, output_root='data_unizero/rjob', run_name=None, use_new_cache_manager=False):
    action_space_size = atari_env_action_space_map[env_id]

    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    collector_env_num = 8
    num_segments = 8
    evaluator_env_num = 3

    game_segment_length = 20
    num_unroll_steps = 10
    infer_context_length = 4

    num_simulations = 50
    # Reproduce the strong 2025-10-10 Pong baseline.
    batch_size = 256
    replay_ratio = 0.1

    num_layers = 2
    norm_type = "LN"

    if env_id == 'ALE/Pong-v5':
        # max_env_step = int(5e5)
        max_env_step = int(10e6)
    else:
        max_env_step = int(10e6)

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
                num_res_blocks=2,
                num_channels=128,
                world_model_cfg=dict(
                    latent_recon_loss_weight=0.0,
                    perceptual_loss_weight=0.0,
                    norm_type=norm_type,
                    final_norm_option_in_obs_head='LayerNorm',
                    final_norm_option_in_encoder='LayerNorm',
                    predict_latent_loss_type='mse',
                    support_size=601,
                    policy_entropy_weight=5e-3,
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
            num_simulations=num_simulations,
            grad_clip_value=5,
            use_augmentation=False,

            # Adaptive target entropy settings from the 2025 Pong run.
            use_adaptive_entropy_weight=True,
            adaptive_entropy_alpha_lr=1e-4,
            target_entropy_start_ratio=0.98,
            target_entropy_end_ratio=0.7,
            target_entropy_decay_steps=100000,

            # The old run name mentioned 30->10 encoder clipping, but the
            # implementation did not apply it. Keep it explicitly disabled.
            use_encoder_clip_annealing=False,
            encoder_clip_anneal_type='cosine',
            encoder_clip_start_value=30.0,
            encoder_clip_end_value=10.0,
            encoder_clip_anneal_steps=100000,
            latent_norm_clip_threshold=0.0,

            # Policy smoothing decays 0.05->0.01; value/reward use 0.1.
            policy_ls_eps_start=0.05,
            policy_ls_eps_end=0.01,
            policy_ls_eps_decay_steps=50000,
            label_smoothing_eps=0.1,
            use_continuous_label_smoothing=False,
            monitor_norm_freq=10000,

            # Priority settings
            use_priority=True,
            priority_prob_alpha=1,
            priority_prob_beta=1,

            # Reanalyze settings
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,

            # Environment settings
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            eval_freq=int(5e3),
            replay_buffer_size=int(5e5),
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
        run_name = f'{game_name.lower()}_sync_2025baseline_seed{seed}_{timestamp}'
    run_name = _safe_run_name(run_name)

    # LightZero internally prefixes exp_name with "./", so keep it relative to
    # the current working directory even when an absolute output root is given.
    run_dir = os.path.relpath(os.path.abspath(os.path.join(output_root, run_name)), os.getcwd())
    if os.path.exists(run_dir):
        raise FileExistsError(f'Run directory already exists: {os.path.abspath(run_dir)}')
    os.makedirs(run_dir)
    main_config.exp_name = run_dir

    metadata = {
        'run_name': run_name,
        'run_dir': os.path.abspath(run_dir),
        'started_at': datetime.now().astimezone().isoformat(),
        'hostname': socket.gethostname(),
        'pid': os.getpid(),
        'command': sys.argv,
        'config_file': os.path.abspath(__file__),
        'env_id': env_id,
        'seed': seed,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'variant': 'sync_2025baseline_encoder_clip_off',
        'key_hyperparameters': {
            'batch_size': batch_size,
            'replay_ratio': replay_ratio,
            'target_entropy_ratio': [0.98, 0.7],
            'target_entropy_decay_steps': 100000,
            'encoder_clip_enabled': False,
            'use_new_cache_manager': use_new_cache_manager,
            'buffer_reanalyze_freq': buffer_reanalyze_freq,
            'reanalyze_batch_size': reanalyze_batch_size,
            'reanalyze_partition': reanalyze_partition,
        },
    }
    with open(os.path.join(run_dir, 'run_metadata.json'), 'w', encoding='utf-8') as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
        file.write('\n')
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
                [main_config, create_config], seed=seed, model_path=None, max_env_step=max_env_step
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
        '--output-root', type=str, default='data_unizero/rjob',
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
    args = parser.parse_args()

    main(
        args.env,
        args.seed,
        output_root=args.output_root,
        run_name=args.run_name,
        use_new_cache_manager=args.use_new_cache_manager,
    )
