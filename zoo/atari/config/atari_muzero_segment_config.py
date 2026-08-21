import os
import re

from easydict import EasyDict
from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map


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


def main(
        env_id,
        seed,
        output_root='data_lz_muzero',
        run_name=None,
        max_env_step_override=None,
        resume_from=None,
        resume_in_place=False,
        resume_buffer_min_transitions_override=None,
        save_ckpt_after_iter_override=None,
        periodic_ckpt_keep_last_override=None,
):
    action_space_size = atari_env_action_space_map[env_id]

    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    collector_env_num = 8
    num_segments = 8
    game_segment_length = 20

    evaluator_env_num = 3
    num_simulations = 50
    update_per_collect = None
    replay_ratio = 0.25

    num_unroll_steps = 5
    batch_size = 256
    max_env_step = int(5e5) if max_env_step_override is None else int(max_env_step_override)
    if max_env_step <= 0:
        raise ValueError(f'max_env_step must be positive, got {max_env_step}')
    resume_buffer_min_transitions = (
        10000 if resume_buffer_min_transitions_override is None
        else int(resume_buffer_min_transitions_override)
    )
    if resume_buffer_min_transitions < 0:
        raise ValueError(
            'resume_buffer_min_transitions must be non-negative, got '
            f'{resume_buffer_min_transitions}'
        )
    save_ckpt_after_iter = (
        1000000 if save_ckpt_after_iter_override is None else int(save_ckpt_after_iter_override)
    )
    if save_ckpt_after_iter <= 0:
        raise ValueError(f'save_ckpt_after_iter must be positive, got {save_ckpt_after_iter}')
    periodic_ckpt_keep_last = (
        0 if periodic_ckpt_keep_last_override is None else int(periodic_ckpt_keep_last_override)
    )
    if periodic_ckpt_keep_last < 0:
        raise ValueError(f'periodic_ckpt_keep_last must be non-negative, got {periodic_ckpt_keep_last}')

    # Defines the frequency of reanalysis. E.g., 1 means reanalyze once per epoch, 2 means reanalyze once every two epochs.
    # buffer_reanalyze_freq = 1/10
    buffer_reanalyze_freq = 1/10000
    # Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
    reanalyze_batch_size = 160
    # The partition of reanalyze. E.g., 1 means reanalyze_batch samples from the whole buffer, 0.5 means samples from the first half of the buffer.
    reanalyze_partition=1

    # =========== for debug ===========
    # collector_env_num = 2
    # num_segments = 2
    # evaluator_env_num = 2
    # num_simulations = 2
    # update_per_collect = 2
    # batch_size = 5
    # ==============================================================
    # end of the most frequently changed config specified by the user
    # ==============================================================

    atari_muzero_config = dict(
        env=dict(
            stop_value=int(1e6),
            env_id=env_id,
            observation_shape=(4, 64, 64),
            frame_stack_num=4,
            gray_scale=True,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
            # TODO: debug
            # collect_max_episode_steps=int(50),
            # eval_max_episode_steps=int(50),
        ),
        policy=dict(
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=save_ckpt_after_iter, ), ), ),
            analysis_sim_norm=False,
            cal_dormant_ratio=False,
            model=dict(
                observation_shape=(4, 64, 64),
                image_channel=1,
                frame_stack_num=4,
                gray_scale=True,
                action_space_size=action_space_size,
                downsample=True,
                self_supervised_learning_loss=True,  # default is False
                discrete_action_encoding_type='one_hot',
                norm_type='BN',
                use_sim_norm=True, # NOTE
                use_sim_norm_kl_loss=False,
                model_type='conv'
            ),
            cuda=True,
            env_type='not_board_games',
            num_segments=num_segments,
            train_start_after_envsteps=2000,
            game_segment_length=game_segment_length,
            random_collect_episode_num=0,
            use_augmentation=True,
            use_priority=False,
            replay_ratio=replay_ratio,
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            optim_type='SGD',
            td_steps=5,
            piecewise_decay_lr_scheduler=True,
            manual_temperature_decay=False,
            learning_rate=0.2,
            target_update_freq=100,
            num_simulations=num_simulations,
            ssl_loss_weight=2,
            policy_entropy_weight=5e-3,
            policy_label_smoothing=0.0,
            eval_freq=int(5e3),
            replay_buffer_size=int(1e6),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            resume_buffer_min_transitions=resume_buffer_min_transitions,
            periodic_ckpt_keep_last=periodic_ckpt_keep_last,
            # ============= The key different params for reanalyze =============
            # Defines the frequency of reanalysis. E.g., 1 means reanalyze once per epoch, 2 means reanalyze once every two epochs.
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            # Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
            reanalyze_batch_size=reanalyze_batch_size,
            # The partition of reanalyze. E.g., 1 means reanalyze_batch samples from the whole buffer, 0.5 means samples from the first half of the buffer.
            reanalyze_partition=reanalyze_partition,
        ),
    )
    atari_muzero_config = EasyDict(atari_muzero_config)
    main_config = atari_muzero_config

    atari_muzero_create_config = dict(
        env=dict(
            type='atari_lightzero',
            import_names=['zoo.atari.envs.atari_lightzero_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(
            type='muzero',
            import_names=['lzero.policy.muzero'],
        ),
    )
    atari_muzero_create_config = EasyDict(atari_muzero_create_config)
    create_config = atari_muzero_create_config

    # ============ use muzero_segment_collector instead of muzero_collector =============
    from lzero.entry import train_muzero_segment
    game_name = _atari_game_name(env_id)
    if run_name is None:
        run_name = (
            f'{game_name.lower()}_mz_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}'
            f'-rp{reanalyze_partition}_numsegments-{num_segments}_gsl{game_segment_length}'
            f'_rr{replay_ratio}_Htrain{num_unroll_steps}_bs{batch_size}_seed{seed}'
        )
    run_name = _safe_run_name(run_name)
    run_dir = os.path.relpath(os.path.abspath(os.path.join(output_root, run_name)), os.getcwd())
    _prepare_run_directory(run_dir, resume_from=resume_from, resume_in_place=resume_in_place)
    main_config.exp_name = run_dir
    train_muzero_segment(
        [main_config, create_config], seed=seed, model_path=resume_from, max_env_step=max_env_step
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process different environments and seeds.')
    parser.add_argument('--env', type=str, help='The environment to use', default='ALE/Pong-v5')
    parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    parser.add_argument(
        '--output-root', '--exp-root', dest='output_root', type=str, default='data_lz_muzero',
        help='Root directory containing one self-contained folder per run.'
    )
    parser.add_argument(
        '--run-name', '--run-tag', dest='run_name', type=str, default=None,
        help='Optional unique run folder name.'
    )
    parser.add_argument(
        '--max-env-step', dest='max_env_step', type=int, default=None,
        help='Override the default 500k environment-step budget.'
    )
    parser.add_argument('--resume-from', dest='resume_from', type=str, default=None)
    parser.add_argument('--resume-in-place', action='store_true')
    parser.add_argument(
        '--resume-buffer-min-transitions', dest='resume_buffer_min_transitions', type=int, default=None
    )
    parser.add_argument('--save-ckpt-after-iter', dest='save_ckpt_after_iter', type=int, default=None)
    parser.add_argument(
        '--periodic-ckpt-keep-last', dest='periodic_ckpt_keep_last', type=int, default=None
    )
    args = parser.parse_args()

    main(
        args.env,
        args.seed,
        output_root=args.output_root,
        run_name=args.run_name,
        max_env_step_override=args.max_env_step,
        resume_from=args.resume_from,
        resume_in_place=args.resume_in_place,
        resume_buffer_min_transitions_override=args.resume_buffer_min_transitions,
        save_ckpt_after_iter_override=args.save_ckpt_after_iter,
        periodic_ckpt_keep_last_override=args.periodic_ckpt_keep_last,
    )
