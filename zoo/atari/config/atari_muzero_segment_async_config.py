from easydict import EasyDict
from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map


def _atari_game_name(env_id):
    game_name = env_id.split('/')[-1].split('-')[0]
    for suffix in ('NoFrameskip', 'Deterministic'):
        if game_name.endswith(suffix):
            game_name = game_name[:-len(suffix)]
    return game_name


def main(
        env_id,
        seed,
        exp_root='data_lz_muzero',
        run_tag=None,
        max_env_step_override=None,
        async_pipeline=False,
        num_collector_actors=1,
        max_policy_lag=0,
        max_train_chunk_steps=4,
        weight_sync_interval=1,
        collector_num_gpus=0,
        evaluator_num_gpus=0,
        smoke_test=False,
):
    if env_id not in atari_env_action_space_map:
        supported_envs = ', '.join(sorted(atari_env_action_space_map.keys()))
        raise KeyError(f"Unsupported Atari env_id: {env_id}. Supported envs: {supported_envs}")

    game_name = _atari_game_name(env_id)
    action_space_size = atari_env_action_space_map[env_id]

    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    collector_env_num = 8
    num_segments = 8
    game_segment_length = 20

    evaluator_env_num = 3
    num_simulations = 50
    eval_freq = int(5e3)
    update_per_collect = None
    replay_ratio = 0.25

    num_unroll_steps = 5
    batch_size = 256
    max_env_step = int(5e5)
    if max_env_step_override is not None:
        max_env_step = int(max_env_step_override)

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
    if smoke_test:
        collector_env_num = 1
        num_segments = 1
        game_segment_length = 16
        evaluator_env_num = 1
        num_simulations = 2
        eval_freq = 1
        update_per_collect = 1
        replay_ratio = 1.0
        batch_size = 2
        max_env_step = min(max_env_step, 40)
    # ==============================================================
    # end of the most frequently changed config specified by the user
    # ==============================================================

    env_config = dict(
        stop_value=int(1e6),
        env_id=env_id,
        observation_shape=(4, 64, 64),
        frame_stack_num=4,
        gray_scale=True,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    )
    if smoke_test:
        env_config.update(
            collect_max_episode_steps=int(30),
            eval_max_episode_steps=int(30),
        )

    atari_muzero_config = dict(
        env=env_config,
        policy=dict(
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000000, ), ), ),  # default is 10000
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
            eval_freq=eval_freq,
            replay_buffer_size=int(1e6),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            # ============= The key different params for reanalyze =============
            # Defines the frequency of reanalysis. E.g., 1 means reanalyze once per epoch, 2 means reanalyze once every two epochs.
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            # Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
            reanalyze_batch_size=reanalyze_batch_size,
            # The partition of reanalyze. E.g., 1 means reanalyze_batch samples from the whole buffer, 0.5 means samples from the first half of the buffer.
            reanalyze_partition=reanalyze_partition,
            async_pipeline=dict(
                enabled=async_pipeline,
                num_collector_actors=num_collector_actors,
                num_evaluator_actors=1,
                max_collect_inflight=num_collector_actors,
                max_eval_inflight=1,
                max_train_chunk_steps=max_train_chunk_steps,
                weight_sync_interval=weight_sync_interval,
                max_policy_lag=max_policy_lag,
                collector_num_cpus=1,
                evaluator_num_cpus=1,
                collector_num_gpus=collector_num_gpus,
                evaluator_num_gpus=evaluator_num_gpus,
                poll_interval_s=0.1,
                shutdown_timeout_s=30,
            ),
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
    if async_pipeline:
        from lzero.entry.train_muzero_segment_async import train_muzero_segment_async as train_entry
    else:
        from lzero.entry.train_muzero_segment import train_muzero_segment as train_entry

    exp_root = exp_root.rstrip('/')
    exp_parts = [exp_root]
    if run_tag:
        exp_parts.append(run_tag)
    run_name = f'{game_name}_mz_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}_bs{batch_size}_seed{seed}'
    if async_pipeline:
        run_name += f'_async-c{num_collector_actors}-chunk{max_train_chunk_steps}-lag{max_policy_lag}'
    if smoke_test:
        run_name += '_smoke'
    exp_parts.extend([
        game_name,
        f'seed{seed}',
        run_name
    ])
    main_config.exp_name = '/'.join(exp_parts)
    train_entry([main_config, create_config], seed=seed, max_env_step=max_env_step)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process different environments and seeds.')
    parser.add_argument('--env', type=str, help='The environment to use', default='ALE/Pong-v5')
    parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    parser.add_argument('--exp-root', type=str, help='Experiment root directory', default='data_lz_muzero')
    parser.add_argument('--run-tag', type=str, help='Optional grouped run tag for rjob/tensorboard layout', default=None)
    parser.add_argument('--max-env-step', type=int, help='Override max env steps for smoke/debug runs', default=None)
    parser.add_argument('--async-pipeline', action='store_true', help='Enable Ray async collector/evaluator pipeline')
    parser.add_argument('--num-collector-actors', type=int, default=1, help='Number of Ray collector actors')
    parser.add_argument('--max-policy-lag', type=int, default=0, help='Allowed collector policy version lag')
    parser.add_argument('--max-train-chunk-steps', type=int, default=4, help='Max learner updates before yielding to async tasks')
    parser.add_argument('--weight-sync-interval', type=int, default=1, help='Learner steps between collect/eval weight publishes')
    parser.add_argument('--collector-num-gpus', type=float, default=0, help='Ray GPU resource per collector actor')
    parser.add_argument('--evaluator-num-gpus', type=float, default=0, help='Ray GPU resource per evaluator actor')
    parser.add_argument('--smoke-test', action='store_true', help='Use a tiny config for startup/rjob smoke validation')
    args = parser.parse_args()

    main(
        args.env,
        args.seed,
        exp_root=args.exp_root,
        run_tag=args.run_tag,
        max_env_step_override=args.max_env_step,
        async_pipeline=args.async_pipeline,
        num_collector_actors=args.num_collector_actors,
        max_policy_lag=args.max_policy_lag,
        max_train_chunk_steps=args.max_train_chunk_steps,
        weight_sync_interval=args.weight_sync_interval,
        collector_num_gpus=args.collector_num_gpus,
        evaluator_num_gpus=args.evaluator_num_gpus,
        smoke_test=args.smoke_test,
    )
