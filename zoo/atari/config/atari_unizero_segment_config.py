from easydict import EasyDict
from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map


def _atari_game_name(env_id):
    game_name = env_id.split('/')[-1].split('-')[0]
    for suffix in ('NoFrameskip', 'Deterministic'):
        if game_name.endswith(suffix):
            game_name = game_name[:-len(suffix)]
    return game_name


def _str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if normalized in ('0', 'false', 'no', 'n', 'off'):
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _parse_zero_init_head_names(value):
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in ('all', 'legacy'):
        return None
    if normalized in ('none', 'off', 'false'):
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def main(
        env_id,
        seed,
        exp_root='data_unizero',
        run_tag=None,
        max_env_step_override=None,
        baseline_name=None,
        latent_recon_loss_weight_override=None,
        perceptual_loss_weight_override=None,
        torch_compile_override=None,
        empty_cuda_cache_on_cache_reset_override=None,
        zero_init_head_names_override=None,
        use_new_cache_manager_override=None,
        save_ckpt_override=None,
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
    evaluator_env_num = 3

    game_segment_length = 20
    num_unroll_steps = 10
    infer_context_length = 4

    num_simulations = 50
    batch_size = 128
    replay_ratio = 0.25

    num_layers = 2
    norm_type = "LN"
    latent_recon_loss_weight = 0.0
    perceptual_loss_weight = 0.0
    torch_compile = False
    empty_cuda_cache_on_cache_reset = False
    zero_init_head_names = ['value', 'reward']
    use_new_cache_manager = True
    save_ckpt = True

    if smoke_test:
        collector_env_num = 1
        num_segments = 1
        evaluator_env_num = 1
        game_segment_length = 16
        num_unroll_steps = 2
        infer_context_length = 1
        num_simulations = 2
        batch_size = 2
        replay_ratio = 1.0
        save_ckpt = False

    if latent_recon_loss_weight_override is not None:
        latent_recon_loss_weight = float(latent_recon_loss_weight_override)
    if perceptual_loss_weight_override is not None:
        perceptual_loss_weight = float(perceptual_loss_weight_override)
    if torch_compile_override is not None:
        torch_compile = bool(torch_compile_override)
    if empty_cuda_cache_on_cache_reset_override is not None:
        empty_cuda_cache_on_cache_reset = bool(empty_cuda_cache_on_cache_reset_override)
    if zero_init_head_names_override is not None:
        zero_init_head_names = _parse_zero_init_head_names(zero_init_head_names_override)
    if use_new_cache_manager_override is not None:
        use_new_cache_manager = bool(use_new_cache_manager_override)
    if save_ckpt_override is not None:
        save_ckpt = bool(save_ckpt_override)

    if env_id == 'ALE/Pong-v5':
        max_env_step = int(5e5)
    else:
        max_env_step = int(10e6)
    if max_env_step_override is not None:
        max_env_step = int(max_env_step_override)
    if smoke_test:
        max_env_step = min(max_env_step, 40)

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
                    latent_recon_loss_weight=latent_recon_loss_weight,
                    perceptual_loss_weight=perceptual_loss_weight,
                    norm_type=norm_type,
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
                    use_new_cache_manager=use_new_cache_manager,
                    last_linear_layer_init_zero=True,
                    zero_init_head_names=zero_init_head_names,
                ),
            ),
            save_ckpt_in_eval=save_ckpt,
            learn=dict(
                learner=dict(
                    hook=dict(
                        save_ckpt_after_iter=int(1e4) if save_ckpt else int(1e12),
                    ),
                ),
            ),
            # Learning settings
            learning_rate=0.0001,
            weight_decay=1e-2,
            batch_size=batch_size,
            replay_ratio=replay_ratio,
            num_unroll_steps=num_unroll_steps,
            num_segments=num_segments,
            game_segment_length=game_segment_length,
            num_simulations=num_simulations,

            # Priority settings
            use_priority=True,
            priority_prob_alpha=1,
            priority_prob_beta=1,
            torch_compile=torch_compile,
            empty_cuda_cache_on_cache_reset=empty_cuda_cache_on_cache_reset,

            # Reanalyze settings
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,

            # Environment settings
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            eval_freq=int(1e4),
            replay_buffer_size=int(5e5),
            async_pipeline=dict(
                enabled=async_pipeline,
                num_collector_actors=num_collector_actors,
                num_evaluator_actors=1,
                max_collect_inflight=num_collector_actors,
                max_eval_inflight=1,
                max_train_chunk_steps=max_train_chunk_steps,
                weight_sync_interval=weight_sync_interval,
                max_policy_lag=max_policy_lag,
                eval_at_start=True,
                collector_num_cpus=1,
                evaluator_num_cpus=1,
                collector_num_gpus=collector_num_gpus,
                evaluator_num_gpus=evaluator_num_gpus,
                buffer_stats_interval=100,
                poll_interval_s=0.1,
                shutdown_timeout_s=30,
            ),
        ),
    )
    if smoke_test:
        atari_unizero_config['env'].update(
            collect_max_episode_steps=int(30),
            eval_max_episode_steps=int(30),
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
    if async_pipeline:
        from lzero.entry.train_unizero_segment_async import train_unizero_segment_async as train_entry
    else:
        from lzero.entry import train_unizero_segment as train_entry
    exp_root = exp_root.rstrip('/')
    exp_parts = [exp_root]
    if run_tag:
        exp_parts.append(run_tag)
    exp_parts.extend([
        game_name,
        f'seed{seed}',
    ])
    if baseline_name:
        exp_parts.append(str(baseline_name))
    exp_parts.append(
        f'{game_name}_uz_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'
    )
    if async_pipeline:
        exp_parts[-1] += f'_async-c{num_collector_actors}-chunk{max_train_chunk_steps}-lag{max_policy_lag}'
    if smoke_test:
        exp_parts[-1] += '_smoke'
    main_config.exp_name = '/'.join(exp_parts)

    train_entry([main_config, create_config], seed=seed, model_path=None, max_env_step=max_env_step)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process different environments and seeds.')
    parser.add_argument('--env', type=str, help='The environment to use', default='ALE/Pong-v5')
    parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    parser.add_argument('--exp-root', type=str, help='Experiment root directory', default='data_unizero')
    parser.add_argument('--run-tag', type=str, help='Optional grouped run tag for rjob/tensorboard layout', default=None)
    parser.add_argument('--max-env-step', type=int, help='Override max env steps for smoke/debug runs', default=None)
    parser.add_argument('--baseline-name', type=str, help='Optional baseline variant name for log layout', default=None)
    parser.add_argument('--latent-recon-loss-weight', type=float, help='Override latent reconstruction loss weight', default=None)
    parser.add_argument('--perceptual-loss-weight', type=float, help='Override perceptual loss weight', default=None)
    parser.add_argument('--torch-compile', type=_str2bool, nargs='?', const=True, help='Override torch.compile usage', default=None)
    parser.add_argument(
        '--empty-cuda-cache-on-cache-reset',
        type=_str2bool,
        nargs='?',
        const=True,
        help='Override torch.cuda.empty_cache() on cache reset',
        default=None
    )
    parser.add_argument(
        '--zero-init-head-names',
        type=str,
        help='Comma separated heads to zero-init, or all/legacy/none',
        default=None
    )
    parser.add_argument('--use-new-cache-manager', type=_str2bool, nargs='?', const=True, help='Enable the structured KV cache manager', default=None)
    parser.add_argument('--save-ckpt', type=_str2bool, nargs='?', const=True, help='Save evaluator/best checkpoints', default=None)
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
        baseline_name=args.baseline_name,
        latent_recon_loss_weight_override=args.latent_recon_loss_weight,
        perceptual_loss_weight_override=args.perceptual_loss_weight,
        torch_compile_override=args.torch_compile,
        empty_cuda_cache_on_cache_reset_override=args.empty_cuda_cache_on_cache_reset,
        zero_init_head_names_override=args.zero_init_head_names,
        use_new_cache_manager_override=args.use_new_cache_manager,
        save_ckpt_override=args.save_ckpt,
        async_pipeline=args.async_pipeline,
        num_collector_actors=args.num_collector_actors,
        max_policy_lag=args.max_policy_lag,
        max_train_chunk_steps=args.max_train_chunk_steps,
        weight_sync_interval=args.weight_sync_interval,
        collector_num_gpus=args.collector_num_gpus,
        evaluator_num_gpus=args.evaluator_num_gpus,
        smoke_test=args.smoke_test,
    )
