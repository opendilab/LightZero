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

    if env_id == 'ALE/Pong-v5':
        max_env_step = int(5e5)
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
                    last_linear_layer_init_zero=True,
                    zero_init_head_names=zero_init_head_names,
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
    main_config.exp_name = '/'.join(exp_parts)

    train_unizero_segment([main_config, create_config], seed=seed, model_path=None, max_env_step=max_env_step)


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
    )
