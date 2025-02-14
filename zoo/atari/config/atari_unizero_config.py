from easydict import EasyDict

from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map


def main(env_id='PongNoFrameskip-v4', seed=0):
    action_space_size = atari_env_action_space_map[env_id]

    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    collector_env_num = 8
    game_segment_length = 20
    evaluator_env_num = 3
    num_simulations = 50
    max_env_step = int(4e5)
    batch_size = 64
    num_unroll_steps = 10
    infer_context_length = 4
    num_layers = 2
    replay_ratio = 0.25

    # only for debug
    collector_env_num = 2
    game_segment_length = 20
    evaluator_env_num = 1
    num_simulations = 2
    max_env_step = int(5e5)
    batch_size = 2
    num_unroll_steps = 5
    infer_context_length = 2
    num_layers = 1
    replay_ratio = 0.1
    # ==============================================================
    # end of the most frequently changed config specified by the user
    # ==============================================================
    atari_unizero_config = dict(
        env=dict(
            stop_value=int(1e6),
            env_id=env_id,
            observation_shape=(3, 96, 96),
            gray_scale=False,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
            # TODO: only for debug
            collect_max_episode_steps=int(50),
            eval_max_episode_steps=int(50),
        ),
        policy=dict(
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000000, ), ), ),  # default is 10000
            model=dict(
                observation_shape=(3, 96, 96),
                action_space_size=action_space_size,
                world_model_cfg=dict(
                    policy_entropy_weight=1e-4,
                    continuous_action_space=False,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,  # NOTE: each timestep has 2 tokens: obs and action
                    context_length=2 * infer_context_length,
                    device='cuda',
                    action_space_size=action_space_size,
                    num_layers=num_layers,
                    num_heads=8,
                    embed_dim=768,
                    obs_type='image',
                    env_num=max(collector_env_num, evaluator_env_num),
                    # for RoPE
                    # rotary_emb=False,
                    rotary_emb=True,
                    rope_theta=10000,
                    # max_seq_len=2048,
                    # max_seq_len=4096,
                    max_seq_len=int(4096*2),

                ),
            ),
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            learning_rate=0.0001,
            num_simulations=num_simulations,
            # train_start_after_envsteps=2000,
            train_start_after_envsteps=0, # debug
            game_segment_length=game_segment_length,
            replay_buffer_size=int(1e6),
            eval_freq=int(5e3),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
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

    # main_config.exp_name = f'data_unizero_20250211/{env_id[:-14]}/{env_id[:-14]}_uz_posembed-mergemain_nlayer{num_layers}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'
    # main_config.exp_name = f'data_unizero_20250211/{env_id[:-14]}/{env_id[:-14]}_uz_rope-mergemain-msl4096*2_nlayer{num_layers}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'
    main_config.exp_name = f'data_unizero_20250211_debug/{env_id[:-14]}/{env_id[:-14]}_uz_rope-mergemain-msl4096*2_nlayer{num_layers}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'
    
    from lzero.entry import train_unizero
    train_unizero([main_config, create_config], seed=seed, model_path=main_config.policy.model_path, max_env_step=max_env_step)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some environment.')
    parser.add_argument('--env', type=str, help='The environment to use', default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    args = parser.parse_args()

    main(args.env, args.seed)

