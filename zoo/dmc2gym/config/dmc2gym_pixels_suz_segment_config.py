from easydict import EasyDict
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================

from zoo.dmc2gym.config.dmc_state_env_space_map import dmc_state_env_action_space_map, dmc_state_env_obs_space_map

def main(env_id, seed):
    action_space_size = dmc_state_env_action_space_map[env_id]
    obs_space_size = dmc_state_env_obs_space_map[env_id]
    print(f'env_id: {env_id}, action_space_size: {action_space_size}, obs_space_size: {obs_space_size}')

    domain_name = env_id.split('-')[0]
    task_name = env_id.split('-')[1]

    continuous_action_space = True
    K = 20  # num_of_sampled_actions
    collector_env_num = 8
    n_episode = 8
    num_segments = 8
    game_segment_length = 100
    evaluator_env_num = 3
    num_simulations = 50
    replay_ratio = 0.1
    max_env_step = int(5e5)
    batch_size = 64
    num_layers = 2
    num_unroll_steps = 5
    infer_context_length = 2
    norm_type = 'LN'

    # Defines the frequency of reanalysis. E.g., 1 means reanalyze once per epoch, 2 means reanalyze once every two epochs.
    buffer_reanalyze_freq = 1/100000
    # Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
    reanalyze_batch_size = 160
    # The partition of reanalyze. E.g., 1 means reanalyze_batch samples from the whole buffer, 0.5 means samples from the first half of the buffer.
    reanalyze_partition=0.75

    # for debug
    # collector_env_num = 2
    # num_segments = 2
    # n_episode = 2
    # evaluator_env_num = 2
    # num_simulations = 3
    # batch_size = 3
    # reanalyze_batch_size = 1
    # ==============================================================
    # end of the most frequently changed config specified by the user
    # ==============================================================

    dmc2gym_pixels_cont_sampled_unizero_config = dict(
        env=dict(
            env_id='dmc2gym-v0',
            domain_name=domain_name,
            task_name=task_name,
            from_pixels=True,  # pixel/image obs
            frame_skip=2,
            continuous=True,
            save_replay_gif=False,
            replay_path_gif='./replay_gif',
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
            # TODO: only for debug
            # collect_max_episode_steps=int(20),
            # eval_max_episode_steps=int(20),
        ),
        policy=dict(
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000000,),),),  # default is 10000
            model=dict(
                observation_shape=(3, 84, 84),
                action_space_size=action_space_size,
                continuous_action_space=continuous_action_space,
                num_of_sampled_actions=K,
                model_type='conv',
                world_model_cfg=dict(
                    policy_loss_type='kl',
                    obs_type='image',
                    num_unroll_steps=num_unroll_steps,
                    policy_entropy_weight=5e-2,
                    continuous_action_space=continuous_action_space,
                    num_of_sampled_actions=K,
                    sigma_type='conditioned',
                    fixed_sigma_value=0.5,
                    bound_type=None,
                    model_type='conv',
                    norm_type=norm_type,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,  # NOTE: each timestep has 2 tokens: obs and action
                    context_length=2 * infer_context_length,
                    device='cuda',
                    action_space_size=action_space_size,
                    num_layers=num_layers,
                    num_heads=8,
                    embed_dim=768,
                    env_num=max(collector_env_num, evaluator_env_num),
                ),
            ),
            # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            cuda=True,
            use_root_value=False,
            use_augmentation=False,
            use_priority=False,
            env_type='not_board_games',
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            discount_factor=0.99,
            td_steps=5,
            piecewise_decay_lr_scheduler=False,
            learning_rate=1e-4,
            grad_clip_value=5,
            manual_temperature_decay=True,
            threshold_training_steps_for_final_temperature=int(2.5e4),
            cos_lr_scheduler=True,
            num_segments=num_segments,
            train_start_after_envsteps=2000,
            game_segment_length=game_segment_length,
            num_simulations=num_simulations,
            reanalyze_ratio=0,
            n_episode=n_episode,
            eval_freq=int(5e3),
            replay_buffer_size=int(1e6),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            # ============= The key different params for ReZero =============
            buffer_reanalyze_freq=buffer_reanalyze_freq, # 1 means reanalyze one times per epoch, 2 means reanalyze one times each two epoch
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
        ),
    )

    dmc2gym_pixels_cont_sampled_unizero_config = EasyDict(dmc2gym_pixels_cont_sampled_unizero_config)
    main_config = dmc2gym_pixels_cont_sampled_unizero_config

    dmc2gym_pixels_cont_sampled_unizero_create_config = dict(
        env=dict(
            type='dmc2gym_lightzero',
            import_names=['zoo.dmc2gym.envs.dmc2gym_lightzero_env'],
        ),
        # env_manager=dict(type='subprocess'),
        env_manager=dict(type='base'),
        policy=dict(
            type='sampled_unizero',
            import_names=['lzero.policy.sampled_unizero'],
        ),
    )
    dmc2gym_pixels_cont_sampled_unizero_create_config = EasyDict(dmc2gym_pixels_cont_sampled_unizero_create_config)
    create_config = dmc2gym_pixels_cont_sampled_unizero_create_config

    # ============ use muzero_segment_collector instead of muzero_collector =============
    from lzero.entry import train_unizero_segment
    main_config.exp_name=f'data_sampled_unizero/dmc2gym_{env_id}_brf{buffer_reanalyze_freq}_image_cont_suz_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_K{K}_ns{num_simulations}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_{norm_type}_seed{seed}_learnsigma'
    train_unizero_segment([main_config, create_config], model_path=main_config.policy.model_path, seed=seed, max_env_step=max_env_step)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some environment.')
    
    parser.add_argument('--env', type=str, help='The environment to use', default='cartpole-swingup')
    parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    args = parser.parse_args()

    main(args.env, args.seed)