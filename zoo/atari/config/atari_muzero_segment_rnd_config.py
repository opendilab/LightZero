from easydict import EasyDict
from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map

def main(env_id, seed):
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
    replay_ratio = 0.1

    num_unroll_steps = 5
    batch_size = 256
    max_env_step = int(5e6)

    # Defines the frequency of reanalysis. E.g., 1 means reanalyze once per epoch, 2 means reanalyze once every two epochs.
    # buffer_reanalyze_freq = 1/10
    buffer_reanalyze_freq = 1/100000
    # Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
    reanalyze_batch_size = 160
    # The partition of reanalyze. E.g., 1 means reanalyze_batch samples from the whole buffer, 0.5 means samples from the first half of the buffer.
    reanalyze_partition=0.75
    use_rnd_model = True
    rnd_weights = 1.0
    use_intrinsic_weight_schedule = True
    intrinsic_weight_max = 0.2
    intrinsic_norm_type = 'return'
    observation_shape = (3, 96, 96)
    frame_stack_num = 1
    gray_scale = False
    image_channel = 3

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
            observation_shape=observation_shape,
            frame_stack_num=frame_stack_num,
            gray_scale=gray_scale,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
            # TODO: debug
            # collect_max_episode_steps=int(50),
            # eval_max_episode_steps=int(50),
        ),
        policy=dict(
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000000, ), ), ),  # default is 10000
            analysis_sim_norm=False,
            cal_dormant_ratio=False,
            model=dict(
                observation_shape=observation_shape,
                image_channel=image_channel,
                frame_stack_num=frame_stack_num,
                gray_scale=gray_scale,
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
            eval_freq=int(5e3),
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
            num_unroll_steps=num_unroll_steps,
            # ============= RND specific settings =============
            use_rnd_model=use_rnd_model,
            rnd_weights=rnd_weights,
            rnd_random_collect_episode_num=30,
            use_momentum_representation_network=True,
            target_model_for_intrinsic_reward_update_type='assign',
            target_update_freq_for_intrinsic_reward=1000,
            target_update_theta_for_intrinsic_reward=0.005,
            reward_model=dict(
                device='cuda',
                type='rnd_unizero',
                intrinsic_reward_type='add',
                input_type='obs',  # options: ['obs', 'latent_state', 'obs_latent_state']
                activation_type='LeakyReLU',
                enable_image_logging=True,
                frame_stack_num=frame_stack_num,
                
                update_proportion=0.25, # 每次计算loss只有部分值参与计算，防止rnd网络更新太快
                # —— 新增：自适应权重调度 —— #
                use_intrinsic_weight_schedule=use_intrinsic_weight_schedule,     # 打开自适应权重
                intrinsic_weight_mode='cosine',         # 'cosine' | 'linear' | 'constant'
                intrinsic_weight_warmup=10000,           # 前多少次 estimate 权重=0
                intrinsic_weight_ramp=20000,            # 从min升到max所需的 estimate 数
                intrinsic_weight_min=0.0,               
                intrinsic_weight_max=intrinsic_weight_max, 
                
                obs_shape=(3, 96, 96),
                # obs_shape=observation_shape,
                latent_state_dim=256,
                hidden_size_list=[32, 64, 64],
                output_dim=512,
                
                input_norm=True,
                input_norm_clamp_max=5,
                input_norm_clamp_min=-5,
                
                intrinsic_norm=True,
                intrinsic_norm_type=intrinsic_norm_type, # 'reward | 'return'
                instrinsic_gamma=0.99,
                
                intrinsic_norm_reward_clamp_max=10, # 只有在reward的情况下生效
                
                extrinsic_sign=False,
                extrinsic_norm=False,
                extrinsic_norm_clamp_min=-5,
                extrinsic_norm_clamp_max=5,
                discount_factor=0.997,
                
            ),
            
            bp_update_sync=True,
            multi_gpu=False,
            
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
    from lzero.entry import train_muzero_segment_with_reward_model
    intrinsic_reward_type = main_config.policy.reward_model.intrinsic_reward_type
    input_type = main_config.policy.reward_model.input_type
    intrinsic_weight_max = main_config.policy.reward_model.intrinsic_weight_max
    input_norm = main_config.policy.reward_model.input_norm
    intrinsic_norm = main_config.policy.reward_model.intrinsic_norm
    use_intrinsic_weight_schedule = main_config.policy.reward_model.use_intrinsic_weight_schedule
    main_config.exp_name = f'./data_muzero_rnd/atari/{env_id[:-14]}/rnd_{intrinsic_norm_type}_loss_w_{rnd_weights}_{intrinsic_reward_type}_{input_type}_wmax_{intrinsic_weight_max}_input_norm_{input_norm}_intrinsic_norm_{intrinsic_norm}_use_intrinsic_weight_schedule_{use_intrinsic_weight_schedule}'
    train_muzero_segment_with_reward_model([main_config, create_config], 
                                            seed=seed, 
                                            max_env_step=max_env_step)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process different environments and seeds.')
    parser.add_argument('--env', type=str, help='The environment to use', default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    args = parser.parse_args()

    main(args.env, args.seed)