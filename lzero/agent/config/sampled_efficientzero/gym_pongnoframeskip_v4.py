from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
continuous_action_space = False
K = 5  # num_of_sampled_actions
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
update_per_collect = 1000
batch_size = 256
max_env_step = int(1e6)
reanalyze_ratio = 0.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

cfg = dict(
    main_config=dict(
        exp_name='PongNoFrameskip-v4-SampledEfficientZero',
        seed=0,
        env=dict(
            env_id='PongNoFrameskip-v4',
            env_name='PongNoFrameskip-v4',
            obs_shape=(4, 96, 96),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
        ),
        policy=dict(
            model=dict(
                observation_shape=(4, 96, 96),
                frame_stack_num=4,
                action_space_size=6,
                downsample=True,
                continuous_action_space=continuous_action_space,
                num_of_sampled_actions=K,
                discrete_action_encoding_type='one_hot',
                norm_type='BN', 
            ),
            cuda=True,
            env_type='not_board_games',
            game_segment_length=400,
            use_augmentation=True,
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            optim_type='SGD',
            lr_piecewise_constant_decay=True,
            learning_rate=0.2,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            policy_loss_type='cross_entropy',
            n_episode=n_episode,
            eval_freq=int(2e3),
            replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
        ),
        wandb_logger=dict(
            gradient_logger=False, video_logger=False, plot_logger=False, action_logger=False, return_logger=False
        ),
    ),
    create_config = dict(
        env=dict(
            type='atari_lightzero',
            import_names=['zoo.atari.envs.atari_lightzero_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(
            type='sampled_efficientzero',
            import_names=['lzero.policy.sampled_efficientzero'],
        ),
    )
)

cfg = EasyDict(cfg)
