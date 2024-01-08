from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
update_per_collect = 1000
batch_size = 256
max_env_step = int(1e6)
reanalyze_ratio = 0.
eps_greedy_exploration_in_collect = True
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

cfg = dict(
    main_config=dict(
        exp_name='BreakoutNoFrameskip-v4-EfficientZero',
        seed=0,
        env=dict(
            env_id='BreakoutNoFrameskip-v4',
            env_name='BreakoutNoFrameskip-v4',
            obs_shape=(4, 96, 96),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
            collect_max_episode_steps=int(5e3),
            eval_max_episode_steps=int(2e4),
        ),
        policy=dict(
            model=dict(
                observation_shape=(4, 96, 96),
                frame_stack_num=4,
                action_space_size=4,
                downsample=True,
                discrete_action_encoding_type='one_hot',
                norm_type='BN',
            ),
            cuda=True,
            env_type='not_board_games',
            game_segment_length=400,
            random_collect_episode_num=0,
            eps=dict(
                eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
                # need to dynamically adjust the number of decay steps according to the characteristics of the environment and the algorithm
                type='linear',
                start=1.,
                end=0.05,
                decay=int(1e5),
            ),
            use_augmentation=True,
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            optim_type='SGD',
            lr_piecewise_constant_decay=True,
            learning_rate=0.2,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
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
            type='efficientzero',
            import_names=['lzero.policy.efficientzero'],
        ),
    )
)

cfg = EasyDict(cfg)
