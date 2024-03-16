from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
update_per_collect = 200
batch_size = 256
max_env_step = int(5e6)
reanalyze_ratio = 0.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

cfg = dict(
    main_config=dict(
        exp_name='LunarLander-v2-EfficientZero',
        seed=0,
        env=dict(
            env_id='LunarLander-v2',
            continuous=False,
            manually_discretization=False,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
        ),
        policy=dict(
            model=dict(
                observation_shape=8,
                action_space_size=4,
                model_type='mlp', 
                lstm_hidden_size=256,
                latent_state_dim=256,
                discrete_action_encoding_type='one_hot',
                res_connection_in_dynamics=True,
                norm_type='BN', 
            ),
            cuda=True,
            env_type='not_board_games',
            game_segment_length=200,
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            optim_type='Adam',
            lr_piecewise_constant_decay=False,
            learning_rate=0.003,
            grad_clip_value=0.5,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            n_episode=n_episode,
            eval_freq=int(1e3),
            replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
        ),
        wandb_logger=dict(
            gradient_logger=False, video_logger=False, plot_logger=False, action_logger=False, return_logger=False
        ),
    ),
    create_config=dict(
        env=dict(
            type='lunarlander',
            import_names=['zoo.box2d.lunarlander.envs.lunarlander_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(
            type='efficientzero',
            import_names=['lzero.policy.efficientzero'],
        ),
    ),
)

cfg = EasyDict(cfg)
