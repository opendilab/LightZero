from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
continuous_action_space = False
K = 2  # num_of_sampled_actions
num_simulations = 25
update_per_collect = 100
batch_size = 256
max_env_step = int(1e5)
reanalyze_ratio = 0.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

cfg = dict(
    main_config = dict(
        exp_name='CartPole-v0-SampledEfficientZero',
        env=dict(
            env_id='CartPole-v0',
            continuous=False,
            manually_discretization=False,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
        ),
        policy=dict(
            model=dict(
                observation_shape=4,
                action_space_size=2,
                continuous_action_space=continuous_action_space,
                num_of_sampled_actions=K,
                model_type='mlp', 
                lstm_hidden_size=128,
                latent_state_dim=128,
                discrete_action_encoding_type='one_hot',
                norm_type='BN', 
            ),
            cuda=True,
            env_type='not_board_games',
            game_segment_length=50,
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            optim_type='Adam',
            lr_piecewise_constant_decay=False,
            learning_rate=0.003,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            n_episode=n_episode,
            eval_freq=int(2e2),
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
            type='cartpole_lightzero',
            import_names=['zoo.classic_control.cartpole.envs.cartpole_lightzero_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(
            type='sampled_efficientzero',
            import_names=['lzero.policy.sampled_efficientzero'],
        ),
    ),
)

cfg = EasyDict(cfg)
