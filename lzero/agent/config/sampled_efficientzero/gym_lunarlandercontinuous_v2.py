from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
continuous_action_space = True
K = 20  # num_of_sampled_actions
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
        exp_name='LunarLanderContinuous-v2-SampledEfficientZero',
        seed=0,
        env=dict(
            env_id='LunarLanderContinuous-v2',
            env_name='LunarLanderContinuous-v2',
            continuous=True,
            manually_discretization=False,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
        ),
        policy=dict(
            mcts_ctree=True,
            model=dict(
                observation_shape=8,
                action_space_size=2,
                continuous_action_space=continuous_action_space,
                num_of_sampled_actions=K,
                sigma_type='conditioned',
                model_type='mlp',
                lstm_hidden_size=256,
                latent_state_dim=256,
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
            random_collect_episode_num=0,
            # NOTE: for continuous gaussian policy, we use the policy_entropy_loss as in the original Sampled MuZero paper.
            policy_entropy_loss_weight=5e-3,
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
            type='sampled_efficientzero',
            import_names=['lzero.policy.sampled_efficientzero'],
        ),
    ),
)

cfg = EasyDict(cfg)
