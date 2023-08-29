from easydict import EasyDict

env_name = 'ptz_simple_spread'
multi_agent = True

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
seed = 0
n_agent = 3
n_landmark = n_agent
collector_env_num = 8
evaluator_env_num = 8
n_episode = 8
batch_size = 256
num_simulations = 25
update_per_collect = 100
reanalyze_ratio = 0.
action_space_size = 5
eps_greedy_exploration_in_collect = False
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

main_config = dict(
    exp_name=
    f'data_ez_ctree/{env_name}_efficientzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed{seed}',
    env=dict(
        env_family='mpe',
        env_id='simple_spread_v2',
        n_agent=n_agent,
        n_landmark=n_landmark,
        max_cycles=25,
        agent_obs_only=False,
        agent_specific_global_state=True,
        continuous_actions=False,
        stop_value=0,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        multi_agent=multi_agent,
        ignore_done=False,
        model=dict(
            model_type='structure',
            latent_state_dim=256,
            frame_stack_num=1,
            action_space='discrete',
            action_space_size=action_space_size,
            agent_num=n_agent,
            agent_obs_shape=2 + 2 + n_landmark * 2 + (n_agent - 1) * 2 + (n_agent - 1) * 2,
            global_obs_shape=2 + 2 + n_landmark * 2 + (n_agent - 1) * 2 + (n_agent - 1) * 2 + n_agent * (2 + 2) +
            n_landmark * 2 + n_agent * (n_agent - 1) * 2,
            discrete_action_encoding_type='one_hot',
            global_cooperation=True, # TODO: doesn't work now
            hidden_size_list=[256, 256],
            norm_type='BN',
        ),
        cuda=True,
        mcts_ctree=True,
        gumbel_algo=False,
        env_type='not_board_games',
        game_segment_length=50,
        random_collect_episode_num=0,
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            type='linear',
            start=1.,
            end=0.05,
            decay=int(2e4),
        ),
        use_augmentation=False,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        ssl_loss_weight=0,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
    learn=dict(learner=dict(
        log_policy=True,
        hook=dict(log_show_after_iter=10, ),
    ), ),
)
main_config = EasyDict(main_config)
create_config = dict(
    env=dict(
        import_names=['zoo.petting_zoo.envs.petting_zoo_simple_spread_env'],
        type='petting_zoo',
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='multi_agent_efficientzero',
        import_names=['lzero.policy.multi_agent_efficientzero'],
    ),
    collector=dict(
        type='episode_muzero',
        import_names=['lzero.worker.muzero_collector'],
    )
)
create_config = EasyDict(create_config)
ptz_simple_spread_efficientzero_config = main_config
ptz_simple_spread_efficientzero_create_config = create_config

if __name__ == '__main__':
    from zoo.petting_zoo.entry import train_muzero
    train_muzero([main_config, create_config], seed=seed)
