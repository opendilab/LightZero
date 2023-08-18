from easydict import EasyDict

env_name = 'gobigger'
multi_agent = True

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
seed = 0
collector_env_num = 32
n_episode = 32
evaluator_env_num = 5
num_simulations = 50
update_per_collect = 1000
batch_size = 256
reanalyze_ratio = 0.
action_space_size = 27
direction_num = 12
eps_greedy_exploration_in_collect = True
player_num_per_team = 2
team_num = 2  
agent_num = player_num_per_team*team_num  # default is GoBigger T2P2
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

gobigger_muzero_config = dict(
    exp_name=f'data_mz_ctree/{env_name}_muzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed{seed}',
    env=dict(
        env_name=env_name,
        player_num_per_team=player_num_per_team,
        team_num=team_num,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        multi_agent=multi_agent,
        ignore_done=True,
        model=dict(
            model_type='structure',
            env_name=env_name,
            agent_num=agent_num,
            team_num=team_num,
            latent_state_dim=176,
            frame_stack_num=1,
            action_space_size=action_space_size,
            self_supervised_learning_loss=False,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
        ),
        cuda=True,
        mcts_ctree=True,
        gumbel_algo=False,
        env_type='not_board_games',
        game_segment_length=500,
        random_collect_episode_num=0,
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            type='linear',
            start=1.,
            end=0.05,
            decay=int(1e5),
        ),
        use_augmentation=False,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='SGD',
        lr_piecewise_constant_decay=True,
        learning_rate=0.2,
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
gobigger_muzero_config = EasyDict(gobigger_muzero_config)
main_config = gobigger_muzero_config

gobigger_muzero_create_config = dict(
    env=dict(
        type='gobigger_lightzero',
        import_names=['zoo.gobigger.env.gobigger_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='multi_agent_muzero',
        import_names=['lzero.policy.multi_agent_muzero'],
    ),
    collector=dict(
        type='multi_agent_episode_muzero',
        import_names=['lzero.worker.multi_agent_muzero_collector'],
    )
)
gobigger_muzero_create_config = EasyDict(gobigger_muzero_create_config)
create_config = gobigger_muzero_create_config

if __name__ == "__main__":
    from zoo.gobigger.entry import train_muzero_gobigger
    train_muzero_gobigger([main_config, create_config], seed=seed)
