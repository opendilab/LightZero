from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 5
num_simulations = 400
update_per_collect = 200
batch_size = 512
max_env_step = int(1e6)
mcts_ctree = True
# mcts_ctree = False


# TODO: for debug
collector_env_num = 2
n_episode = 2
evaluator_env_num = 2
num_simulations = 4
update_per_collect = 2
batch_size = 2
max_env_step = int(1e4)
# mcts_ctree = False
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================
chess_alphazero_config = dict(
    exp_name='data_az_ctree/chess_sp-mode_alphazero_seed0',
    env=dict(
        board_size=8,
        battle_mode='self_play_mode',
        channel_last=False,  
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        # ==============================================================
        # for the creation of simulation env
        agent_vs_human=False,
        prob_random_agent=0,
        prob_expert_agent=0,
        scale=True,
        alphazero_mcts_ctree=mcts_ctree,
        save_replay_gif=False,
        replay_path_gif='./replay_gif',
        # ==============================================================
    ),
    policy=dict(
        mcts_ctree=mcts_ctree,
        # ==============================================================
        # for the creation of simulation env
        simulation_env_id='chess',
        simulation_env_config_type='self_play',  
        # ==============================================================
        model=dict(
            observation_shape=(8, 8, 20),
            action_space_size=int(8 * 8 * 73),
            # TODO: for debug
            num_res_blocks=1,
            num_channels=1,
            value_head_hidden_channels=[16],
            policy_head_hidden_channels=[16],
            # num_res_blocks=8,
            # num_channels=256,
            # value_head_hidden_channels=[256, 256],
            # policy_head_hidden_channels=[256, 256],
        ),
        cuda=True,
        board_size=8,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='AdamW',
        lr_piecewise_constant_decay=False,
        learning_rate=0.0001,
        grad_clip_value=0.5,
        value_weight=1.0,
        entropy_weight=0.01,
        n_episode=n_episode,
        eval_freq=int(1e3),
        mcts=dict(num_simulations=num_simulations),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

chess_alphazero_config = EasyDict(chess_alphazero_config)
main_config = chess_alphazero_config

chess_alphazero_create_config = dict(
    env=dict(
        type='chess_lightzero',
        import_names=['zoo.board_games.chess.envs.chess_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='alphazero',
        import_names=['lzero.policy.alphazero'],
    ),
    collector=dict(
        type='episode_alphazero',
        import_names=['lzero.worker.alphazero_collector'], 
    ),
    evaluator=dict(
        type='alphazero',
        import_names=['lzero.worker.alphazero_evaluator'],
    )
)
chess_alphazero_create_config = EasyDict(chess_alphazero_create_config)
create_config = chess_alphazero_create_config

if __name__ == '__main__':
    from lzero.entry import train_alphazero
    train_alphazero([main_config, create_config], seed=0, max_env_step=max_env_step)