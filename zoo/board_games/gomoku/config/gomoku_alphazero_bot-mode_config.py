from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
# board_size = 6  # default_size is 15
# collector_env_num = 32
# n_episode = 32
# evaluator_env_num = 5
# num_simulations = 50
# update_per_collect = 50
# batch_size = 256
# max_env_step = int(1e6)
# prob_random_action_in_bot = 0.5

# debug config
board_size = 6  # default_size is 15
collector_env_num = 1
n_episode = 1
evaluator_env_num = 1
num_simulations = 5
update_per_collect = 2
batch_size = 4
max_env_step = int(2e6)
prob_random_action_in_bot = 0.1
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================
gomoku_alphazero_config = dict(
    exp_name=f'data_az_ptree/gomoku_alphazero_bot-mode_rand{prob_random_action_in_bot}_ns{num_simulations}_upc{update_per_collect}_seed0',
    env=dict(
        board_size=board_size,
        battle_mode='play_with_bot_mode',
        bot_action_type='v0',
        prob_random_action_in_bot=prob_random_action_in_bot,
        channel_last=False,  # NOTE
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            # We use the half size model for gomoku
            observation_shape=(3, board_size, board_size),
            action_space_size=int(1 * board_size * board_size),
            representation_network_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
            num_res_blocks=1,
            num_channels=32,
        ),
        stop_value=2,
        cuda=True,
        board_size=board_size,
        lr_piecewise_constant_decay=False,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        learning_rate=0.003,
        weight_decay=0.0001,
        grad_norm=0.5,
        value_weight=1.0,
        entropy_weight=0.0,
        n_episode=n_episode,
        eval_freq=int(2e3),
        mcts=dict(num_simulations=num_simulations),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

gomoku_alphazero_config = EasyDict(gomoku_alphazero_config)
main_config = gomoku_alphazero_config

gomoku_alphazero_create_config = dict(
    env=dict(
        type='gomoku',
        import_names=['zoo.board_games.gomoku.envs.gomoku_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='alphazero',
        import_names=['lzero.policy.alphazero'],
    ),
    collector=dict(
        type='episode_alphazero',
        get_train_sample=False,
        import_names=['lzero.worker.alphazero_collector'],
    ),
    evaluator=dict(
        type='alphazero',
        import_names=['lzero.worker.alphazero_evaluator'],
    )
)
gomoku_alphazero_create_config = EasyDict(gomoku_alphazero_create_config)
create_config = gomoku_alphazero_create_config

if __name__ == '__main__':
    from lzero.entry import train_alphazero

    train_alphazero([main_config, create_config], seed=0, max_env_step=max_env_step)
