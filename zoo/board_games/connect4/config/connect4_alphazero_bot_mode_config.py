from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 32
n_episode = 32
evaluator_env_num = 5
num_simulations = 10
update_per_collect = 50
batch_size = 256
max_env_step = int(1e6)
prob_random_action_in_bot = 0
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================
connect4_alphazero_config = dict(
    exp_name='data_az_ptree/connect4_mcts-bot_seed0',
    env=dict(
        battle_mode='play_with_bot_mode',
        mcts_mode='play_with_bot_mode',
        bot_action_type='mcts',
        prob_random_action_in_bot=prob_random_action_in_bot,
        channel_last=False,  # NOTE
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        torch_compile=False,
        tensor_float_32=False,
        model=dict(
            observation_shape=(3, 6, 7),
            action_space_size=int(1*7),
            num_res_blocks=1,
            num_channels=32,
        ),
        cuda=True,
        env_type='board_games',
        board_size=3,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        grad_clip_value=0.5,
        value_weight=1.0,
        entropy_weight=0.0,
        n_episode=n_episode,
        eval_freq=int(2e3),
        mcts=dict(num_simulations=num_simulations),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

connect4_alphazero_config = EasyDict(connect4_alphazero_config)
main_config = connect4_alphazero_config

connect4_alphazero_create_config = dict(
    env=dict(
        type='connect4',
        import_names=['zoo.board_games.connect4.envs.connect4_env'],
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
connect4_alphazero_create_config = EasyDict(connect4_alphazero_create_config)
create_config = connect4_alphazero_create_config

if __name__ == '__main__':
    from lzero.entry import train_alphazero
    train_alphazero([main_config, create_config], seed=0, max_env_step=max_env_step)
