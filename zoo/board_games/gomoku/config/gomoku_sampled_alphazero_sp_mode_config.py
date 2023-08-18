from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
board_size = 6  # default_size is 15
collector_env_num = 32
n_episode = 32
evaluator_env_num = 5
num_simulations = 50
update_per_collect = 100
batch_size = 256
max_env_step = int(10e6)
prob_random_action_in_bot = 0.5
mcts_ctree = False
num_of_sampled_actions = 20

# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================
gomoku_sampled_alphazero_config = dict(
    exp_name=
    f'data_saz_ptree/gomoku_sampled_alphazero_sp-mode_rand{prob_random_action_in_bot}_na{num_of_sampled_actions}_ns{num_simulations}_upc{update_per_collect}_seed0',
    env=dict(
        stop_value=2,
        board_size=board_size,
        battle_mode='self_play_mode',
        bot_action_type='v0',
        prob_random_action_in_bot=prob_random_action_in_bot,
        channel_last=False,  # NOTE
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        env_name="Gomoku",
        prob_random_agent=0,
        mcts_mode='self_play_mode',  # only used in AlphaZero
        scale=True,
        agent_vs_human=False,
        check_action_to_connect4_in_bot_v0=False,
        mcts_ctree=mcts_ctree,
    ),
    policy=dict(
        torch_compile=False,
        tensor_float_32=False,
        model=dict(
            observation_shape=(3, board_size, board_size),
            action_space_size=int(1 * board_size * board_size),
            num_res_blocks=1,
            num_channels=32,
            num_of_sampled_actions=num_of_sampled_actions,
        ),
        sampled_algo=True,
        simulate_env_name="gomoku",
        mcts_ctree=mcts_ctree,
        simulate_env_config_type='self_play',
        cuda=True,
        board_size=board_size,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        manual_temperature_decay=True,
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

gomoku_sampled_alphazero_config = EasyDict(gomoku_sampled_alphazero_config)
main_config = gomoku_sampled_alphazero_config

gomoku_sampled_alphazero_create_config = dict(
    env=dict(
        type='gomoku',
        import_names=['zoo.board_games.gomoku.envs.gomoku_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sampled_alphazero',
        import_names=['lzero.policy.sampled_alphazero'],
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
gomoku_sampled_alphazero_create_config = EasyDict(gomoku_sampled_alphazero_create_config)
create_config = gomoku_sampled_alphazero_create_config

if __name__ == '__main__':
    if main_config.policy.tensor_float_32:
        import torch

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = True
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = True

    from lzero.entry import train_alphazero
    train_alphazero([main_config, create_config], seed=0, max_env_step=max_env_step)
