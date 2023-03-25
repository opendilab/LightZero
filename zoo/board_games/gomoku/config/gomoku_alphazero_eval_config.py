from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
board_size = 6  # default_size is 15
collector_env_num = 1
n_episode = 1
evaluator_env_num = 1
num_simulations = 100
update_per_collect = 100
batch_size = 256
max_env_step = int(2e6)
prob_random_action_in_bot = 0.5
agent_vs_human = True

# debug config
# board_size = 6  # default_size is 15
# collector_env_num = 1
# n_episode = 1
# evaluator_env_num = 2
# num_simulations = 5
# update_per_collect = 2
# batch_size = 4
# max_env_step = int(2e6)
# prob_random_action_in_bot = 0.1
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================
gomoku_alphazero_config = dict(
    exp_name=f'data_az_ptree/gomoku_alphazero_bot-mode_rand{prob_random_action_in_bot}_ns{num_simulations}_upc{update_per_collect}_seed0',
    env=dict(
        stop_value=2,
        board_size=board_size,
        battle_mode='eval_mode',
        bot_action_type='v0',
        prob_random_action_in_bot=prob_random_action_in_bot,
        channel_last=False,  # NOTE
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        agent_vs_human=agent_vs_human,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        cuda=True,
        board_size=board_size,
        model=dict(
            # We use the half size model for gomoku
            observation_shape=(3, board_size, board_size),
            action_space_size=int(1 * board_size * board_size),
            representation_network_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
            num_res_blocks=1,
            num_channels=32,
        ),
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
    env_manager=dict(type='base'),
    # env_manager=dict(type='subprocess'),
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
    from lzero.entry import eval_alphazero
    import numpy as np

    """ 
    model_path (:obj:`Optional[str]`): The pretrained model path, which should
    point to the ckpt file of the pretrained model, and an absolute path is recommended.
    In LightZero, the path is usually something like ``exp_name/ckpt/ckpt_best.pth.tar``.
     """
    model_path = '/Users/puyuan/code/LightZero/zoo/board_games/gomoku/gomoku_alphazero_bot-mode_rand0.5_ns50_upc50_rr0.3_rbs1e5_seed0/ckpt/ckpt_best.pth.tar'

    returns_mean_seeds = []
    returns_seeds = []
    seeds = [0]
    num_episodes_each_seed = 3
    total_test_episodes = num_episodes_each_seed * len(seeds)
    for seed in seeds:
        returns_mean, returns = eval_alphazero([main_config, create_config], seed=seed,
                                               num_episodes_each_seed=num_episodes_each_seed,
                                               print_seed_details=True, model_path=model_path)
        returns_mean_seeds.append(returns_mean)
        returns_seeds.append(returns)

    returns_mean_seeds = np.array(returns_mean_seeds)
    returns_seeds = np.array(returns_seeds)

    print("=" * 20)
    print(f'We eval total {len(seeds)} seeds. In each seed, we eval {num_episodes_each_seed} episodes.')
    print(f'In seeds {seeds}, returns_mean_seeds is {returns_mean_seeds}, returns is {returns_seeds}')
    print('In all seeds, reward_mean:', returns_mean_seeds.mean(), end='. ')
    print(
        f'win rate: {len(np.where(returns_seeds == 1.)[0]) / total_test_episodes}, draw rate: {len(np.where(returns_seeds == 0.)[0]) / total_test_episodes}, lose rate: {len(np.where(returns_seeds == -1.)[0]) / total_test_episodes}')
    print("=" * 20)
