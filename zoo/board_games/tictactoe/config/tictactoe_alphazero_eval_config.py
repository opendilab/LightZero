from easydict import EasyDict

board_size = 3  # fixed

collector_env_num = 1
n_episode = 1
evaluator_env_num = 1
num_simulations = 50
update_per_collect = 100
batch_size = 256
agent_vs_human = False

tictactoe_alphazero_config = dict(
    exp_name='data_ez_ptree/tictactoe_self-play_alphazero',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        board_size=board_size,
        battle_mode='self_play_mode',
        channel_last=False,  # NOTE
        agent_vs_human=agent_vs_human,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        cuda=True,
        board_size=board_size,
        model=dict(
            categorical_distribution=False,
            representation_network_type='conv_res_blocks',
            observation_shape=(3, board_size, board_size),
            action_space_size=int(1 * board_size * board_size),
            downsample=False,
            reward_support_size=1,
            value_support_size=1,
            num_res_blocks=1,
            num_channels=16,
            value_head_channels=16,
            policy_head_channels=16,
            fc_value_layers=[8],
            fc_policy_layers=[8],
            batch_norm_momentum=0.1,
            last_linear_layer_init_zero=True,
            state_norm=False,
        ),
        multi_gpu=False,
        batch_size=batch_size,
        optim_type='Adam',
        learning_rate=0.003,
        weight_decay=0.0001,
        update_per_collect=update_per_collect,
        grad_norm=0.5,
        value_weight=1.0,
        entropy_weight=0.0,
        n_episode=n_episode,
        eval_freq=int(2e3),
        mcts=dict(num_simulations=num_simulations),
    ),
)

tictactoe_alphazero_config = EasyDict(tictactoe_alphazero_config)
main_config = tictactoe_alphazero_config

tictactoe_alphazero_create_config = dict(
    env=dict(
        type='tictactoe',
        import_names=['zoo.board_games.tictactoe.envs.tictactoe_env'],
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
tictactoe_alphazero_create_config = EasyDict(tictactoe_alphazero_create_config)
create_config = tictactoe_alphazero_create_config

if __name__ == '__main__':
    from lzero.entry import eval_alphazero
    import numpy as np

    seed = 0
    test_episodes = 5
    reward_mean, reward_lst = eval_alphazero(
        [main_config, create_config], seed=seed, test_episodes=test_episodes, max_env_step=int(1e5)
    )

    reward_lst = np.array(reward_lst)
    reward_mean = np.array(reward_mean)

    print("=" * 20)
    print(f'we eval total {seed} seed. In each seed, we test {test_episodes} episodes.')
    print('reward_mean:', reward_mean)
    print(
        f'win rate: {len(np.where(reward_lst == 1.)[0]) / test_episodes}, draw rate: {len(np.where(reward_lst == 0.)[0]) / test_episodes}, lose rate: {len(np.where(reward_lst == -1.)[0]) / test_episodes}'
    )
    print("=" * 20)
