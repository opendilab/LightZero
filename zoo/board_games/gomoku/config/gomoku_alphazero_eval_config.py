from easydict import EasyDict

board_size = 6  # default_size is 15

collector_env_num = 1
n_episode = 1
evaluator_env_num = 1
num_simulations = 50
update_per_collect = 100
batch_size = 256
agent_vs_human = False

gomoku_alphazero_config = dict(
    exp_name='data_ez_ptree/gomoku_self-play_alphazero',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        board_size=board_size,
        battle_mode='self_play_mode',
        bot_action_type='v0',
        # NOTE
        channel_last=False,
        scale=True,
        prob_random_agent=0.,
        agent_vs_human=agent_vs_human,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model_path='/mnt/lustre/yangzhenjie/code/LightZero/zoo/board_games/gomoku/experiment/gomoku_alphazero_bs256_sub32325_up100_s50/data_ez_ptree/gomoku_self-play_alphazero/ckpt/ckpt_best.pth.tar',
        type='alphazero',
        env_name='gomoku',
        cuda=True,
        board_size=board_size,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        model=dict(
            categorical_distribution=False,
            # representation_network_type='identity',
            representation_network_type='conv_res_blocks',
            observation_shape=(3, board_size, board_size),
            action_space_size=int(1 * board_size * board_size),
            downsample=False,
            reward_support_size=1,
            value_support_size=1,
            num_blocks=1,
            num_channels=32,
            reduced_channels_value=16,
            reduced_channels_policy=16,
            fc_value_layers=[32],
            fc_policy_layers=[32],
            bn_mt=0.1,
            last_linear_layer_init_zero=True,
            state_norm=False,
        ),
        learn=dict(
            multi_gpu=False,
            batch_size=batch_size,
            learning_rate=0.001,
            weight_decay=0.0001,
            update_per_collect=update_per_collect,
            grad_norm=0.5,
            value_weight=1.0,
            entropy_weight=0.0,
            optim_type='Adam',
            learner=dict(
                hook=dict(
                    load_ckpt_before_run='',
                    log_show_after_iter=1,
                    save_ckpt_after_iter=10000,
                    save_ckpt_after_run=True, ),
            )
        ),
        collect=dict(
            unroll_len=1,
            n_episode=n_episode,
            collector=dict(
                env=dict(
                    type='gomoku',
                    import_names=['zoo.board_games.gomoku.envs.gomoku_env'], ),
                augmentation=True,
            ),
            mcts=dict(num_simulations=num_simulations)
        ),
        eval=dict(
            evaluator=dict(
                n_episode=evaluator_env_num,
                eval_freq=int(100),
                stop_value=1,
                env=dict(
                    type='gomoku',
                    import_names=['zoo.board_games.gomoku.envs.gomoku_env'], ),
            ),
            mcts=dict(num_simulations=num_simulations)
        ),
        other=dict(
            replay_buffer=dict(
                replay_buffer_size=int(1e6),
                type='naive',
                save_episode=False,
                periodic_thruput_seconds=60,
            )
        ),
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
        import_names=['core.policy.alphazero'],
    ),
    collector=dict(
        type='episode_alphazero',
        get_train_sample=False,
        # get_train_sample=True,
        import_names=['core.worker.collector.alphazero_collector'],
    ),
    evaluator=dict(
        type='alphazero',
        import_names=['core.worker.collector.alphazero_evaluator'],
    )

)
gomoku_alphazero_create_config = EasyDict(gomoku_alphazero_create_config)
create_config = gomoku_alphazero_create_config

if __name__ == '__main__':
    from lzero.entry import eval_alphazero
    import numpy as np

    returns_mean_seeds = []
    returns_seeds = []
    seeds = [0, 1]
    num_episodes_each_seed = 2
    total_test_episodes = num_episodes_each_seed * len(seeds)
    for seed in seeds:
        returns_mean, returns = eval_alphazero([main_config, create_config], seed=seed,
                                                            num_episodes_each_seed=num_episodes_each_seed,
                                                            print_seed_details=True, max_env_step=int(1e5))
        returns_mean_seeds.append(returns_mean)
        returns_seeds.append(returns)

    returns_mean_seeds = np.array(returns_mean_seeds)
    returns_seeds = np.array(returns_seeds)

    print("=" * 20)
    print(f'We eval total {len(seeds)} seeds. In each seed, we eval {num_episodes_each_seed} episodes.')
    print(f'In seeds {seeds}, returns_mean_seeds is {returns_mean_seeds}, returns is {returns_seeds}')
    print('In all seeds, reward_mean:', returns_mean_seeds.mean(), end='. ')
    print(f'win rate: {len(np.where(returns_seeds == 1.)[0]) / total_test_episodes}, draw rate: {len(np.where(returns_seeds == 0.)[0]) / total_test_episodes}, lose rate: {len(np.where(returns_seeds == -1.)[0]) / total_test_episodes}')
    print("=" * 20)
