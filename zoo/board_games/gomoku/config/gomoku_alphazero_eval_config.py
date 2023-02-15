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
        channel_last=False,
        collect_max_episode_steps=int(1.08e4),
        eval_max_episode_steps=int(1.08e5),
        board_size=board_size,
        battle_mode='self_play_mode',
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
        model=dict(
            categorical_distribution=False,
            # representation_model_type='identity',
            representation_model_type='conv_res_blocks',
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
                replay_buffer_size=int(1e5),
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
    from core.entry import serial_pipeline_alphazero_eval
    import numpy as np

    seed = 0
    test_episodes = 15
    for i in range(15):
        reward_mean, reward_lst = serial_pipeline_alphazero_eval([main_config, create_config], seed=i, test_episodes=1, max_env_step=int(1e5))

    reward_lst = np.array(reward_lst)
    reward_mean = np.array(reward_mean)

    print("=" * 20)
    print(f'we eval total {seed} seed. In each seed, we test {test_episodes} episodes.')
    print('reward_mean:', reward_mean)
    print(f'win rate: {len(np.where(reward_lst == 1.)[0]) / test_episodes}, draw rate: {len(np.where(reward_lst == 0.)[0]) / test_episodes}, lose rate: {len(np.where(reward_lst == -1.)[0]) / test_episodes}')
    print("=" * 20)