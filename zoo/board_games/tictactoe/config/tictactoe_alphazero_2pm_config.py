from easydict import EasyDict

board_size = 3  # fixed

collector_env_num = 32
n_episode = 32
evaluator_env_num = 16
num_simulations = 50
update_per_collect = 100
batch_size = 256

tictactoe_alphazero_config = dict(
    exp_name='data_ez_ptree/tictactoe_self-play_alphazero',
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
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        type='alphazero',
        env_name='tictactoe',
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
                    save_ckpt_after_run=True,
                ),
            )
        ),
        collect=dict(
            unroll_len=1,
            n_episode=n_episode,
            collector=dict(
                env=dict(
                    type='tictactoe',
                    import_names=['zoo.board_games.tictactoe.envs.tictactoe_env'],
                ),
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
                    type='tictactoe',
                    import_names=['zoo.board_games.tictactoe.envs.tictactoe_env'],
                ),
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
        # get_train_sample=True,
        import_names=['lzero.worker.collector.alphazero_collector'],
    ),
    evaluator=dict(
        type='alphazero',
        import_names=['lzero.worker.collector.alphazero_evaluator'],
    )
)
tictactoe_alphazero_create_config = EasyDict(tictactoe_alphazero_create_config)
create_config = tictactoe_alphazero_create_config

if __name__ == '__main__':
    from lzero.entry import serial_pipeline_alphazero

    serial_pipeline_alphazero([main_config, create_config], seed=0)
