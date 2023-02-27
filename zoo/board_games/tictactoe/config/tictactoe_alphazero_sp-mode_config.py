from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
# collector_env_num = 32
# n_episode = 32
# evaluator_env_num = 16
# num_simulations = 50
# update_per_collect = 100
# batch_size = 256
# max_env_step = int(1e5)

# debug config
collector_env_num = 1
n_episode = 1
evaluator_env_num = 1
num_simulations = 5
update_per_collect = 2
batch_size = 3
max_env_step = int(1e3)

# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================
tictactoe_alphazero_config = dict(
    exp_name='data_az_ptree/tictactoe_sp-mode_alphazero_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        board_size=3,
        battle_mode='self_play_mode',
        # NOTE
        channel_last=False,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        type='alphazero',
        env_name='tictactoe',
        cuda=True,
        board_size=3,
        model=dict(
            # ==============================================================
            # We use the small size model for tictactoe
            # ==============================================================
            representation_model_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
            observation_shape=(3, 3, 3),
            action_space_size=int(1 * 3 * 3),
            downsample=False,
            num_res_blocks=1,
            num_channels=16,
            value_head_channels=16,
            policy_head_channels=16,
            fc_value_layers=[8],
            fc_policy_layers=[8],
            reward_support_size=1,
            value_support_size=1,
        ),
        learn=dict(
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            learning_rate=0.001,
            weight_decay=0.0001,
            grad_norm=0.5,
            value_weight=1.0,
            entropy_weight=0.0,
            optim_type='Adam',
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
                replay_buffer_size=int(1e6),
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
    from lzero.entry import serial_pipeline_alphazero
    serial_pipeline_alphazero([main_config, create_config], seed=0, max_env_step=max_env_step)
