from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 5
num_simulations = 50
update_per_collect = 50
batch_size = 256
max_env_step = int(1e5)
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
        channel_last=False,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        type='alphazero',
        env_name='tictactoe',
        cuda=True,
        board_size=3,
        model=dict(
            observation_shape=(3, 3, 3),
            action_space_size=int(1 * 3 * 3),
            downsample=False,
            # We use the small size model for tictactoe
            num_res_blocks=1,
            num_channels=16,
            value_head_channels=16,
            policy_head_channels=16,
            fc_value_layers=[8],
            fc_policy_layers=[8],
            last_linear_layer_init_zero=True,
            categorical_distribution=False,
            representation_network_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
        ),
        learn=dict(
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            optim_type='Adam',
            learning_rate=0.003,
            weight_decay=0.0001,
            grad_norm=0.5,
            value_weight=1.0,
            entropy_weight=0.0,
        ),
        collect=dict(
            unroll_len=1,
            n_episode=n_episode,
            collector=dict(augmentation=True,),
            mcts=dict(num_simulations=num_simulations)
        ),
        eval=dict(
            evaluator=dict(
                n_episode=evaluator_env_num,
                eval_freq=int(100),
                stop_value=1,
            ),
            mcts=dict(num_simulations=num_simulations)
        ),
        other=dict(
            replay_buffer=dict(
                replay_buffer_size=int(1e6),
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
tictactoe_alphazero_create_config = EasyDict(tictactoe_alphazero_create_config)
create_config = tictactoe_alphazero_create_config

if __name__ == '__main__':
    from lzero.entry import train_alphazero
    train_alphazero([main_config, create_config], seed=0, max_env_step=max_env_step)
