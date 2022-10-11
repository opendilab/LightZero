from easydict import EasyDict

gomoku_alphazero_config = dict(
    seed=0,
    exp_name='data_ez_ptree/gomoku_2pm_alphazero_seed0',
    env=dict(
        type='gomoku',
        import_names=['zoo.board_games.gomoku.envs.gomoku_env'],
        board_size=6,
        battle_mode='two_player_mode',
        prob_random_agent=0.,
        max_episode_steps=108000,
        collect_max_episode_steps=10800,
        eval_max_episode_steps=108000,
    ),
    model=dict(
        type='gomoku_model',
        import_names=['core.model.template.alphazero.alphazero_model_gomoku'],
        model_cfg=dict(
            input_channels=3,
            board_size=6,
        )
    ),
    policy=dict(
        # (string) RL policy register name (refer to function "register_policy").
        type='alphazero',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) whether use on-policy training pipeline(behaviour policy and training policy are the same)
        on_policy=True,  # for a2c strictly on policy algorithm this line should not be seen by users
        priority=False,
        # learn_mode config
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            batch_size=64,
            learning_rate=0.001,
            weight_decay=0.0001,
            # (List[float])
            betas=(0.9, 0.999),
            # (float)
            eps=1e-8,
            # (float)
            grad_norm=0.5,
            # : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : :
            # The following configs is algorithm-specific
            # : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : :
            # (float) loss weight of the value network the weight of policy network is set to 1
            value_weight=1.0,
            #    # (float) loss weight of the entropy regularization the weight of policy network is set to 1
            #    entropy_weight: 0.01
            learner=dict(
                # TODO(pu)
                log_policy=True,
                train_iterations=10000,
                hook=dict(
                    load_ckpt_before_run='',
                    log_show_after_iter=100,
                    save_ckpt_after_iter=10000,
                    save_ckpt_after_run=True,
                )
            )
        ),
        collect=dict(
            collector=dict(
                collect_n_episode=1,
                max_moves=1000,
                print_freq=10,
                augmentation=True,
            ),
            mcts=dict(num_simulations=50, )
        ),
        # the eval cost is expensive, so we set eval_freq larger
        eval=dict(evaluator=dict(n_episode=10, eval_freq=int(10), stop_value=1), mcts=dict(num_simulations=50, )),

        # command_mode config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # Decay type. Support ['exp', 'linear'].
                type='exp',
                start=0.95,
                end=0.1,
                decay=int(5e4),
            ),
            # the replay_buffer_size is ineffective, we specify it in game config
            replay_buffer=dict(
                replay_buffer_size=int(1e3),
                type='naive',
                deepcopy=False,
                enable_track_used_data=False,
                periodic_thruput_seconds=10000
            )
        ),
    ),
)
gomoku_alphazero_config = EasyDict(gomoku_alphazero_config)
main_config = gomoku_alphazero_config

if __name__ == '__main__':
    from core.entry import serial_pipeline_alphazero

    serial_pipeline_alphazero(main_config)
