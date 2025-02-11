from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 5
num_simulations = 25
update_per_collect = 50
batch_size = 256
max_env_step = int(2e5)
mcts_ctree = False
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

cfg = dict(
    main_config=dict(
        exp_name='TicTacToe-play-with-bot-AlphaZero',
        seed=0,
        env=dict(
            env_id='TicTacToe-play-with-bot',
            board_size=3,
            battle_mode='play_with_bot_mode',
            bot_action_type='v0',  # {'v0', 'alpha_beta_pruning'}
            channel_last=False,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
            # ==============================================================
            # for the creation of simulation env
            agent_vs_human=False,
            prob_random_agent=0,
            prob_expert_agent=0,
            scale=True,
            alphazero_mcts_ctree=mcts_ctree,
            save_replay_gif=False,
            replay_path_gif='./replay_gif',
            # ==============================================================
        ),
        policy=dict(
            mcts_ctree=mcts_ctree,
            # ==============================================================
            # for the creation of simulation env
            simulation_env_id='tictactoe',
            simulation_env_config_type='play_with_bot',
            # ==============================================================
            model=dict(
                observation_shape=(3, 3, 3),
                action_space_size=int(1 * 3 * 3),
                # We use the small size model for tictactoe.
                num_res_blocks=1,
                num_channels=16,
                value_head_hidden_channels=[8],
                policy_head_hidden_channels=[8],
            ),
            cuda=True,
            board_size=3,
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            optim_type='Adam',
            piecewise_decay_lr_scheduler=False,
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
        wandb_logger=dict(
            gradient_logger=False, video_logger=False, plot_logger=False, action_logger=False, return_logger=False
        ),
    ),
    create_config = dict(
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
            import_names=['lzero.worker.alphazero_collector'],
        ),
        evaluator=dict(
            type='alphazero',
            import_names=['lzero.worker.alphazero_evaluator'],
        )
    )
)

cfg = EasyDict(cfg)
