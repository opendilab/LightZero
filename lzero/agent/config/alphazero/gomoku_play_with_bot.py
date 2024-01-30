from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
board_size = 6  # default_size is 15
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
update_per_collect = 50
batch_size = 256
max_env_step = int(5e5)
prob_random_action_in_bot = 0.5
mcts_ctree = False
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

cfg = dict(
    main_config=dict(
        exp_name='Gomoku-play-with-bot-AlphaZero',
        seed=0,
        env=dict(
            env_id='Gomoku-play-with-bot',
            battle_mode='play_with_bot_mode',
            render_mode='image_savefile_mode',
            replay_format='mp4',
            board_size=board_size,
            bot_action_type='v1',
            prob_random_action_in_bot=prob_random_action_in_bot,
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
            screen_scaling=9,
            render_mode=None,
            replay_path=None,
            alphazero_mcts_ctree=mcts_ctree,
            # ==============================================================
        ),
        policy=dict(
            mcts_ctree=mcts_ctree,
            # ==============================================================
            # for the creation of simulation env
            simulation_env_name='gomoku',
            simulation_env_config_type='play_with_bot',
            # ==============================================================
            torch_compile=False,
            tensor_float_32=False,
            model=dict(
                observation_shape=(3, board_size, board_size),
                action_space_size=int(1 * board_size * board_size),
                num_res_blocks=1,
                num_channels=32,
            ),
            cuda=True,
            board_size=board_size,
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            optim_type='Adam',
            lr_piecewise_constant_decay=False,
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
            type='gomoku',
            import_names=['zoo.board_games.gomoku.envs.gomoku_env'],
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
        ),
    )
)

cfg = EasyDict(cfg)
