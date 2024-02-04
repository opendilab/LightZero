from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
board_size = 6
num_simulations = 100
update_per_collect = 50
# board_size = 9
# num_simulations = 200
# update_per_collect = 100
num_of_sampled_actions = 20
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
batch_size = 256
max_env_step = int(10e6)
prob_random_action_in_bot = 0.5
mcts_ctree = False
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

cfg = dict(
    main_config=dict(
        exp_name='Gomoku-play-with-bot-SampledAlphaZero',
        seed=0,
        env=dict(
            env_id='Gomoku-play-with-bot',
            battle_mode='play_with_bot_mode',
            replay_format='mp4',
            stop_value=2,
            board_size=board_size,
            bot_action_type='v0',
            prob_random_action_in_bot=prob_random_action_in_bot,
            channel_last=False,
            use_katago_bot=False,
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
            check_action_to_connect4_in_bot_v0=False,
            simulation_env_id="gomoku",
            screen_scaling=9,
            render_mode='image_savefile_mode',
            replay_path=None,
            alphazero_mcts_ctree=mcts_ctree,
            # ==============================================================
        ),
        policy=dict(
            # ==============================================================
            # for the creation of simulation env
            simulation_env_id='gomoku',
            simulation_env_config_type='sampled_play_with_bot',
            # ==============================================================
            torch_compile=False,
            tensor_float_32=False,
            model=dict(
                observation_shape=(3, board_size, board_size),
                action_space_size=int(1 * board_size * board_size),
                num_res_blocks=1,
                num_channels=64,
            ),
            sampled_algo=True,
            mcts_ctree=mcts_ctree,
            policy_loss_type='KL',
            cuda=True,
            board_size=board_size,
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            optim_type='Adam',
            lr_piecewise_constant_decay=False,
            learning_rate=0.003,
            value_weight=1.0,
            entropy_weight=0.0,
            n_episode=n_episode,
            eval_freq=int(2e3),
            mcts=dict(num_simulations=num_simulations, num_of_sampled_actions=num_of_sampled_actions),
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
            type='sampled_alphazero',
            import_names=['lzero.policy.sampled_alphazero'],
        ),
        collector=dict(
            type='episode_alphazero',
            get_train_sample=False,
            import_names=['lzero.worker.alphazero_collector'],
        ),
        evaluator=dict(
            type='alphazero',
            import_names=['lzero.worker.alphazero_evaluator'],
        ),
    )
)

cfg = EasyDict(cfg)
