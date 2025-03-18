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
reanalyze_ratio = 0.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

cfg = dict(
    main_config=dict(
        exp_name='TicTacToe-play-with-bot-MuZero',
        seed=0,
        env=dict(
            env_id='TicTacToe-play-with-bot',
            battle_mode='play_with_bot_mode',
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
        ),
        policy=dict(
            model=dict(
                observation_shape=(3, 3, 3),
                action_space_size=9,
                image_channel=3,
                # We use the small size model for tictactoe.
                num_res_blocks=1,
                num_channels=16,
                reward_head_hidden_channels=[8],
                value_head_hidden_channels=[8],
                policy_head_hidden_channels=[8],
                support_scale=10,
                reward_support_size=21,
                value_support_size=21,
                norm_type='BN', 
            ),
            cuda=True,
            env_type='board_games',
            action_type='varied_action_space',
            game_segment_length=5,
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            optim_type='Adam',
            piecewise_decay_lr_scheduler=False,
            learning_rate=0.003,
            grad_clip_value=0.5,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            # NOTE：In board_games, we set large td_steps to make sure the value target is the final outcome.
            td_steps=9,
            num_unroll_steps=3,
            # NOTE：In board_games, we set discount_factor=1.
            discount_factor=1,
            n_episode=n_episode,
            eval_freq=int(2e3),
            replay_buffer_size=int(1e4),
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
            type='muzero',
            import_names=['lzero.policy.muzero'],
        ),
    )
)

cfg = EasyDict(cfg)
