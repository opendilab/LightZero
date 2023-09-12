from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 5
num_simulations = 5
update_per_collect = 10
batch_size = 4
max_env_step = int(2e6)
reanalyze_ratio = 0.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

tictactoe_muzero_config = dict(
    exp_name='data_mz_ctree/tictactoe_muzero_bot_mode_seed0',
    env=dict(
        battle_mode='play_with_bot_mode',
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        env_name="TicTacToe",
        mcts_mode='self_play_mode',  # only used in AlphaZero
        bot_action_type='v0',  # {'v0', 'alpha_beta_pruning'}
        agent_vs_human=False,
        prob_random_agent=0,
        prob_expert_agent=0,
        channel_last=True,
        scale=True,
        stop_value=1,
    ),
    policy=dict(
        sampled_algo=False,
        gumbel_algo=False,
        model=dict(
            observation_shape=(3, 3, 3),
            action_space_size=9,
            image_channel=3,
            # We use the small size model for tictactoe
            num_res_blocks=1,
            num_channels=16,
            frame_stack_num=1,
            model_type='conv',
            support_scale=10,
            reward_support_size=21,
            value_support_size=21,
            categorical_distribution=True,
        ),
        cuda=True,
        env_type='board_games',
        transform2string=False,
        gray_scale=False,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,  # lr for Adam optimizer
        grad_clip_value=0.5,
        manual_temperature_decay=True,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        game_segment_length=5,
        # NOTE：In board_games, we set large td_steps to make sure the value target is the final outcome.
        td_steps=9,
        num_unroll_steps=3,
        discount_factor=1,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(3e3),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        lstm_horizon_len=5,
        use_ture_chance_label_in_chance_encoder=False,
    ),
)
tictactoe_muzero_config = EasyDict(tictactoe_muzero_config)
main_config = tictactoe_muzero_config

tictactoe_muzero_create_config = dict(
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
tictactoe_muzero_create_config = EasyDict(tictactoe_muzero_create_config)
create_config = tictactoe_muzero_create_config
