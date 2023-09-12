from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 32
n_episode = 32
evaluator_env_num = 5
num_simulations = 50
update_per_collect = 50
reanalyze_ratio = 0.
batch_size = 256
max_env_step = int(1e6)

board_size = 6  # default_size is 15
bot_action_type = 'v0'  # options={'v0', 'v1'}
prob_random_action_in_bot = 0.5
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

gomoku_muzero_config = dict(
    exp_name=
    f'data_mz_ctree/gomoku_b{board_size}_rand{prob_random_action_in_bot}_muzero_bot-mode_type-{bot_action_type}_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
    env=dict(
        board_size=board_size,
        battle_mode='play_with_bot_mode',
        bot_action_type=bot_action_type,
        prob_random_action_in_bot=prob_random_action_in_bot,
        channel_last=True,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=(3, board_size, board_size),
            action_space_size=int(board_size * board_size),
            image_channel=3,
            num_res_blocks=1,
            num_channels=32,
            support_scale=10,
            reward_support_size=21,
            value_support_size=21,
        ),
        cuda=True,
        env_type='board_games',
        game_segment_length=int(board_size * board_size / 2),  # for battle_mode='play_with_bot_mode'
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        # NOTE：In board_games, we set large td_steps to make sure the value target is the final outcome.
        td_steps=int(board_size * board_size / 2),  # for battle_mode='play_with_bot_mode'
        # NOTE：In board_games, we set discount_factor=1.
        discount_factor=1,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e5),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
gomoku_muzero_config = EasyDict(gomoku_muzero_config)
main_config = gomoku_muzero_config

gomoku_muzero_create_config = dict(
    env=dict(
        type='gomoku',
        import_names=['zoo.board_games.gomoku.envs.gomoku_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)
gomoku_muzero_create_config = EasyDict(gomoku_muzero_create_config)
create_config = gomoku_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero

    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
