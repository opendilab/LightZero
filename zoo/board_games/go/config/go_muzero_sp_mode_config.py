from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
board_size = 6
collector_env_num = 8
n_episode = 8
evaluator_env_num = 5
update_per_collect = 50
batch_size = 256
max_env_step = int(1e6)
prob_random_action_in_bot = 1
reanalyze_ratio = 0

if board_size == 19:
    num_simulations = 800
elif board_size == 9:
    num_simulations = 180
elif board_size == 6:
    num_simulations = 80

# board_size = 6
# collector_env_num = 1
# n_episode = 1
# evaluator_env_num = 1
# num_simulations = 2
# update_per_collect = 2
# batch_size = 2
# max_env_step = int(5e5)
# prob_random_action_in_bot = 0.
# reanalyze_ratio = 0

# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

go_muzero_config = dict(
    exp_name=
    f'data_mz_ctree/go_muzero_sp-mode_rand{prob_random_action_in_bot}_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
    env=dict(
        board_size=board_size,
        komi=7.5,
        battle_mode='self_play_mode',
        bot_action_type='v0',
        prob_random_action_in_bot=prob_random_action_in_bot,
        channel_last=True,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=(17, board_size, board_size),
            action_space_size=int(1 * board_size * board_size + 1),
            image_channel=17,
            num_res_blocks=1,
            num_channels=64,
        ),
        cuda=True,
        env_type='board_games',
        # game_segment_length=int(board_size * board_size),  # for battle_mode='self_play_mode'
        game_segment_length=100,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='AdamW',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        grad_clip_value=10,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        # NOTE：In board_games, we set large td_steps to make sure the value target is the final outcome.
        # td_steps=int(board_size * board_size),  # for battle_mode='self_play_mode'
        td_steps=100,
        # NOTE：In board_games, we set discount_factor=1.
        discount_factor=1,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
go_muzero_config = EasyDict(go_muzero_config)
main_config = go_muzero_config

go_muzero_create_config = dict(
    env=dict(
        type='go_lightzero',
        import_names=['zoo.board_games.go.envs.go_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
    collector=dict(
        type='episode_muzero',
        get_train_sample=True,
        import_names=['lzero.worker.muzero_collector'],
    )
)
go_muzero_create_config = EasyDict(go_muzero_create_config)
create_config = go_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
