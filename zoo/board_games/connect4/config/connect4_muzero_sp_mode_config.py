from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 5
num_simulations = 50
update_per_collect = 50
reanalyze_ratio = 0.
batch_size = 256
max_env_step = int(5e5)
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

connect4_muzero_config = dict(
    exp_name=f'data_mz_ctree/connect4_self-play-mode_seed0',
    env=dict(
        battle_mode='self_play_mode',
        bot_action_type='rule',
        channel_last=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=(3, 6, 7),
            action_space_size=7,
            image_channel=3,
            num_res_blocks=1,
            num_channels=64,
            support_scale=300,
            reward_support_size=601,
            value_support_size=601,
        ),
        cuda=True,
        env_type='board_games',
        action_type='varied_action_space',
        game_segment_length=int(6 * 7),  # for battle_mode='self_play_mode'
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        # NOTE：In board_games, we set large td_steps to make sure the value target is the final outcome.
        td_steps=int(6 * 7),  # for battle_mode='self_play_mode'
        # NOTE：In board_games, we set discount_factor=1.
        discount_factor=1,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e5),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
connect4_muzero_config = EasyDict(connect4_muzero_config)
main_config = connect4_muzero_config

connect4_muzero_create_config = dict(
    env=dict(
        type='connect4',
        import_names=['zoo.board_games.connect4.envs.connect4_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)
connect4_muzero_create_config = EasyDict(connect4_muzero_create_config)
create_config = connect4_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero

    train_muzero([main_config, create_config], seed=1, max_env_step=max_env_step)