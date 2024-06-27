from easydict import EasyDict
# import torch
# torch.cuda.set_device(0)
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 5
num_simulations = 50
update_per_collect = 50
batch_size = 256
max_env_step = int(1e6)

reuse_search = True
collect_with_pure_policy = True
use_priority = False
buffer_reanalyze_freq = 1
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

connect4_muzero_config = dict(
    exp_name=f'data_rezero_mz/connect4_muzero_bot-mode_ns{num_simulations}_upc{update_per_collect}_brf{buffer_reanalyze_freq}_seed0',
    env=dict(
        battle_mode='play_with_bot_mode',
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
        game_segment_length=int(6 * 7 / 2),  # for battle_mode='play_with_bot_mode'
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=0,
        # NOTE：In board_games, we set large td_steps to make sure the value target is the final outcome.
        td_steps=int(6 * 7 / 2),  # for battle_mode='play_with_bot_mode'
        # NOTE：In board_games, we set discount_factor=1.
        discount_factor=1,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e5),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        reanalyze_noise=True,
        reuse_search=reuse_search,
        collect_with_pure_policy=collect_with_pure_policy,
        use_priority=use_priority,
        buffer_reanalyze_freq=buffer_reanalyze_freq,
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
    # Define a list of seeds for multiple runs
    seeds = [0, 1, 2]  # You can add more seed values here
    for seed in seeds:
        # Update exp_name to include the current seed
        main_config.exp_name = f'data_rezero_mz/connect4_muzero_bot-mode_ns{num_simulations}_upc{update_per_collect}_brf{buffer_reanalyze_freq}_seed{seed}'
        from lzero.entry import train_rezero
        train_rezero([main_config, create_config], seed=seed, max_env_step=max_env_step)