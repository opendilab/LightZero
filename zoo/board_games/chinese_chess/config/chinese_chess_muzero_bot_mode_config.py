"""MuZero CTree preset for learning against the random Xiangqi bot."""

from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 4
n_episode = 8
num_simulations = 50
update_per_collect = 100
batch_size = 256
max_env_step = int(2e6)

chinese_chess_muzero_config = dict(
    exp_name=(
        f'data_muzero/chinese_chess_muzero_random-bot_ctree_ns{num_simulations}'
        f'_upc{update_per_collect}_seed0'
    ),
    env=dict(
        battle_mode='play_with_bot_mode',
        bot_action_type='random',
        channel_last=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False),
        agent_vs_human=False,
        max_episode_steps=500,
    ),
    policy=dict(
        model=dict(
            observation_shape=(57, 10, 9),
            action_space_size=2086,
            image_channel=57,
            num_res_blocks=6,
            num_channels=128,
            reward_head_hidden_channels=[128],
            value_head_hidden_channels=[128],
            policy_head_hidden_channels=[256],
            reward_support_range=(-1.0, 2.0, 1.0),
            value_support_range=(-1.0, 2.0, 1.0),
            discrete_action_encoding_type='not_one_hot',
        ),
        cuda=True,
        env_type='board_games',
        action_type='varied_action_space',
        mcts_ctree=True,
        game_segment_length=128,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='AdamW',
        piecewise_decay_lr_scheduler=False,
        learning_rate=3e-4,
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=0.0,
        num_unroll_steps=5,
        td_steps=100,
        discount_factor=1.0,
        n_episode=n_episode,
        eval_freq=int(5e3),
        replay_buffer_size=int(2e5),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
main_config = EasyDict(chinese_chess_muzero_config)

create_config = EasyDict(
    dict(
        env=dict(
            type='chinese_chess',
            import_names=['zoo.board_games.chinese_chess.envs.chinese_chess_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(type='muzero', import_names=['lzero.policy.muzero']),
    )
)

if __name__ == '__main__':
    from lzero.entry import train_muzero

    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
