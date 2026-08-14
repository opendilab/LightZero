"""AlphaZero CTree preset for learning against the random Xiangqi bot."""

from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 4
n_episode = 8
num_simulations = 50
update_per_collect = 100
batch_size = 256
max_env_step = int(2e6)
mcts_ctree = True

chinese_chess_alphazero_config = dict(
    exp_name=(
        f'data_az_ctree/chinese_chess_alphazero_random-bot_ns{num_simulations}'
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
        prob_random_agent=0.0,
        prob_expert_agent=0.0,
        scale=True,
        alphazero_mcts_ctree=mcts_ctree,
        max_episode_steps=500,
    ),
    policy=dict(
        mcts_ctree=mcts_ctree,
        simulation_env_id='chinese_chess',
        simulation_env_config_type='play_with_bot',
        model=dict(
            observation_shape=(57, 10, 9),
            action_space_size=2086,
            num_res_blocks=6,
            num_channels=128,
            value_head_hidden_channels=[128],
            policy_head_hidden_channels=[256],
        ),
        cuda=True,
        board_size=10,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='AdamW',
        piecewise_decay_lr_scheduler=False,
        learning_rate=3e-4,
        grad_clip_value=0.5,
        value_weight=1.0,
        entropy_weight=1e-3,
        n_episode=n_episode,
        eval_freq=int(5e3),
        mcts=dict(num_simulations=num_simulations, max_moves=500),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
main_config = EasyDict(chinese_chess_alphazero_config)

create_config = EasyDict(
    dict(
        env=dict(
            type='chinese_chess',
            import_names=['zoo.board_games.chinese_chess.envs.chinese_chess_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(type='alphazero', import_names=['lzero.policy.alphazero']),
        collector=dict(type='episode_alphazero', import_names=['lzero.worker.alphazero_collector']),
        evaluator=dict(type='alphazero', import_names=['lzero.worker.alphazero_evaluator']),
    )
)

if __name__ == '__main__':
    from lzero.entry import train_alphazero

    train_alphazero([main_config, create_config], seed=0, max_env_step=max_env_step)
