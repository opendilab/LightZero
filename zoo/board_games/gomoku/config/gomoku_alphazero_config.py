import sys

# sys.path.append('/Users/yangzhenjie/code/jayyoung0802/LightZero/')
sys.path.append('/Users/puyuan/code/LightZero/')

import torch

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

from easydict import EasyDict

board_size = 6  # default_size is 15

collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
update_per_collect = 100
batch_size = 256

gomoku_alphazero_config = dict(
    exp_name='data_ez_ptree/gomoku_2pm_alphazero',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        channel_last=False,
        collect_max_episode_steps=int(1.08e4),
        eval_max_episode_steps=int(1.08e5),
        board_size=board_size,
        battle_mode='two_player_mode',
        prob_random_agent=0.,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        type='alphazero',
        env_name='gomoku',
        cuda=True,
        model=dict(
            input_channels=3,
            board_size=board_size,
        ),
        learn=dict(
            multi_gpu=False,
            batch_size=batch_size,
            learning_rate=0.001,
            weight_decay=0.0001,
            update_per_collect=update_per_collect,
            grad_norm=0.5,
            value_weight=1.0,
            optim_type='Adam',
            learner=dict(
                hook=dict(
                    load_ckpt_before_run='',
                    log_show_after_iter=1,
                    save_ckpt_after_iter=10000,
                    save_ckpt_after_run=True, ),
            )
        ),
        collect=dict(
            unroll_len=1,
            n_episode=n_episode,
            collector=dict(
                env=dict(
                    type='gomoku',
                    import_names=['zoo.board_games.gomoku.envs.gomoku_env'], ),
                augmentation=True,
            ),
            mcts=dict(num_simulations=num_simulations)
        ),
        eval=dict(
            evaluator=dict(
                n_episode=evaluator_env_num,
                eval_freq=int(100),
                stop_value=1,
                env=dict(
                    type='gomoku',
                    import_names=['zoo.board_games.gomoku.envs.gomoku_env'], ),
            ),
            mcts=dict(num_simulations=num_simulations)
        ),
        other=dict(
            replay_buffer=dict(
                replay_buffer_size=int(1e5),
                type='naive',
                save_episode=False,
                periodic_thruput_seconds=60,
            )
        ),
    ),
)

gomoku_alphazero_config = EasyDict(gomoku_alphazero_config)
main_config = gomoku_alphazero_config

gomoku_alphazero_create_config = dict(
    env=dict(
        type='gomoku',
        import_names=['zoo.board_games.gomoku.envs.gomoku_env'],
    ),
    # env_manager=dict(type='base'),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='alphazero',
        import_names=['core.policy.alphazero'],
    ),
    collector=dict(
        type='episode_alphazero',
        get_train_sample=False,
        # get_train_sample=True,
        import_names=['core.worker.collector.alphazero_collector'],
    ),
    evaluator=dict(
        type='alphazero',
        import_names=['core.worker.collector.alphazero_evaluator'],
    )

)
gomoku_alphazero_create_config = EasyDict(gomoku_alphazero_create_config)
create_config = gomoku_alphazero_create_config

if __name__ == '__main__':
    from core.entry import serial_pipeline_alphazero

    serial_pipeline_alphazero([main_config, create_config], seed=0)
