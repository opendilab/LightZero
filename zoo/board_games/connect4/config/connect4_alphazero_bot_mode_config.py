from easydict import EasyDict

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
model_path = None
mcts_ctree = False

# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================
connect4_alphazero_config = dict(
    exp_name='data_az_ptree/connect4_bot-mode_seed0',
    env=dict(
        battle_mode='play_with_bot_mode',
        bot_action_type='rule',
        channel_last=False,
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
        screen_scaling=9,
        render_mode=None,
        replay_path=None,
        alphazero_mcts_ctree=mcts_ctree,
        # ==============================================================
    ),
    policy=dict(
        mcts_ctree=mcts_ctree,
        # ==============================================================
        # for the creation of simulation env
        simulation_env_name='connect4',
        simulation_env_config_type='play_with_bot',
        # ==============================================================
        model=dict(
            observation_shape=(3, 6, 7),
            action_space_size=7,
            num_res_blocks=1,
            num_channels=64,
        ),
        cuda=True,
        env_type='board_games',
        action_type='varied_action_space',
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        grad_clip_value=0.5,
        value_weight=1.0,
        entropy_weight=0.0,
        n_episode=n_episode,
        eval_freq=int(2e3),
        mcts=dict(num_simulations=num_simulations),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

connect4_alphazero_config = EasyDict(connect4_alphazero_config)
main_config = connect4_alphazero_config

connect4_alphazero_create_config = dict(
    env=dict(
        type='connect4',
        import_names=['zoo.board_games.connect4.envs.connect4_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='alphazero',
        import_names=['lzero.policy.alphazero'],
    ),
    collector=dict(
        type='episode_alphazero',
        import_names=['lzero.worker.alphazero_collector'],
    ),
    evaluator=dict(
        type='alphazero',
        import_names=['lzero.worker.alphazero_evaluator'],
    )
)
connect4_alphazero_create_config = EasyDict(connect4_alphazero_create_config)
create_config = connect4_alphazero_create_config

if __name__ == '__main__':
    from lzero.entry import train_alphazero
    train_alphazero([main_config, create_config], seed=0, model_path=model_path, max_env_step=max_env_step)
