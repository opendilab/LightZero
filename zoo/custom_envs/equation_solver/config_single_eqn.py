from my_train_alphazero import my_train_alphazero
from easydict import EasyDict

# ==============================================================
# Frequently changed config specified by the user (lightweight settings)
# ==============================================================
collector_env_num = 4         # Number of parallel environments for data collection
n_episode = 4                 # Number of episodes per training iteration
evaluator_env_num = 1         # Number of evaluator environments
num_simulations = 50          # MCTS simulations per move (try increasing if needed)
update_per_collect = 100      # Number of gradient updates per data collection cycle
batch_size = 32               # Mini-batch size for training
max_env_step = int(1e3)       # Maximum total environment steps for a quick run
model_path = None
mcts_ctree = False

# ==============================================================
# Configurations for singleEqn_env (lightweight version)
# ==============================================================
singleEqn_alphazero_config = dict(
    exp_name='data_alphazero/singleEqn/x+b/',
    env=dict(
        battle_mode='play_with_bot_mode',
        battle_mode_in_simulation_env='self_play_mode',  # For simulation during MCTS
        channel_last=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False),
        agent_vs_human=False,
        prob_random_agent=0,
        prob_expert_agent=0,
        prob_random_action_in_bot=0,
        scale=True,
        render_mode=None,
        replay_path=None,
        alphazero_mcts_ctree=mcts_ctree,
    ),
    policy=dict(
        mcts_ctree=mcts_ctree,
        simulation_env_id='singleEqn_env',  # Must match the registered name of your environment
        model=dict(
            type='AlphaZeroMLPModel',
            import_names=['zoo.custom_envs.equation_solver.my_alphazero_mlp_model'],
            observation_shape=(41,),        # Flat vector of length 41
            action_space_size=50,             # Matches your environment's action_dim
            hidden_sizes=[64, 64],          # MLP hidden layer sizes
        ),
        cuda=True,
        env_type='not_board_games',
        action_type='varied_action_space',
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        # learning_rate=0.003,
        learning_rate=3e-4,
        grad_clip_value=0.5,
        value_weight=1.0,
        entropy_weight=0.0,
        n_episode=n_episode,
        eval_freq=int(2e3),
        mcts=dict(num_simulations=num_simulations),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        other=dict(
            replay_buffer=dict(
                type='advanced',              # Use advanced (or prioritized) replay buffer
                replay_buffer_size=10000,       # Set a smaller buffer for lightweight runs
                sample_min_limit_ratio=0.25,      # Allow sampling even if only 50% of batch size is available.
                alpha=0.6,
                beta=0.4,
                anneal_step=100000,
                enable_track_used_data=False,
                deepcopy=False,
                save_episode=False,
            )
        ),
    ),
)
singleEqn_alphazero_config = EasyDict(singleEqn_alphazero_config)
main_config = singleEqn_alphazero_config

singleEqn_alphazero_create_config = dict(
    env=dict(
        type='singleEqn_env',
        import_names=['zoo.custom_envs.equation_solver.env_single_eqn'],  # Adjust this path if needed
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='MyAlphaZeroPolicy',  # Your custom policy subclass
        import_names=['zoo.custom_envs.equation_solver.my_alphazero_policy'],
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
singleEqn_alphazero_create_config = EasyDict(singleEqn_alphazero_create_config)
create_config = singleEqn_alphazero_create_config

if __name__ == '__main__':
    from lzero.entry import train_alphazero
    # Merge the environment configuration into the policy config.
    main_config.policy.env = main_config.env
    my_train_alphazero([main_config, create_config], seed=0, model_path=model_path, max_env_step=max_env_step)
