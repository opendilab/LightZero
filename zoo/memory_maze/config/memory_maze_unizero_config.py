from easydict import EasyDict

# Environment ID and task-specific parameters
env_id = 'memory_maze:MemoryMaze-9x9-v0'  # The name of the environment
memory_length = 1000  # Length of memory for the agent to store

# memory_length = 10  # TODO: DEBUG


# env_id = 'memory_maze:MemoryMaze-11x11-v0'  # The name of the environment
# memory_length = 2000  # Length of memory for the agent to store
# env_id = 'memory_maze:MemoryMaze-13x13-v0'  # The name of the environment
# memory_length = 3000  # Length of memory for the agent to store
# env_id = 'memory_maze:MemoryMaze-15x15-v0'  # The name of the environment
# memory_length = 4000  # Length of memory for the agent to store

max_env_step = int(10e6)  # Maximum number of environment steps
# embed_dim = 256  # Embedding dimension for the model
# num_layers = 8  # Number of layers in the model
# num_heads = 8  # Number of heads in the attention mechanism

embed_dim = 512  # Embedding dimension for the model
num_layers = 4  # Number of layers in the model
num_heads = 4  # Number of heads in the attention mechanism


# Unroll steps and game segment length for the training process
num_unroll_steps = memory_length
game_segment_length = memory_length
collector_env_num = 8  # Number of collector environments
n_episode = 8  # Number of episodes per collection
evaluator_env_num = 10  # Number of evaluator environments

# Simulation and replay buffer settings
num_simulations = 50  # Number of simulations for MCTS
update_per_collect = 50  # Number of updates per data collection
replay_ratio = 0.1  # Ratio of replay buffer usage
batch_size = 64  # Batch size for training
reanalyze_ratio = 0  # Ratio for reanalyzing the replay buffer
td_steps = game_segment_length  # Temporal difference steps for value estimation

# ========= only for debug ===========
# collector_env_num = 2
# n_episode = 2
# evaluator_env_num = 2
# num_simulations = 1
# update_per_collect = 1
# replay_ratio = 0.25
# batch_size = 4

# Main configuration dictionary for the memory maze environment
memory_maze_unizero_config = dict(
    # Experiment name for logging and saving
    env=dict(
        stop_value=int(1e6),  # Stop training when this value is reached
        env_id=env_id,  # Environment ID
        flatten_observation=False,  # Whether to flatten the observation
        collector_env_num=collector_env_num,  # Number of collector envs
        evaluator_env_num=evaluator_env_num,  # Number of evaluator envs
        n_evaluator_episode=evaluator_env_num,  # Number of evaluation episodes
        manager=dict(shared_memory=False, ),  # Memory management settings
        # max_steps=10, # TODO: DEBUG
        max_steps=memory_length,  # The default maximum number of steps per episode
        # save_replay=True,
    ),
    # Policy configuration for the model
    policy=dict(
        learn=dict(
            learner=dict(
                hook=dict(save_ckpt_after_iter=1000000, ),  # Save checkpoint after 1M iterations
            ),
        ),
        sample_type='episode',  # Sampling type for memory environments
        model=dict(
            observation_shape=(3, 64, 64),  # Observation shape for the environment
            action_space_size=6,  # Number of possible actions
            world_model_cfg=dict(
                max_blocks=num_unroll_steps + 5,  # Maximum number of blocks
                max_tokens=2 * (num_unroll_steps + 5),  # Maximum number of tokens
                context_length=2 * (num_unroll_steps + 5),  # Context length for memory
                device='cuda',  # Use GPU for training
                action_space_size=6,  # Action space size
                num_layers=num_layers,  # Number of layers in the model
                num_heads=num_heads,  # Number of attention heads
                embed_dim=embed_dim,  # Embedding dimension
                env_num=max(collector_env_num, evaluator_env_num),  # Number of envs
                obs_type='image_memory_maze',  # Observation type
                policy_entropy_weight=5e-2,  # TODO: Entropy weight for policy regularization
            ),
        ),
        # Path for loading a pre-trained model (if any)
        model_path=None,
        num_unroll_steps=num_unroll_steps,  # Number of unroll steps for MCTS
        td_steps=td_steps,  # Temporal difference steps
        discount_factor=0.99,  # Discount factor for future rewards
        game_segment_length=game_segment_length,  # Length of each game segment
        replay_ratio=replay_ratio,  # Replay ratio for replay buffer
        update_per_collect=update_per_collect,  # Number of updates per data collection
        batch_size=batch_size,  # Batch size for training
        optim_type='AdamW',  # Optimizer type
        learning_rate=1e-4,  # Learning rate
        num_simulations=num_simulations,  # Number of simulations for MCTS
        reanalyze_ratio=reanalyze_ratio,  # Ratio for reanalyzing the replay buffer
        n_episode=n_episode,  # Number of episodes per collection
        eval_freq=int(5e3),  # Evaluation frequency
        replay_buffer_size=int(1e6),  # Size of the replay buffer
        # replay_buffer_size=int(1e7),  # Size of the replay buffer
        collector_env_num=collector_env_num,  # Number of collector environments
        evaluator_env_num=evaluator_env_num,  # Number of evaluator environments
    ),
)

# Convert the dictionary to EasyDict for more convenient attribute access
memory_maze_unizero_config = EasyDict(memory_maze_unizero_config)
main_config = memory_maze_unizero_config

# Configuration for creating the environment and policy
memory_maze_unizero_create_config = dict(
    env=dict(
        type='memory_maze_lightzero',  # Type of environment
        import_names=['zoo.memory_maze.envs.memory_maze_lightzero_env'],  # Import path for the environment
    ),
    # env_manager=dict(type='subprocess'),  # Use subprocesses to manage envs
    env_manager=dict(type='base'),  # Use subprocesses to manage envs
    policy=dict(
        type='unizero',  # Type of policy
        import_names=['lzero.policy.unizero'],  # Import path for the policy
    ),
)
memory_maze_unizero_create_config = EasyDict(memory_maze_unizero_create_config)
create_config = memory_maze_unizero_create_config

# Main function for training
if __name__ == "__main__":
    seeds = [1]  # List of seeds for multiple experiments
    for seed in seeds:
        # Define the experiment name based on the configuration parameters
        main_config.exp_name = f'data_{env_id}/{env_id}_td{td_steps}_layer{num_layers}-head{num_heads}_unizero_edim{embed_dim}_H{num_unroll_steps}_bs{batch_size}_upc{update_per_collect}_seed{seed}'
        # Import the training function and start training
        from lzero.entry import train_unizero
        train_unizero([main_config, create_config], seed=seed, model_path=main_config.policy.model_path,
                      max_env_step=max_env_step)