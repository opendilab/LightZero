from easydict import EasyDict

env_id = 'key_to_door'  # The name of the environment, options: 'visual_match', 'key_to_door'
memory_length = 30

max_env_step = int(1e6)

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
seed = 0
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
update_per_collect = 200
batch_size = 256
reanalyze_ratio = 0
td_steps = 5
policy_entropy_weight = 0.
threshold_training_steps_for_final_temperature = int(5e5)
eps_greedy_exploration_in_collect = False
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

memory_efficientzero_config = dict(
    exp_name=f'data_ez/{env_id}_memlen-{memory_length}_efficientzero_ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}_seed{seed}',
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        flate_observation=True,  # Whether to flatten the observation
        max_frames={
            "explore": 15,
            "distractor": memory_length,
            "reward": 15
        },  # Maximum frames per phase
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=25,
            action_space_size=4,
            model_type='mlp', 
            lstm_hidden_size=128,
            latent_state_dim=128,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
        ),
        policy_entropy_weight=policy_entropy_weight,
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            decay=int(2e5),
        ),
        td_steps=td_steps,
        manual_temperature_decay=True,
        threshold_training_steps_for_final_temperature=threshold_training_steps_for_final_temperature,
        cuda=True,
        env_type='not_board_games',
        game_segment_length=60,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

memory_efficientzero_config = EasyDict(memory_efficientzero_config)
main_config = memory_efficientzero_config

memory_efficientzero_create_config = dict(
    env=dict(
        type='memory_lightzero',
        import_names=['zoo.memory.envs.memory_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='efficientzero',
        import_names=['lzero.policy.efficientzero'],
    ),
    collector=dict(
        type='episode_muzero',
        import_names=['lzero.worker.muzero_collector'],
    )
)
memory_efficientzero_create_config = EasyDict(memory_efficientzero_create_config)
create_config = memory_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=seed, max_env_step=max_env_step)
