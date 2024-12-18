from easydict import EasyDict

# The typical MiniGrid env id: {'MiniGrid-Empty-8x8-v0', 'MiniGrid-FourRooms-v0', 'MiniGrid-DoorKey-8x8-v0','MiniGrid-DoorKey-16x16-v0'},
# please refer to https://github.com/Farama-Foundation/MiniGrid for details.
env_id = 'MiniGrid-Empty-8x8-v0'
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

# key exploration related config
policy_entropy_weight = 0.
threshold_training_steps_for_final_temperature = int(5e5)
eps_greedy_exploration_in_collect = True
input_type = 'obs'  # options=['obs', 'latent_state', 'obs_latent_state']
target_model_for_intrinsic_reward_update_type = 'assign'  # 'assign' or 'momentum'

# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

minigrid_muzero_rnd_config = dict(
    exp_name=f'data_mz_rnd_ctree/{env_id}_muzero-rnd_ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}'
             f'_collect-eps-{eps_greedy_exploration_in_collect}_temp-final-steps-{threshold_training_steps_for_final_temperature}_pelw{policy_entropy_weight}'
             f'_rnd-rew-{input_type}-{target_model_for_intrinsic_reward_update_type}_seed{seed}',
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        continuous=False,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    reward_model=dict(
        type='rnd_muzero',
        intrinsic_reward_type='add',
        input_type=input_type,  # options=['obs', 'latent_state', 'obs_latent_state']
        # intrinsic_reward_weight means the relative weight of RND intrinsic_reward.
        # Specifically for sparse reward env MiniGrid, in this env, if we reach goal, the agent gets reward ~1, otherwise 0.
        # We could set the intrinsic_reward_weight approximately equal to the inverse of max_episode_steps.Please refer to rnd_reward_model for details.
        intrinsic_reward_weight=0.003,  # 1/300
        obs_shape=2835,
        latent_state_dim=512,
        hidden_size_list=[256, 256],
        learning_rate=3e-3,
        weight_decay=1e-4,
        batch_size=batch_size,
        update_per_collect=200,
        rnd_buffer_size=int(1e6),
        input_norm=True,
        input_norm_clamp_max=5,
        input_norm_clamp_min=-5,
        extrinsic_reward_norm=True,
        extrinsic_reward_norm_max=1,
    ),
    policy=dict(
        model=dict(
            observation_shape=2835,
            action_space_size=7,
            model_type='mlp',
            lstm_hidden_size=256,
            latent_state_dim=512,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            self_supervised_learning_loss=True,  # NOTE: default is False.
        ),
        use_rnd_model=True,
        # RND related config
        use_momentum_representation_network=True,
        target_model_for_intrinsic_reward_update_type=target_model_for_intrinsic_reward_update_type,
        target_update_freq_for_intrinsic_reward=1000,
        target_update_theta_for_intrinsic_reward=0.005,
        # key exploration related config
        policy_entropy_weight=policy_entropy_weight,
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            decay=int(2e5),
        ),
        manual_temperature_decay=True,
        threshold_training_steps_for_final_temperature=threshold_training_steps_for_final_temperature,

        cuda=True,
        env_type='not_board_games',
        game_segment_length=300,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.003,
        ssl_loss_weight=2,  # NOTE: default is 0.
        td_steps=td_steps,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e2),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

minigrid_muzero_rnd_config = EasyDict(minigrid_muzero_rnd_config)
main_config = minigrid_muzero_rnd_config

minigrid_muzero_create_config = dict(
    env=dict(
        type='minigrid_lightzero',
        import_names=['zoo.minigrid.envs.minigrid_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
    collector=dict(
        type='episode_muzero',
        import_names=['lzero.worker.muzero_collector'],
    )
)
minigrid_muzero_create_config = EasyDict(minigrid_muzero_create_config)
create_config = minigrid_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero_with_reward_model
    train_muzero_with_reward_model([main_config, create_config], seed=seed, max_env_step=max_env_step)