from easydict import EasyDict

# env_name = 'MiniGrid-Empty-8x8-v0'
# max_env_step = int(1e6)

env_name = 'MiniGrid-FourRooms-v0'
max_env_step = int(2e6)

# typical MiniGrid env id: {'MiniGrid-Empty-8x8-v0', 'MiniGrid-FourRooms-v0', 'MiniGrid-DoorKey-8x8-v0','MiniGrid-DoorKey-16x16-v0'},
# please refer to https://github.com/Farama-Foundation/MiniGrid for details.

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

# collector_env_num = 1
# n_episode = 1
# evaluator_env_num = 1
# num_simulations = 3
# update_per_collect = 2
# batch_size = 4

reanalyze_ratio = 0
random_collect_episode_num = 0
init_temperature_value_for_decay = 1.0
td_steps = 5
eval_sample_action = False

policy_entropy_loss_weight = 0.
# policy_entropy_loss_weight = 0.005
threshold_training_steps_for_final_temperature = int(5e4)
# threshold_training_steps_for_final_temperature = int(5e5)
eps_greedy_exploration_in_collect = False
# eps_greedy_exploration_in_collect = True

# input_type = 'latent_state'  # 'obs' or 'latent_state'
input_type = 'obs'  # 'obs' or 'latent_state'

# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

minigrid_muzero_config = dict(
    exp_name=f'data_mz_ctree/{env_name}_muzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_pelw{policy_entropy_loss_weight}_temp-final-steps-{threshold_training_steps_for_final_temperature}_collect-eps-{eps_greedy_exploration_in_collect}_rnd-rew-{input_type}_rndbs1e6_gsl300_seed{seed}',
    env=dict(
        stop_value=int(1e6),
        env_name=env_name,
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
        input_type=input_type,  # 'obs' or 'latent_state'
        # input_type='obs', # 'obs' or 'latent_state'
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
        update_per_collect=100,
        # rnd_buffer_size=int(1e5),
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
        eval_sample_action=eval_sample_action,
        policy_entropy_loss_weight=policy_entropy_loss_weight,
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            type='linear',
            start=1.,
            end=0.05,
            decay=int(2e5),
        ),
        ignore_done=False,

        random_collect_episode_num=random_collect_episode_num,
        td_steps=td_steps,
        manual_temperature_decay=True,
        # To make init policy be more random in sparse reward env.
        init_temperature_value_for_decay=init_temperature_value_for_decay,
        threshold_training_steps_for_final_temperature=threshold_training_steps_for_final_temperature,
        # TODO: test the effect
        # use_max_priority_for_new_data=False,
        # priority_prob_alpha=1,
        use_max_priority_for_new_data=True,
        priority_prob_alpha=0.6,
        cuda=True,
        env_type='not_board_games',
        # game_segment_length=50,
        game_segment_length=300,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='AdamW',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        ssl_loss_weight=2,  # NOTE: default is 0.
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e2),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

minigrid_muzero_config = EasyDict(minigrid_muzero_config)
main_config = minigrid_muzero_config

minigrid_muzero_create_config = dict(
    env=dict(
        type='minigrid_lightzero',
        import_names=['zoo.minigrid.envs.minigrid_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero_rnd',
        import_names=['lzero.policy.muzero_rnd'],
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