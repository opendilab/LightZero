from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
continuous_action_space = True
K = 20
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
update_per_collect = 200
batch_size = 64
max_env_step = int(1e6)
reanalyze_ratio = 0.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

metadrive_sampled_efficientzero_config = dict(
    exp_name=
    f'data_sez_ctree/sez_metadrive_old{K}_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
    env=dict(
        env_name='MetaDrive',
        continuous=True,
        obs_shape = [5, 84, 84],
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        metadrive=dict(
            use_render=False,
            traffic_density=0.20,  # Density of vehicles occupying the roads, range in [0,1]
            map='XSOS',  # Int or string: an easy way to fill map_config
            horizon=4000,  # Max step number
            driving_reward=1.0,  # Reward to encourage agent to move forward.
            speed_reward=0.1,  # Reward to encourage agent to drive at a high speed
            use_lateral_reward=False,  # reward for lane keeping
            out_of_road_penalty=40.0,  # Penalty to discourage driving out of road
            crash_vehicle_penalty=40.0,  # Penalty to discourage collision
            decision_repeat=10,  # Reciprocal of decision frequency
            out_of_route_done=True,  # Game over if driving out of road
        ),
    
    ),
    policy=dict(
        model=dict(
            observation_shape=[5, 84, 84],
            action_space_size=2,
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            sigma_type='conditioned',
            model_type='conv',  # options={'mlp', 'conv'}
            lstm_hidden_size=128,
            latent_state_dim=128,
            downsample = True,
            image_channel=5,
        ),
        cuda=True,
        env_type='not_board_games',
        game_segment_length=50,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.003,
        # NOTE: for continuous gaussian policy, we use the policy_entropy_loss as in the original Sampled MuZero paper.
        policy_entropy_weight=5e-3,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2000),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
metadrive_sampled_efficientzero_config = EasyDict(metadrive_sampled_efficientzero_config)
main_config = metadrive_sampled_efficientzero_config

metadrive_sampled_efficientzero_create_config = dict(
    env=dict(
        type='metadrive_lightzero',
        import_names=['zoo.metadrive.env.metadrive_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sampled_efficientzero',
        import_names=['lzero.policy.sampled_efficientzero'],
    ),
    collector=dict(
        type='episode_muzero',
        get_train_sample=True,
        import_names=['lzero.worker.muzero_collector'],
    )
)
metadrive_sampled_efficientzero_create_config = EasyDict(metadrive_sampled_efficientzero_create_config)
create_config = metadrive_sampled_efficientzero_create_config
if __name__ == "__main__":

    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
