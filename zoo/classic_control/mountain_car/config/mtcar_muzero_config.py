from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 25
update_per_collect = 100
batch_size = 256
max_env_step = int(1e6)
reanalyze_ratio = 0
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

mountain_car_muzero_config = dict(
    exp_name=f'mountain_car_muzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
    env=dict(
        env_name='MountainCar-v0',
        continuous=False,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        eps=dict(
            # (bool) Whether to use eps greedy exploration in collecting data.
            eps_greedy_exploration_in_collect=True,
            # (str) The type of decaying epsilon. Options are 'linear', 'exp'.
            type='linear',
            # (float) The start value of eps.
            start=1.,
            # (float) The end value of eps.
            end=0.05,
            # (int) The decay steps from start to end eps.
            decay=int(1e5),
        ),
        model=dict(
            observation_shape=2,
            action_space_size=3,
            model_type='mlp', 
            lstm_hidden_size=128,
            latent_state_dim=64,
            self_supervised_learning_loss=True,  # NOTE: default is False.
            discrete_action_encoding_type='one_hot',
            norm_type='BN', 
        ),
        cuda=True,
        env_type='not_board_games',
        # action_type='varied_action_space',
        game_segment_length=50,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
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

mountain_car_muzero_config = EasyDict(mountain_car_muzero_config)
main_config = mountain_car_muzero_config

mountain_car_muzero_create_config = dict(
    env=dict(
        type='mountain_car_lightzero',
        import_names=['zoo.classic_control.mountain_car.envs.mtcar_lightzero_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)
mountain_car_muzero_create_config = EasyDict(mountain_car_muzero_create_config)
create_config = mountain_car_muzero_create_config

if __name__ == "__main__":
    # Users can use different train entry by specifying the entry_type.
    from lzero.entry import train_muzero

    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)