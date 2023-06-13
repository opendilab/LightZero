from easydict import EasyDict

# options={'memory_len/0', 'memory_len/9', 'memory_len/17', 'memory_len/20', 'memory_len/22', 'memory_size/0', 'bsuite_swingup/0', 'bandit_noise/0'}
env_name = 'memory_len/9'


if env_name in ['memory_len/0', 'memory_len/9', 'memory_len/17', 'memory_len/20', 'memory_len/22']:
    # memory_length=1, 10, 50, 80, 100
    action_space_size = 2
    observation_shape = 3
elif env_name in ['bsuite_swingup/0']:
    action_space_size = 3
    observation_shape = 8
elif env_name == 'bandit_noise/0':
    action_space_size = 11
    observation_shape = 1
elif env_name in ['memory_size/0']:
    action_space_size = 2
    observation_shape = 3

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
seed = 0
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
continuous_action_space = False
K = 2  # num_of_sampled_actions
num_simulations = 50
update_per_collect = 100
batch_size = 256
max_env_step = int(5e5)
reanalyze_ratio = 0.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

bsuite_sampled_efficientzero_config = dict(
    exp_name=
    f'data_sez_ctree/bsuite_sampled_efficientzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed{seed}',
    env=dict(
        env_name=env_name,
        stop_value=int(1e6),
        continuous=False,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=observation_shape,
            action_space_size=action_space_size,
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            model_type='mlp', 
            lstm_hidden_size=128,
            latent_state_dim=128,
            discrete_action_encoding_type='one_hot',
            norm_type='BN', 
        ),
        # policy_entropy_loss_weight=0.005,
        cuda=True,
        env_type='not_board_games',
        game_segment_length=50,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e2),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

bsuite_sampled_efficientzero_config = EasyDict(bsuite_sampled_efficientzero_config)
main_config = bsuite_sampled_efficientzero_config

bsuite_sampled_efficientzero_create_config = dict(
    env=dict(
        type='bsuite_lightzero',
        import_names=['zoo.bsuite.envs.bsuite_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sampled_efficientzero',
        import_names=['lzero.policy.sampled_efficientzero'],
    ),
    collector=dict(
        type='episode_muzero',
        import_names=['lzero.worker.muzero_collector'],
    )
)
bsuite_sampled_efficientzero_create_config = EasyDict(bsuite_sampled_efficientzero_create_config)
create_config = bsuite_sampled_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=seed, max_env_step=max_env_step)
