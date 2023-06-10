from easydict import EasyDict

# options={'Hopper-v3', 'HalfCheetah-v3', 'Walker2d-v3', 'Ant-v3', 'Humanoid-v3'}
env_name = 'Hopper-v3'
# env_name = 'Walker2d-v3'
# env_name = 'HalfCheetah-v3'
# env_name = 'Humanoid-v3'

if env_name == 'Hopper-v3':
    action_space_size = 3
    observation_shape = 11
elif env_name in ['HalfCheetah-v3', 'Walker2d-v3']:
    action_space_size = 6
    observation_shape = 17
elif env_name == 'Ant-v3':
    action_space_size = 8
    observation_shape = 111
elif env_name == 'Humanoid-v3':
    action_space_size = 17
    observation_shape = 376

ignore_done = False
if env_name == 'HalfCheetah-v3':
    ignore_done = True

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
seed = 0
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
continuous_action_space = True
K = 20  # num_of_sampled_actions
num_simulations = 50
update_per_collect = 200
batch_size = 256

# collector_env_num = 1
# n_episode = 1
# evaluator_env_num = 1
# continuous_action_space = True
# K = 20  # num_of_sampled_actions
# num_simulations = 5
# update_per_collect = 2
# batch_size = 3

max_env_step = int(5e6)
reanalyze_ratio = 0.
policy_entropy_loss_weight = 0.005
sample_fixed_extreme_action = True
fixed_actions_num = 8

# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

mujoco_sampled_efficientzero_config = dict(
    exp_name=
    f'data_sez_ctree/{env_name[:-3]}_sampled_efficientzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_bs-{batch_size}_pelw{policy_entropy_loss_weight}_sfea{sample_fixed_extreme_action}-{fixed_actions_num}_seed{seed}',
    env=dict(
        env_name=env_name,
        action_clip=True,
        continuous=True,
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
            lstm_hidden_size=256,
            latent_state_dim=256,
            self_supervised_learning_loss=True,
            res_connection_in_dynamics=True,
            norm_type='BN',
        ),
        mcts_ctree=True,
        action_tanh=True,
        sample_fixed_extreme_action=sample_fixed_extreme_action,
        fixed_actions_num=fixed_actions_num,

        # NOTE: for continuous gaussian policy, we use the policy_entropy_loss as in the original Sampled MuZero paper.
        policy_entropy_loss_weight=policy_entropy_loss_weight,
        eval_sample_action=False,
        eps=dict(
            eps_greedy_exploration_in_collect=False,
            type='exp',
            start=1.,
            end=0.05,
            decay=int(1e6),
        ),
        ignore_done=ignore_done,
        cuda=True,
        env_type='not_board_games',
        game_segment_length=200,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        discount_factor=0.997,

        optim_type='AdamW',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        grad_clip_value=0.5,

        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

mujoco_sampled_efficientzero_config = EasyDict(mujoco_sampled_efficientzero_config)
main_config = mujoco_sampled_efficientzero_config

mujoco_sampled_efficientzero_create_config = dict(
    env=dict(
        type='mujoco_lightzero',
        import_names=['zoo.mujoco.envs.mujoco_lightzero_env'],
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
mujoco_sampled_efficientzero_create_config = EasyDict(mujoco_sampled_efficientzero_create_config)
create_config = mujoco_sampled_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=seed, max_env_step=max_env_step)
