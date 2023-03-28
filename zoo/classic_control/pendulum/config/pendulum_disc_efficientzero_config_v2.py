import torch
from easydict import EasyDict

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
# collector_env_num = 8
# n_episode = 8
# evaluator_env_num = 3
# num_simulations = 50
# update_per_collect = 200
# batch_size = 256
# max_env_step = int(1e6)
# reanalyze_ratio = 0
collector_env_num = 2
n_episode = 2
evaluator_env_num = 2
num_simulations = 5
update_per_collect = 2
batch_size = 4
max_env_step = int(1e6)
reanalyze_ratio = 0
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

pendulum_disc_efficientzero_config = dict(
    exp_name=f'data_ez_ctree/pendulum_disc_efficientzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
    env=dict(
        stop_value=int(1e6),
        env_name='Pendulum-v1',
        continuous=False,
        discretization=True,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=(1, 3, 1),  # if frame_stack_num=1
            action_space_size=11,
            categorical_distribution=True,
            representation_network_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
            image_channel=1,
            frame_stack_num=1,
            downsample=False,
            # We use the small size model for pendulum.
            num_res_blocks=1,
            num_channels=16,
            lstm_hidden_size=128,
            support_scale=25,
            reward_support_size=51,
            value_support_size=51,
        ),
        device=device,
        env_type='not_board_games',
        game_segment_length=50,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        downsample=False,
        use_augmentation=False,
        policy_entropy_loss_weight=0,
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        lr_piecewise_constant_decay=True,
        optim_type='SGD',
        learning_rate=0.2,  # init lr for manually decay schedule
        n_episode=n_episode,
        eval_freq=int(2e3),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

pendulum_disc_efficientzero_config = EasyDict(pendulum_disc_efficientzero_config)
main_config = pendulum_disc_efficientzero_config

pendulum_disc_efficientzero_create_config = dict(
    env=dict(
        type='pendulum_lightzero',
        import_names=['zoo.classic_control.pendulum.envs.pendulum_lightzero_env'],
    ),
    # env_manager=dict(type='subprocess'),
    env_manager=dict(type='base'),
    policy=dict(
        type='efficientzero',
        import_names=['lzero.policy.efficientzero'],
    ),
    collector=dict(
        type='episode_muzero',
        import_names=['lzero.worker.muzero_collector'],
    )
)
pendulum_disc_efficientzero_create_config = EasyDict(pendulum_disc_efficientzero_create_config)
create_config = pendulum_disc_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero_with_gym_env
    train_muzero_with_gym_env([main_config, create_config], env_name=main_config.env.env_name, seed=0, max_env_step=max_env_step)