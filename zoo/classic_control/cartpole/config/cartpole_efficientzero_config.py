import torch
from easydict import EasyDict

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

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
reanalyze_ratio = 0.
# debug config
# collector_env_num = 2
# n_episode = 2
# evaluator_env_num = 2
# num_simulations = 5
# update_per_collect = 2
# batch_size = 4
# max_env_step = int(1e6)
# reanalyze_ratio = 0
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

cartpole_efficientzero_config = dict(
    exp_name=f'data_ez_ctree/cartpole_efficientzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
    env=dict(
        env_name='CartPole-v0',
        continuous=False,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=(1, 4, 1),  # if frame_stack_num=1
            action_space_size=2,
            representation_network_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
            downsample=False,
            image_channel=1,
            frame_stack_num=1,
            # We use the small size model for cartpole.
            num_res_blocks=1,
            num_channels=16,
            lstm_hidden_size=128,
        ),
        device=device,
        env_type='not_board_games',
        game_segment_length=50,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        use_augmentation=False,
        ssl_loss_weight=2,
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

cartpole_efficientzero_config = EasyDict(cartpole_efficientzero_config)
main_config = cartpole_efficientzero_config

cartpole_efficientzero_create_config = dict(
    env=dict(
        type='cartpole_lightzero',
        import_names=['zoo.classic_control.cartpole.envs.cartpole_lightzero_env'],
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
cartpole_efficientzero_create_config = EasyDict(cartpole_efficientzero_create_config)
create_config = cartpole_efficientzero_create_config

if __name__ == "__main__":
    # Users can use different train entry by specifying the entry_type.
    entry_type = "train_muzero"  # options={"train_muzero", "train_muzero_with_gym_env"}

    if entry_type == "train_muzero":
        from lzero.entry import train_muzero
    elif entry_type == "train_muzero_with_gym_env":
        """
        The ``train_muzero_with_gym_env`` entry means that the environment used in the training process is generated by wrapping the original gym environment with LightZeroEnvWrapper.
        Users can refer to lzero/envs/wrappers for more details.
        """
        from lzero.entry import train_muzero_with_gym_env as train_muzero

    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)