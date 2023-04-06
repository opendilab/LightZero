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
continuous_action_space = False
each_dim_disc_size = 4  # thus the total discrete action number is 4**4=256
K = 20  # num_of_sampled_actions
num_simulations = 50
update_per_collect = 200
batch_size = 256
max_env_step = int(10e6)
reanalyze_ratio = 0.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

bipedalwalker_cont_disc_sampled_efficientzero_config = dict(
    exp_name=f'data_sez_ctree/bipedalwalker_cont_disc_sampled_efficientzero_k{K}_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
    env=dict(
        stop_value=int(1e6),
        env_name='BipedalWalker-v3',
        continuous=True,
        manually_discretization=True,
        each_dim_disc_size=each_dim_disc_size,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=24,  # if frame_stack_num=1
            action_space_size=int(each_dim_disc_size ** 4),
            representation_network_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
            frame_stack_num=1,
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            sigma_type='conditioned',  # options={'conditioned', 'fixed'}
            # We use the medium size model for bipedalwalker_cont.
            num_res_blocks=1,
            num_channels=32,
            lstm_hidden_size=256,
            self_supervised_learning_loss=True,
        ),
        cuda=True,
        env_type='not_board_games',
        game_segment_length=200,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        ssl_loss_weight=2,
        # NOTE: for continuous gaussian policy, we use the policy_entropy_loss as in thee original Sampled MuZero paper.
        policy_entropy_loss_weight=5e-3,
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        lr_piecewise_constant_decay=True,
        optim_type='SGD',
        learning_rate=0.2,  # init lr for manually decay schedule
        grad_clip_value=0.5,  # NOTE: this parameter is important for stability.
        n_episode=n_episode,
        eval_freq=int(2e3),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
bipedalwalker_cont_disc_sampled_efficientzero_config = EasyDict(bipedalwalker_cont_disc_sampled_efficientzero_config)
main_config = bipedalwalker_cont_disc_sampled_efficientzero_config

bipedalwalker_cont_disc_sampled_efficientzero_create_config = dict(
    env=dict(
        type='bipedalwalker_cont_disc',
        import_names=['zoo.box2d.bipedalwalker.envs.bipedalwalker_cont_disc_env'],
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
bipedalwalker_cont_disc_sampled_efficientzero_create_config = EasyDict(bipedalwalker_cont_disc_sampled_efficientzero_create_config)
create_config = bipedalwalker_cont_disc_sampled_efficientzero_create_config

if __name__ == "__main__":
    # Users can use different train entry by specifying the entry_type.
    entry_type = "train_muzero_with_gym_env"  # options={"train_muzero", "train_muzero_with_gym_env"}

    if entry_type == "train_muzero":
        from lzero.entry import train_muzero
    elif entry_type == "train_muzero_with_gym_env":
        """
        The ``train_muzero_with_gym_env`` entry means that the environment used in the training process is generated by wrapping the original gym environment with LightZeroEnvWrapper.
        Users can refer to lzero/envs/wrappers for more details.
        """
        from lzero.entry import train_muzero_with_gym_env as train_muzero

    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)

