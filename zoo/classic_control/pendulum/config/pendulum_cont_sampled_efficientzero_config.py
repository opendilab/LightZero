import torch
from easydict import EasyDict

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
continuous_action_space = True
K = 20  # num_of_sampled_actions
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
update_per_collect = 200
batch_size = 256
max_env_step = int(1e6)
reanalyze_ratio = 0.3

## debug config
# continuous_action_space = True
# K = 3  # num_of_sampled_actions
# collector_env_num = 1
# n_episode = 1
# evaluator_env_num = 1
# num_simulations = 5
# update_per_collect = 2
# batch_size = 4
# max_env_step = int(1e4)
# reanalyze_ratio = 0.3

# only used for adjusting temperature/lr manually
average_episode_length_when_converge = 200
threshold_env_steps_for_final_lr = int(1e6)
threshold_env_steps_for_final_temperature = int(2e5)
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

pendulum_sampled_efficientzero_config = dict(
    exp_name=f'data_sez_ctree/pendulum_sampled_efficientzero_k{K}_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
    env=dict(
        stop_value=int(1e6),
        continuous=True,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        device=device,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        env_type='not_board_games',
        game_block_length=50,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        cvt_string=False,
        gray_scale=False,
        downsample=False,
        use_augmentation=False,
        # NOTE: for continuous gaussian policy, we use the policy_entropy_loss as in thee original Sampled MuZero paper.
        policy_entropy_loss_weight=5e-3,
        # NOTE: for vector input, we don't use the ssl loss.
        ssl_loss_weight=0,
        model=dict(
            observation_shape=(1, 3, 1),  # if frame_stack_num=1
            action_space_size=2,
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            categorical_distribution=True,
            image_channel=1,
            frame_stack_num=1,
            downsample=False,
            representation_model_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
            sigma_type='conditioned',  # options={'conditioned', 'fixed'}
            # ==============================================================
            # We use the small size model for pendulum.
            # ==============================================================
            num_res_blocks=1,
            num_channels=16,
            lstm_hidden_size=128,
            support_scale=25,
            reward_support_size=51,
            value_support_size=51,
        ),
        # learn_mode config
        learn=dict(
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            lr_manually=True,
            optim_type='SGD',
            learning_rate=0.2,  # init lr for manually decay schedule
            policy_loss_type='cross_entropy',  # options={'cross_entropy', 'KL'}
        ),
        # collect_mode config
        collect=dict(
            # Get "n_episode" episodes per collect.
            n_episode=n_episode,
        ),
        # If the eval cost is expensive, we could set eval_freq larger.
        eval=dict(evaluator=dict(eval_freq=int(2e3), )),
        # ``threshold_training_steps_for_final_lr`` is only used for adjusting lr manually.
        threshold_training_steps_for_final_lr=int(
            threshold_env_steps_for_final_lr / collector_env_num / average_episode_length_when_converge * update_per_collect),
        # ``threshold_training_steps_for_final_temperature`` is only used for adjusting temperature manually.
        threshold_training_steps_for_final_temperature=int(
            threshold_env_steps_for_final_temperature / collector_env_num / average_episode_length_when_converge * update_per_collect),
    ),
)
pendulum_sampled_efficientzero_config = EasyDict(pendulum_sampled_efficientzero_config)
main_config = pendulum_sampled_efficientzero_config

pendulum_sampled_efficientzero_create_config = dict(
    env=dict(
        type='pendulum',
        import_names=['zoo.classic_control.pendulum.envs.pendulum_lightzero_env'],
    ),
    # env_manager=dict(type='subprocess'),
    env_manager=dict(type='base'),
    policy=dict(
        type='sampled_efficientzero',
        import_names=['lzero.policy.sampled_efficientzero'],
    ),
    collector=dict(
        type='episode_sampled_efficientzero',
        get_train_sample=True,
        import_names=['lzero.worker.sampled_efficientzero_collector'],
    )
)
pendulum_sampled_efficientzero_create_config = EasyDict(pendulum_sampled_efficientzero_create_config)
create_config = pendulum_sampled_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
