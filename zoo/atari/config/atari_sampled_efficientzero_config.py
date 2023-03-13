import torch
from easydict import EasyDict

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# options={'PongNoFrameskip-v4', 'QbertNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4', 'BreakoutNoFrameskip-v4', ...}
env_name = 'SpaceInvadersNoFrameskip-v4'  

if env_name == 'PongNoFrameskip-v4':
    action_space_size = 6
    average_episode_length_when_converge = 2000
elif env_name == 'QbertNoFrameskip-v4':
    action_space_size = 6
    average_episode_length_when_converge = 2000
elif env_name == 'MsPacmanNoFrameskip-v4':
    action_space_size = 9
    average_episode_length_when_converge = 500
elif env_name == 'SpaceInvadersNoFrameskip-v4':
    action_space_size = 6
    average_episode_length_when_converge = 1000
elif env_name == 'BreakoutNoFrameskip-v4':
    action_space_size = 4
    average_episode_length_when_converge = 800
    
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
# only used for adjusting temperature/lr manually
threshold_env_steps_for_final_lr = int(1e6)
# if we set threshold_env_steps_for_final_temperature=0, i.e. we use the fixed final temperature=0.25.
threshold_env_steps_for_final_temperature = int(0)

continuous_action_space = False
K = 5  # num_of_sampled_actions
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
update_per_collect = 1000
batch_size = 256
max_env_step = int(1e6)
reanalyze_ratio = 0.3

## debug config
# continuous_action_space = False
# K = 3  # num_of_sampled_actions
# collector_env_num = 1
# n_episode = 1
# evaluator_env_num = 1
# num_simulations = 5
# update_per_collect = 2
# batch_size = 10
# max_env_step = int(1e4)
# reanalyze_ratio = 0.1

# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_sampled_efficientzero_config = dict(
    exp_name=f'data_sez_ctree/{env_name[:-14]}_sampled_efficientzero_k{K}_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
    env=dict(
        stop_value=int(1e6),
        env_name=env_name,
        obs_shape=(4, 96, 96),
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
        game_block_length=400,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        model=dict(
            observation_shape=(4, 96, 96),  # if frame_stack_num=4, gray_scale=True
            action_space_size=action_space_size,
            representation_network_type='conv_res_blocks',
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
        ),
        # learn_mode config
        learn=dict(
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            lr_piecewise_constant_decay=True,
            optim_type='SGD',
            learning_rate=0.2,  # init lr for manually decay schedule
            policy_loss_type='cross_entropy',  # options={'cross_entropy', 'KL'}
        ),
        collect=dict(n_episode=n_episode, ),  # Get "n_episode" episodes per collect.
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
atari_sampled_efficientzero_config = EasyDict(atari_sampled_efficientzero_config)
main_config = atari_sampled_efficientzero_config

atari_sampled_efficientzero_create_config = dict(
    env=dict(
        type='atari_lightzero',
        import_names=['zoo.atari.envs.atari_lightzero_env'],
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
atari_sampled_efficientzero_create_config = EasyDict(atari_sampled_efficientzero_create_config)
create_config = atari_sampled_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
