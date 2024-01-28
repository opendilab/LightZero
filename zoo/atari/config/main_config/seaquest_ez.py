from easydict import EasyDict

# options={'PongNoFrameskip-v4', 'QbertNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4', 'BreakoutNoFrameskip-v4', ...}
env_name = 'SeaquestNoFrameskip-v4'

if env_name == 'PongNoFrameskip-v4':
    action_space_size = 6
elif env_name == 'QbertNoFrameskip-v4':
    action_space_size = 6
elif env_name == 'MsPacmanNoFrameskip-v4':
    action_space_size = 9
elif env_name == 'SpaceInvadersNoFrameskip-v4':
    action_space_size = 6
elif env_name == 'BreakoutNoFrameskip-v4':
    action_space_size = 4
elif env_name == 'SeaquestNoFrameskip-v4':   
    action_space_size = 18

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
update_per_collect = None
batch_size = 256
model_update_ratio = 0.25
max_env_step = int(8e5)
reanalyze_ratio = 0.99



eps_greedy_exploration_in_collect = True
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_efficientzero_config = dict(
    exp_name=
    f'data_ez_ctree/{env_name[:-14]}/final_ez_seed1',
    env=dict(
        env_name=env_name,
        obs_shape=(4, 96, 96),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        # collect_max_episode_steps=5000.0,
        # eval_max_episode_steps=20000.0,
    ),
    policy=dict(
        model=dict(
            observation_shape=(4, 96, 96),
            frame_stack_num=4,
            action_space_size=action_space_size,
            downsample=True,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
        ),
        cuda=True,
        env_type='not_board_games',
        game_segment_length=400,
        random_collect_episode_num=0,
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            # need to dynamically adjust the number of decay steps according to the characteristics of the environment and the algorithm
            type='linear',
            start=1.,
            end=0.05,
            decay=int(1e5),
        ),
        use_augmentation=True,
        update_per_collect=update_per_collect,
        model_update_ratio=model_update_ratio,
        batch_size=batch_size,
        optim_type='SGD',
        lr_piecewise_constant_decay=True,
        learning_rate=0.2,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
atari_efficientzero_config = EasyDict(atari_efficientzero_config)
main_config = atari_efficientzero_config

atari_efficientzero_create_config = dict(
    env=dict(
        type='atari_lightzero',
        import_names=['zoo.atari.envs.atari_lightzero_env'],
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
atari_efficientzero_create_config = EasyDict(atari_efficientzero_create_config)
create_config = atari_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=1, max_env_step=max_env_step)
