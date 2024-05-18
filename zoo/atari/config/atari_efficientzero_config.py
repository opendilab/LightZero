from easydict import EasyDict
import torch
torch.cuda.set_device(5)
# options={'PongNoFrameskip-v4', 'QbertNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4', 'BreakoutNoFrameskip-v4', ...}
# env_id = 'PongNoFrameskip-v4'
env_id = 'FrostbiteNoFrameskip-v4'
update_per_collect = None
model_update_ratio = 0.25

if env_id == 'AlienNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'AmidarNoFrameskip-v4':
    action_space_size = 10
elif env_id == 'AssaultNoFrameskip-v4':
    action_space_size = 7
elif env_id == 'AsterixNoFrameskip-v4':
    action_space_size = 9
elif env_id == 'BankHeistNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'BattleZoneNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'ChopperCommandNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'CrazyClimberNoFrameskip-v4':
    action_space_size = 9
elif env_id == 'DemonAttackNoFrameskip-v4':
    action_space_size = 6
    model_update_ratio = 0.25
elif env_id == 'FreewayNoFrameskip-v4':
    action_space_size = 3
    model_update_ratio = 0.25
elif env_id == 'FrostbiteNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'GopherNoFrameskip-v4':
    action_space_size = 8
elif env_id == 'HeroNoFrameskip-v4':
    action_space_size = 18
    model_update_ratio = 0.25
elif env_id == 'JamesbondNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'KangarooNoFrameskip-v4':
    action_space_size = 18
elif env_id ==  'KrullNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'KungFuMasterNoFrameskip-v4':
    action_space_size = 14
elif env_id == 'PrivateEyeNoFrameskip-v4':
    action_space_size = 18
    # model_update_ratio = 0.25
    model_update_ratio = 0.5
elif env_id == 'RoadRunnerNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'UpNDownNoFrameskip-v4':
    action_space_size = 6
elif env_id == 'PongNoFrameskip-v4':
    action_space_size = 6
    model_update_ratio = 0.25
elif env_id == 'MsPacmanNoFrameskip-v4':
    action_space_size = 9
elif env_id == 'QbertNoFrameskip-v4':
    action_space_size = 6
elif env_id == 'SeaquestNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'BoxingNoFrameskip-v4':
    action_space_size = 18
    model_update_ratio = 0.25
elif env_id == 'BreakoutNoFrameskip-v4':
    action_space_size = 4

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
# update_per_collect = 1000
batch_size = 256
# max_env_step = int(1e6)
max_env_step = int(5e5)

reanalyze_ratio = 0.

eps_greedy_exploration_in_collect = False
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_efficientzero_config = dict(
    exp_name=
    f'data_ez_ctree/{env_id[:-14]}_efficientzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
    env=dict(
        env_id=env_id,
        obs_shape=(4, 96, 96),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
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
)
atari_efficientzero_create_config = EasyDict(atari_efficientzero_create_config)
create_config = atari_efficientzero_create_config

if __name__ == "__main__":
    # from lzero.entry import train_muzero
    # train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
    
    seeds = [0,1,2]  # You can add more seed values here
    # seeds = [0]  # You can add more seed values here
    for seed in seeds:
        # Update exp_name to include the current seed TODO
        main_config.exp_name=f'data_paper_ez_0518/{env_id[:-14]}_efficintzero_upc{update_per_collect}-mur{model_update_ratio}_bs{batch_size}_stack4_seed{seed}'
        from lzero.entry import train_muzero
        train_muzero([main_config, create_config], seed=seed, max_env_step=max_env_step)
