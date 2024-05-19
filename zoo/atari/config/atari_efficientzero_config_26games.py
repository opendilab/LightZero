from easydict import EasyDict
import torch
torch.cuda.set_device(2)

# env_id = 'AlienNoFrameskip-v4' # 18
# env_id = 'AmidarNoFrameskip-v4' # 10
# env_id = 'AssaultNoFrameskip-v4' # 7
# env_id = 'AsterixNoFrameskip-v4' # 9

# env_id = 'BankHeistNoFrameskip-v4' # 18
# env_id = 'BattleZoneNoFrameskip-v4' # 18
# env_id = 'ChopperCommandNoFrameskip-v4' # 18
# env_id = 'CrazyClimberNoFrameskip-v4' # 9

# env_id = 'DemonAttackNoFrameskip-v4' # 6
# env_id = 'FreewayNoFrameskip-v4' # 3
# env_id = 'FrostbiteNoFrameskip-v4' # 18
# env_id = 'GopherNoFrameskip-v4' # 8

# env_id = 'HeroNoFrameskip-v4' # 18
# env_id = 'JamesbondNoFrameskip-v4' # 18
# env_id = 'KangarooNoFrameskip-v4' # 18
# env_id = 'KrullNoFrameskip-v4' # 18

# env_id = 'KungFuMasterNoFrameskip-v4' # 14
# env_id = 'PrivateEyeNoFrameskip-v4' # 18
# env_id = 'RoadRunnerNoFrameskip-v4' # 18
# env_id = 'UpNDownNoFrameskip-v4' # 6

# env_id = 'PongNoFrameskip-v4' # 6
# env_id = 'MsPacmanNoFrameskip-v4' # 9
# env_id = 'QbertNoFrameskip-v4'  # 6
# env_id = 'SeaquestNoFrameskip-v4' # 18
env_id = 'BoxingNoFrameskip-v4' # 18

# env_id = 'BreakoutNoFrameskip-v4'  # TODO: eval_sample, episode_steps


update_per_collect = None # for others
# update_per_collect = 1000
# model_update_ratio = 1.
# model_update_ratio = 0.5
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
    model_update_ratio = 0.25
    # model_update_ratio = 0.5
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
# collector_env_num = 8
# n_episode = 8
collector_env_num = 1 # TODO ========
n_episode = 1
evaluator_env_num = 3
num_simulations = 50
batch_size = 256
# max_env_step = int(1e6)
max_env_step = int(5e5)
reanalyze_ratio = 0.
eps_greedy_exploration_in_collect = True
num_unroll_steps = 5
context_length_init = 1

# for debug ===========
# collector_env_num = 1
# n_episode = 1
# evaluator_env_num = 1
# num_simulations = 2
# update_per_collect = 2
# model_update_ratio = 0.25
# batch_size = 2
# max_env_step = int(5e5)
# reanalyze_ratio = 0.
# eps_greedy_exploration_in_collect = True
# num_unroll_steps = 5
# context_length_init = 1
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_efficientzero_config = dict(
    exp_name=f'data_paper_efficientzero_atari-20-games_0510/{env_id[:-14]}_efficientzero_stack4_H{num_unroll_steps}_initconlen{context_length_init}_simnorm-cossim_sgd02_seed0',
    # exp_name=f'data_paper_learn-dynamics_atari-20-games_0424/{env_id[:-14]}_efficientzero_stack4_H{num_unroll_steps}_initconlen{context_length_init}_simnorm-cossim_sgd02_analysis_dratio0025_seed0',
    # exp_name=f'data_paper_efficientzero_variants_0422/{env_id[:-14]}_efficientzero_stack4_H{num_unroll_steps}_conlen1_simnorm-cossim_adamw1e-4_seed0',
    # exp_name=f'data_paper_efficientzero_variants_0422/{env_id[:-14]}_efficientzero_stack4_H{num_unroll_steps}_conlen1_simnorm-cossim_adamw1e-4_seed0',
    # exp_name=f'data_paper_efficientzero_variants_0422/{env_id[:-14]}_efficientzero_stack4_H{num_unroll_steps}_conlen1_sslw2-cossim_adamw1e-4_seed0',
    # exp_name=f'data_paper_efficientzero_variants_0422/{env_id[:-14]}_efficientzero_stack4_H{num_unroll_steps}_conlen1_sslw2-cossim_sgd02_seed0',
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        # observation_shape=(4, 64, 64),
        observation_shape=[4, 64, 64],
        frame_stack_num=4,
        gray_scale=True,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        # TODO: debug
        # collect_max_episode_steps=int(20),
        # eval_max_episode_steps=int(20),
    ),
    policy=dict(
        learn=dict(
            learner=dict(
                hook=dict(
                    load_ckpt_before_run='',
                    log_show_after_iter=100,
                    save_ckpt_after_iter=1000000,  # default is 1000
                    save_ckpt_after_run=True,
                ),
            ),
        ),
        cal_dormant_ratio=False, # TODO
        analysis_sim_norm=False, # TODO
        model=dict(
            analysis_sim_norm=False, # TODO
            # observation_shape=(4, 64, 64),
            observation_shape=[4, 64, 64],
            image_channel=1,
            frame_stack_num=4,
            gray_scale=True,
            action_space_size=action_space_size,
            downsample=True,
            self_supervised_learning_loss=True,  # default is False
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            reward_support_size=101,
            value_support_size=101,
            support_scale=50,
            context_length_init=context_length_init,  # NOTE:TODO num_unroll_steps
            use_sim_norm=True,
            # use_sim_norm_kl_loss=True,  # TODO
            use_sim_norm_kl_loss=False,  # TODO
        ),
        cuda=True,
        env_type='not_board_games',
        game_segment_length=400, # for collector orig
        # game_segment_length=50, # for collector game_segment
        random_collect_episode_num=0,
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            # need to dynamically adjust the number of decay steps 
            # according to the characteristics of the environment and the algorithm
            type='linear',
            start=1.,
            end=0.01,
            decay=int(2e4),  # TODO: 20k
        ),
        use_augmentation=True,  # TODO
        # use_augmentation=False,
        use_priority=False,
        model_update_ratio = model_update_ratio,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        dormant_threshold=0.025,

        optim_type='SGD', # for collector orig
        lr_piecewise_constant_decay=True,
        learning_rate=0.2,

        # optim_type='AdamW', # for collector game_segment
        # lr_piecewise_constant_decay=False,
        # learning_rate=1e-4,

        target_update_freq=100,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        ssl_loss_weight=2,  # default is 0
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
    # from lzero.entry import train_efficientzero
    # train_efficientzero([main_config, create_config], seed=0, max_env_step=max_env_step)

    # Define a list of seeds for multiple runs
    seeds = [0,1,2]  # You can add more seed values here
    for seed in seeds:
        # Update exp_name to include the current seed TODO
        main_config.exp_name=f'data_paper_efficientzero_0519/{env_id[:-14]}_efficientzero_stack4_collectenv{collector_env_num}_H{num_unroll_steps}_initconlen{context_length_init}_sgd02_seed{seed}'
        from lzero.entry import train_muzero
        train_muzero([main_config, create_config], seed=seed, max_env_step=max_env_step)