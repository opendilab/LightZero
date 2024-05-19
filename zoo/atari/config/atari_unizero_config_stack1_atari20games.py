from easydict import EasyDict


# ==== NOTE: 需要设置cfg_atari中的action_shape =====
import torch
torch.cuda.set_device(0)

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
# env_id = 'BoxingNoFrameskip-v4' # 18

env_id = 'BreakoutNoFrameskip-v4'  # TODO: eval_sample, episode_steps


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
collector_env_num = 1 # TODO
n_episode = 1

evaluator_env_num = 3
num_simulations = 50
# max_env_step = int(1e6)
max_env_step = int(5e5)
reanalyze_ratio = 0. 
# reanalyze_ratio = 0.05 # TODO

batch_size = 64

# num_unroll_steps = 5
# num_unroll_steps = 8
num_unroll_steps = 10 # 默认的
# num_unroll_steps = 20 # TODO
# num_unroll_steps = 40 # TODO


threshold_training_steps_for_final_temperature = int(5e4)  # train_iter 50k 1->0.5->0.25
eps_greedy_exploration_in_collect = True # for breakout, qbert, boxing
# eps_greedy_exploration_in_collect = False 
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_unizero_config = dict(
    # TODO: 
    # mcts_ctree
    # muzero_collector/evaluator: empty_cache
    # exp_name=f'data_paper_learn-dynamics_0423/{env_id[:-14]}_unizero_upc{update_per_collect}-mur{model_update_ratio}_H{num_unroll_steps}_bs{batch_size}_stack1_conlen{8}_lsd768-nlayer1-nh8_bacth-kvmaxsize_analysis_dratio0025_seed0',
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        observation_shape=(3, 64, 64),
        gray_scale=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        # TODO: debug
        # collect_max_episode_steps=int(50),
        # eval_max_episode_steps=int(50),
        # TODO: for breakout
        # collect_max_episode_steps=int(5e3), # for breakout
        # eval_max_episode_steps=int(5e3), # for breakout
        # TODO: for others
        collect_max_episode_steps=int(2e4), 
        eval_max_episode_steps=int(1e4),
        clip_rewards=True,
    ),
    policy=dict(
        analysis_sim_norm=False, # TODO
        cal_dormant_ratio=False, # TODO
        learn=dict(
            learner=dict(
                hook=dict(
                    load_ckpt_before_run='',
                    log_show_after_iter=100,
                    save_ckpt_after_iter=500000,  # default is 1000
                    save_ckpt_after_run=True,
                ),
            ),
        ),
        model_path=None,
        tokenizer_start_after_envsteps=int(0),
        transformer_start_after_envsteps=int(0),
        update_per_collect_transformer=update_per_collect,
        update_per_collect_tokenizer=update_per_collect,
        num_unroll_steps=num_unroll_steps,
        model=dict(
            analysis_sim_norm = False,
            observation_shape=(3, 64, 64),
            image_channel=3,
            frame_stack_num=1,
            gray_scale=False,
            action_space_size=action_space_size,
            downsample=True,
            self_supervised_learning_loss=True,  # default is False
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            # reward_support_size=601,
            # value_support_size=601,
            # support_scale=300,
            reward_support_size=101,
            value_support_size=101,
            support_scale=50,
        ),
        use_priority=False, # TODO
        use_augmentation=False,  # TODO
        # use_augmentation=True,  # NOTE: only for image-based atari
        cuda=True,
        env_type='not_board_games',
        game_segment_length=400,
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
        update_per_collect=update_per_collect,
        model_update_ratio = model_update_ratio,
        batch_size=batch_size,
        # manual_temperature_decay=True,
        # threshold_training_steps_for_final_temperature=threshold_training_steps_for_final_temperature,
        optim_type='AdamW',
        lr_piecewise_constant_decay=False,
        learning_rate=0.0001,
        target_update_freq=100,
        grad_clip_value = 5, # TODO: 1
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
atari_unizero_config = EasyDict(atari_unizero_config)
main_config = atari_unizero_config

atari_unizero_create_config = dict(
    env=dict(
        type='atari_lightzero',
        import_names=['zoo.atari.envs.atari_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='unizero',
        import_names=['lzero.policy.unizero'],
    ),
)
atari_unizero_create_config = EasyDict(atari_unizero_create_config)
create_config = atari_unizero_create_config

if __name__ == "__main__":
    # max_env_step = 10000
    # from lzero.entry import train_unizero
    # train_unizero([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)


    # Define a list of seeds for multiple runs
    seeds = [2,1,0]  # You can add more seed values here
    # seeds = [0,1]  # You can add more seed values here
    # seeds = [1,2]  # You can add more seed values here
    for seed in seeds:
        # Update exp_name to include the current seed TODO
        main_config.exp_name=f'data_paper_unizero_0519/{env_id[:-14]}_unizero_upc{update_per_collect}-mur{model_update_ratio}_H{num_unroll_steps}_bs{batch_size}_stack1_conlen{8}_lsd768-nlayer4-nh8_bacth-kvmaxsize_collectenv{collector_env_num}_reclw0_seed{seed}'

        # main_config.exp_name=f'data_paper_unizero_ablation_0502/target_world_model_{env_id[:-14]}/{env_id[:-14]}_unizero_upc{update_per_collect}-mur{model_update_ratio}_H{num_unroll_steps}_bs{batch_size}_stack1_conlen{8}_lsd768-nlayer4-nh8_bacth-kvmaxsize_collectenv{collector_env_num}_seed{seed}'

        # main_config.exp_name=f'data_paper_unizero_ablation_0502/regu_loss_{env_id[:-14]}/{env_id[:-14]}_unizero_upc{update_per_collect}-mur{model_update_ratio}_H{num_unroll_steps}_bs{batch_size}_stack1_conlen{8}_lsd768-nlayer4-nh8_bacth-kvmaxsize_collectenv{collector_env_num}_reclw0_seed{seed}'
        # main_config.exp_name=f'data_paper_unizero_ablation_0502/latent_norm_{env_id[:-14]}/{env_id[:-14]}_unizero_upc{update_per_collect}-mur{model_update_ratio}_H{num_unroll_steps}_bs{batch_size}_stack1_conlen{8}_lsd768-nlayer4-nh8_bacth-kvmaxsize_collectenv{collector_env_num}_latentsoftmax-true_seed{seed}'
        # main_config.exp_name=f'data_paper_unizero_atari_0505/{env_id[:-14]}/{env_id[:-14]}_unizero_upc{update_per_collect}-mur{model_update_ratio}_H{num_unroll_steps}_bs{batch_size}_stack1_conlen{8}_lsd768-nlayer4-nh8_bacth-kvmaxsize_collectenv{collector_env_num}_seed{seed}'

        from lzero.entry import train_unizero
        train_unizero([main_config, create_config], seed=seed, max_env_step=max_env_step)