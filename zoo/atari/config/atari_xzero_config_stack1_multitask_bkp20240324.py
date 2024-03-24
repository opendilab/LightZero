from easydict import EasyDict
import torch
torch.cuda.set_device(5)

# ==== NOTE: 需要设置cfg_atari中的action_shape =====
# options={'PongNoFrameskip-v4', 'QbertNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4', 'BreakoutNoFrameskip-v4', ...}
env_name = 'PongNoFrameskip-v4'
# env_name = 'MsPacmanNoFrameskip-v4'
# env_name = 'QbertNoFrameskip-v4'
# env_name = 'SeaquestNoFrameskip-v4'
# env_name = 'BoxingNoFrameskip-v4'
# env_name = 'FrostbiteNoFrameskip-v4'
# env_name = 'BreakoutNoFrameskip-v4'  # TODO: eval_sample

if env_name == 'PongNoFrameskip-v4':
    action_space_size = 6
    update_per_collect = 1000  # for pong boxing
elif env_name == 'QbertNoFrameskip-v4':
    action_space_size = 6
    update_per_collect = None # for others
elif env_name == 'MsPacmanNoFrameskip-v4':
    action_space_size = 9
    update_per_collect = None # for others
elif env_name == 'SpaceInvadersNoFrameskip-v4':
    action_space_size = 6
    update_per_collect = None # for others
elif env_name == 'BreakoutNoFrameskip-v4':
    action_space_size = 4
    update_per_collect = None # for others
elif env_name == 'SeaquestNoFrameskip-v4':
    action_space_size = 18
    update_per_collect = None # for others
elif env_name == 'BoxingNoFrameskip-v4':
    action_space_size = 18
    update_per_collect = 1000  # for pong boxing
elif env_name == 'FrostbiteNoFrameskip-v4':
    action_space_size = 18
    update_per_collect = None # for others

# share action space
action_space_size = 18
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3

num_simulations = 50
model_update_ratio = 0.25
batch_size = 256
max_env_step = int(3e6)
reanalyze_ratio = 0.
batch_size = 64
num_unroll_steps = 5
threshold_training_steps_for_final_temperature = int(5e4)  # train_iter 50k 1->0.5->0.25
eps_greedy_exploration_in_collect = True # for breakout, qbert, boxing
# eps_greedy_exploration_in_collect = False 

# exp_name_prefix = 'data_mt/mz_gpt_ctree_mt_stack1_pong-qbert-seaquest/'
exp_name_prefix = 'data_mt/xzero_mt_stack1_pong-qbert/'


# num_simulations = 8 # debug
# update_per_collect = 1 # debug

# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_muzero_config = dict(
    # mcts_ctree, muzero_collector: empty_cache
    exp_name=exp_name_prefix+f'{env_name[:-14]}_mt-xzero_envnum{collector_env_num}_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}_new-rr{reanalyze_ratio}_H{num_unroll_steps}_bs{batch_size}_stack1_contembdings_lsd1024_lr1e-4-reconlwperlw-005_seed0',
    env=dict(
        stop_value=int(1e6),
        env_name=env_name,
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
        # eval_max_episode_steps=int(108000),
        clip_rewards=True,
    ),
    policy=dict(
        learn=dict(
            learner=dict(
                hook=dict(
                    load_ckpt_before_run='',
                    log_show_after_iter=100,
                    save_ckpt_after_iter=100000,  # default is 1000
                    save_ckpt_after_run=True,
                ),
            ),
        ),
        task_id=0,
        model_path=None,
        tokenizer_start_after_envsteps=int(0),
        transformer_start_after_envsteps=int(0),
        update_per_collect_transformer=update_per_collect,
        update_per_collect_tokenizer=update_per_collect,
        num_unroll_steps=num_unroll_steps,
        model=dict(
            observation_shape=(3, 64, 64),
            image_channel=3,
            frame_stack_num=1,
            gray_scale=False,

            action_space_size=action_space_size,
            downsample=True,
            self_supervised_learning_loss=True,  # default is False
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            
            reward_support_size=601,
            value_support_size=601,
            support_scale=300,
            embedding_dim=1024,
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
        # threshold_training_steps_for_final_temperature=int(5e4), # 100k 1->0.5->0.25
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.0001,
        target_update_freq=100,
        grad_clip_value = 0.5, # TODO
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        ssl_loss_weight=2,  # default is 0
        n_episode=n_episode,
        eval_freq=int(1e4),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
atari_muzero_config = EasyDict(atari_muzero_config)
main_config = atari_muzero_config

atari_muzero_create_config = dict(
    env=dict(
        type='atari_lightzero',
        import_names=['zoo.atari.envs.atari_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero_gpt_multi_task_v2',
        import_names=['lzero.policy.muzero_gpt_multi_task_v2'],
    ),
)
atari_muzero_create_config = EasyDict(atari_muzero_create_config)
create_config = atari_muzero_create_config

if __name__ == "__main__":
    # from lzero.entry import train_muzero_gpt_multi_task_v2
    # import copy
    # [main_config_2, main_config_3] = [copy.deepcopy(main_config) for _ in range(2)]
    # [create_config_2, create_config_3] = [copy.deepcopy(create_config) for _ in range(2)]

    # main_config_2.env.env_name = 'QbertNoFrameskip-v4'
    # main_config_3.env.env_name = 'SeaquestNoFrameskip-v4'
    
    # main_config_2.exp_name = exp_name_prefix + f'Qbert_mt-muzero-gpt_seed0'
    # main_config_3.exp_name = exp_name_prefix + f'Seaquest_mt-muzero-gpt_seed0'

    # # main_config_2.policy.model.action_space_size = 6
    # # main_config_3.policy.model.action_space_size = 18
    # main_config_2.policy.task_id = 1
    # main_config_3.policy.task_id = 2
    # train_muzero_gpt_multi_task_v2([[0, [main_config, create_config]], [1, [main_config_2, create_config_2]], [2, [main_config_3, create_config_3]]], seed=0, max_env_step=max_env_step)


    from lzero.entry import train_muzero_gpt_multi_task_v2
    import copy
    [main_config_2] = [copy.deepcopy(main_config) for _ in range(1)]
    [create_config_2] = [copy.deepcopy(create_config) for _ in range(1)]
    main_config_2.env.env_name = 'QbertNoFrameskip-v4'
    main_config_2.exp_name = exp_name_prefix + f'Qbert_mt-muzero_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}_new-rr{reanalyze_ratio}_46464_seed0'
    # main_config_2.policy.model.action_space_size = 6
    # main_config_3.policy.model.action_space_size = 18
    main_config_2.policy.task_id = 1
    train_muzero_gpt_multi_task_v2([[0, [main_config, create_config]], [1, [main_config_2, create_config_2]]], seed=0, max_env_step=max_env_step)