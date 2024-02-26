from easydict import EasyDict
import torch
torch.cuda.set_device(0)
# options={'PongNoFrameskip-v4', 'QbertNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4', 'BreakoutNoFrameskip-v4', ...}
env_name = 'PongNoFrameskip-v4'
# env_name = 'MsPacmanNoFrameskip-v4'
# env_name = 'BreakoutNoFrameskip-v4'
# env_name = 'QbertNoFrameskip-v4'
# env_name = 'SeaquestNoFrameskip-v4'
# env_name = 'BoxingNoFrameskip-v4'

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
elif env_name == 'BoxingNoFrameskip-v4':
    action_space_size = 18

# share action space
action_space_size = 18
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
# collector_env_num = 8
# n_episode = 8
# evaluator_env_num = 3
collector_env_num = 2
n_episode = 2
evaluator_env_num = 1
num_simulations = 50
# update_per_collect = 1000
update_per_collect = 2

# update_per_collect = None  # TODO
model_update_ratio = 0.25

# num_simulations = 50 
num_simulations = 5

# batch_size = 256
batch_size = 5

max_env_step = int(10e6)
reanalyze_ratio = 0.
# reanalyze_ratio = 0.5

eps_greedy_exploration_in_collect = False

exp_name_prefix = 'data_debug_mz_ctree_mt_pong-qbert-seaquest_0226/'

# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_muzero_config = dict(
    # TODO NOTE: 
    # muzero_collector: empty_cache
    # atari env full action space
    # game_buffer_muzero task_id
    # mcts_ctree, 
    exp_name=exp_name_prefix+f'{env_name[:-14]}_muzero-mt-v2_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}_new-rr{reanalyze_ratio}_46464_collect-orig_tep025_gsl400_noprio_target100_sgd02_seed0',
    # exp_name=exp_name_prefix+f'{env_name[:-14]}_mt-muzero_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}_new-rr{reanalyze_ratio}_46464_collect-orig_tep025_gsl400_noprio_target100_sgd02_seed0',
    env=dict(
        stop_value=int(1e6),
        env_name=env_name,
        observation_shape=(4, 64, 64),
        frame_stack_num=4,
        gray_scale=True,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        # TODO: run
        # collect_max_episode_steps=int(2e4),
        # eval_max_episode_steps=int(1e4),
        # TODO: debug
        collect_max_episode_steps=int(50),
        eval_max_episode_steps=int(50),
    ),
    policy=dict(
        task_id=0,
        model=dict(
            # observation_shape=(4, 96, 96),
            observation_shape=(4, 64, 64),
            image_channel=1,
            frame_stack_num=4,
            gray_scale=True,
            action_space_size=action_space_size,
            downsample=True,
            self_supervised_learning_loss=True,  # default is False
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
        ),
        # (int) The number of samples required for mini inference.
        mini_infer_size=1024,
        cuda=True,
        env_type='not_board_games',
        game_segment_length=400,
        # game_segment_length=50,
        random_collect_episode_num=0,
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            # need to dynamically adjust the number of decay steps 
            # according to the characteristics of the environment and the algorithm
            type='linear',
            start=1.,
            end=0.05,
            decay=int(1e5),
        ),
        use_augmentation=True,
        model_update_ratio = model_update_ratio,
        update_per_collect=update_per_collect,
        batch_size=batch_size,

        optim_type='SGD',
        lr_piecewise_constant_decay=True,
        learning_rate=0.2,

        # optim_type='SGD',
        # lr_piecewise_constant_decay=False,
        # learning_rate=1e-4,
        # weight_decay=1e-4,

        # optim_type='AdamW',
        # lr_piecewise_constant_decay=False,
        # learning_rate=1e-4,
        # weight_decay=1e-4,
        # weight_decay=0.1,

        # manual_temperature_decay=True,
        # threshold_training_steps_for_final_temperature=int(5e4),
        target_update_freq=100,
        use_priority=False,

        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        ssl_loss_weight=2,  # default is 0
        n_episode=n_episode,
        eval_freq=int(2e3),
        # eval_freq=int(2),

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
        type='muzero_multi_task_v2',
        import_names=['lzero.policy.muzero_multi_task_v2'],
    ),
)
atari_muzero_create_config = EasyDict(atari_muzero_create_config)
create_config = atari_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero_multi_task_v2
    import copy
    [main_config_2, main_config_3] = [copy.deepcopy(main_config) for _ in range(2)]
    [create_config_2, create_config_3] = [copy.deepcopy(create_config) for _ in range(2)]

    main_config_2.env.env_name = 'QbertNoFrameskip-v4'
    main_config_3.env.env_name = 'SeaquestNoFrameskip-v4'
    
    main_config_2.exp_name = exp_name_prefix + f'Qbert_mt-muzero_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}_new-rr{reanalyze_ratio}_46464_seed0'
    main_config_3.exp_name = exp_name_prefix + f'Seaquest_mt-muzero_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}_new-rr{reanalyze_ratio}_46464_seed0'

    # main_config_2.policy.model.action_space_size = 6
    # main_config_3.policy.model.action_space_size = 18
    main_config_2.policy.task_id = 1
    main_config_3.policy.task_id = 2

    train_muzero_multi_task_v2([[0, [main_config, create_config]], [1, [main_config_2, create_config_2]], [2, [main_config_3, create_config_3]]], seed=0, max_env_step=max_env_step)