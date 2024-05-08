from easydict import EasyDict
import torch
torch.cuda.set_device(0)
# options={'PongNoFrameskip-v4', 'QbertNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4', 'BreakoutNoFrameskip-v4', ...}
env_id = 'PongNoFrameskip-v4'
# env_id = 'MsPacmanNoFrameskip-v4'
# env_id = 'BreakoutNoFrameskip-v4'
# env_id = 'QbertNoFrameskip-v4'
# env_id = 'SeaquestNoFrameskip-v4'
# env_id = 'BoxingNoFrameskip-v4'

if env_id == 'PongNoFrameskip-v4':
    action_space_size = 6
elif env_id == 'QbertNoFrameskip-v4':
    action_space_size = 6
elif env_id == 'MsPacmanNoFrameskip-v4':
    action_space_size = 9
elif env_id == 'SpaceInvadersNoFrameskip-v4':
    action_space_size = 6
elif env_id == 'BreakoutNoFrameskip-v4':
    action_space_size = 4
elif env_id == 'SeaquestNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'BoxingNoFrameskip-v4':
    action_space_size = 18

# share action space
action_space_size = 18
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
update_per_collect = None  # TODO
model_update_ratio = 0.25

max_env_step = int(3e6)
reanalyze_ratio = 0.
batch_size = 64

num_unroll_steps = 10
eps_greedy_exploration_in_collect = True

# debug
batch_size = 2 
update_per_collect = 1
num_simulations = 1 
exp_name_prefix = 'data_debug_unizero_mt_stack1_pong-mspacman/'

# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_muzero_config = dict(
    # mcts_ctree, muzero_collector: empty_cache
    exp_name=exp_name_prefix+f'{env_id[:-14]}/{env_id[:-14]}_unizero_upc{update_per_collect}-mur{model_update_ratio}_H{num_unroll_steps}_bs{batch_size}_stack1_conlen{8}_lsd768-nlayer4-nh8_bacth-kvmaxsize_seed0',
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
        collect_max_episode_steps=int(50),
        eval_max_episode_steps=int(50),
        # TODO: run
        # collect_max_episode_steps=int(2e4),
        # eval_max_episode_steps=int(1e4),
        # collect_max_episode_steps=int(2e4),
        # eval_max_episode_steps=int(108000),
        # clip_rewards=False,
        clip_rewards=True,
    ),
    policy=dict(
        task_id=0,  # TODO
        model_path=None,
        analysis_sim_norm=False, # TODO
        cal_dormant_ratio=False, # TODO
        learn=dict(
            learner=dict(
                hook=dict(
                    load_ckpt_before_run='',
                    log_show_after_iter=100,
                    save_ckpt_after_iter=200000,  # default is 1000
                    save_ckpt_after_run=True,
                ),
            ),
        ),
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
        # grad_clip_value = 0.5, # TODO: 1
        grad_clip_value = 5, # TODO: 1
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        # eval_freq=int(9e9),
        # eval_freq=int(1e4),
        eval_freq=int(2e3),
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
        type='unizero_multi_task',
        import_names=['lzero.policy.unizero_multi_task'],
    ),
)
atari_muzero_create_config = EasyDict(atari_muzero_create_config)
create_config = atari_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_unizero_multi_task_v2
    import copy
    [main_config_2, main_config_3] = [copy.deepcopy(main_config) for _ in range(2)]
    [create_config_2, create_config_3] = [copy.deepcopy(create_config) for _ in range(2)]

    main_config_2.env.env_id = 'MsPacmanNoFrameskip-v4'
    main_config_3.env.env_id = 'SeaquestNoFrameskip-v4'
    
    main_config_2.exp_name = exp_name_prefix + f'MsPacman_unizero-mt_seed0'
    main_config_3.exp_name = exp_name_prefix + f'Seaquest_unizero-mt_seed0'

    main_config_2.policy.task_id = 1
    main_config_3.policy.task_id = 2

    train_unizero_multi_task_v2([[0, [main_config, create_config]], [1, [main_config_2, create_config_2]]], seed=0, max_env_step=max_env_step)
    # train_unizero_multi_task_v2([[0, [main_config, create_config]]], seed=0, max_env_step=max_env_step)

    # train_unizero_multi_task([[0, [main_config, create_config]], [1, [main_config_2, create_config_2]], [2, [main_config_3, create_config_3]]], seed=0, max_env_step=max_env_step)