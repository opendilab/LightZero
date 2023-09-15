from easydict import EasyDict

env_name = 'PongNoFrameskip-v4'
action_space_size = 6

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 5
update_per_collect = 10
batch_size = 4
max_env_step = int(1e6)
reanalyze_ratio = 0.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_efficientzero_config = dict(
    exp_name='data_ez_ctree/efficientzero_seed0',
    env=dict(
        env_name=env_name,
        obs_shape=(4, 96, 96),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        env_type='Atari',
        max_episode_steps=int(1.08e5),
        gray_scale=True,
        frame_skip=4,
        episode_life=True,
        clip_rewards=True,
        channel_last=True,
        render_mode_human=False,
        scale=True,
        warp_frame=True,
        save_video=False,
        transform2string=False,
        game_wrapper=True,
        stop_value=int(1e6),
    ),
    policy=dict(
        sampled_algo=False,
        gumbel_algo=False,
        mcts_ctree=True,
        model=dict(
            observation_shape=(4, 96, 96),
            frame_stack_num=4,
            action_space_size=action_space_size,
            representation_network_type='conv_res_blocks',
            downsample=True,
            model_type='conv',  # options={'mlp', 'conv'}
            # (bool) If True, the action space of the environment is continuous, otherwise discrete.
            continuous_action_space=False,
            self_supervised_learning_loss=True,
            categorical_distribution=True,
            image_channel=1,
            support_scale=300,
            lstm_hidden_size=512,
        ),
        cuda=True,
        env_type='not_board_games',
        transform2string=False,
        gray_scale=False,
        game_segment_length=400,
        use_augmentation=True,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        manual_temperature_decay=False,
        fixed_temperature_value=0.25,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        lr_piecewise_constant_decay=True,
        optim_type='Adam',
        learning_rate=0.2,  # init lr for manually decay schedule
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        discount_factor=0.997,
        lstm_horizon_len=5,
        use_ture_chance_label_in_chance_encoder=False,
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
