from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
continuous_action_space = True
K = 20  # num_of_sampled_actions
num_simulations = 50
update_per_collect = None
replay_ratio = 0.25
batch_size = 256
max_env_step = int(5e6)
reanalyze_ratio = 0.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

dmc2gym_pixels_sampled_efficientzero_config = dict(
    exp_name=f'data_sez/dmc2gym_pixels_sampled_efficientzero_k{K}_ns{num_simulations}_upc{update_per_collect}-rr{replay_ratio}_rer{reanalyze_ratio}_seed0',
    env=dict(
        env_id='dmc2gym-v0',
        domain_name="cartpole",
        task_name="swingup",
        from_pixels=True,  # pixel/image obs
        frame_skip=4,
        warp_frame=True,
        scale=True,
        frame_stack_num=3,
        channels_first=True,
        stop_value=1e6,
        continuous=True,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            # observation_shape=(3, 84, 84),  # for frame_stack_num=1
            observation_shape=(9, 84, 84),
            action_space_size=1,
            image_channel=3,
            frame_stack_num=3,
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            sigma_type='conditioned',
            model_type='conv',
            lstm_hidden_size=512,
            latent_state_dim=256,
            downsample=True,
            self_supervised_learning_loss=True,
            res_connection_in_dynamics=True,
            norm_type='LN',
            use_sim_norm=False,
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        cuda=True,
        env_type='not_board_games',
        game_segment_length=200,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='AdamW',
        learning_rate=0.0001,
        grad_clip_value=5,
        policy_entropy_loss_weight=5e-3,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_ratio=replay_ratio,
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
dmc2gym_pixels_sampled_efficientzero_config = EasyDict(dmc2gym_pixels_sampled_efficientzero_config)
main_config = dmc2gym_pixels_sampled_efficientzero_config

dmc2gym_pixels_sampled_efficientzero_create_config = dict(
    env=dict(
        type='dmc2gym_lightzero',
        import_names=['zoo.dmc2gym.envs.dmc2gym_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sampled_efficientzero',
        import_names=['lzero.policy.sampled_efficientzero'],
    ),
)
dmc2gym_pixels_sampled_efficientzero_create_config = EasyDict(
    dmc2gym_pixels_sampled_efficientzero_create_config
)
create_config = dmc2gym_pixels_sampled_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)
