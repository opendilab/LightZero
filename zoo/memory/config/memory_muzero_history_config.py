from easydict import EasyDict

env_id = 'visual_match'  # The name of the environment, options: 'visual_match', 'key_to_door'
memory_length = 60
max_env_step = int(5e5)

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
seed = 0
collector_env_num = 8
n_episode = 8
evaluator_env_num = 8
num_simulations = 50
update_per_collect = None # for others
replay_ratio = 0.25 
# batch_size = 256
reanalyze_ratio = 0
td_steps = 5
game_segment_length = 30+memory_length

# num_unroll_steps = 16+memory_length
# TODO
num_unroll_steps = 5

policy_entropy_weight = 1e-4
threshold_training_steps_for_final_temperature = int(1e5)
eps_greedy_exploration_in_collect = True
# history_length = 20
# history_length = 40
# history_length = 60
history_length = 70
batch_size = 128

# debug
# num_simulations = 3
# batch_size = 3
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

memory_muzero_config = dict(
    exp_name=f'data_lz/data_muzero_history_20250324/{env_id}_memlen-{memory_length}_muzero_HL{history_length}_transformer_ns{num_simulations}_upc{update_per_collect}_seed{seed}',
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        observation_shape=(3, 5, 5),
        frame_stack_num=1,
        image_channel=3,
        gray_scale=False,
        # rgb_img_observation=False,  # Whether to return RGB image observation
        rgb_img_observation=True,  # Whether to return RGB image observation
        scale_rgb_img_observation=True,  # Whether to scale the RGB image observation to [0, 1]
        # flatten_observation=True,  # Whether to flatten the observation
        flatten_observation=False,  # Whether to flatten the observation
        max_frames={
            # "explore": 15,  # for key_to_door
            "explore": 1,  # for visual_match
            "distractor": memory_length,
            "reward": 15
        },  # Maximum frames per phase
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        sample_type='episode',  # NOTE: very important for memory env
        num_unroll_steps=num_unroll_steps,
        history_length=history_length,
        model=dict(
            observation_shape=(3, 5, 5),
            image_channel=3,
            frame_stack_num=1,
            gray_scale=False,
            action_space_size=4,
            analysis_sim_norm=False,
            # model_type='mlp',
            latent_state_dim=128,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            # norm_type='LN',
            # self_supervised_learning_loss=True,  # NOTE: default is False.
            self_supervised_learning_loss=False,
            downsample=False,
            model_type='conv_history',
            history_length=history_length,
            fusion_mode= 'transformer',  # 可选: 'mean', 'transformer', 其它未来方式
            num_unroll_steps=num_unroll_steps,
        ),
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            decay=int(5e4),  # NOTE: 50k env steps  for key_to_door
        ),
        policy_entropy_weight=policy_entropy_weight,
        td_steps=td_steps,
        cuda=True,
        env_type='not_board_games',
        game_segment_length=game_segment_length,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='AdamW',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.0001,
        ssl_loss_weight=2,  # NOTE: default is 0.
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(5e3),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

memory_muzero_config = EasyDict(memory_muzero_config)
main_config = memory_muzero_config

memory_muzero_create_config = dict(
    env=dict(
        type='memory_lightzero',
        import_names=['zoo.memory.envs.memory_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero_history',
        import_names=['lzero.policy.muzero_history'],
    ),
    collector=dict(
        type='episode_muzero',
        import_names=['lzero.worker.muzero_collector'],
    )
)
memory_muzero_create_config = EasyDict(memory_muzero_create_config)
create_config = memory_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=seed, max_env_step=max_env_step)
