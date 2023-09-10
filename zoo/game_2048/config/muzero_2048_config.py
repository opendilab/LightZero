from easydict import EasyDict


# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
env_name = 'game_2048'
action_space_size = 4
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 100
update_per_collect = 200
batch_size = 512
max_env_step = int(5e6)
reanalyze_ratio = 0.
num_of_possible_chance_tile = 2
chance_space_size = 16 * num_of_possible_chance_tile
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_muzero_config = dict(
    exp_name=f'data_mz_ctree/game_2048_npct-{num_of_possible_chance_tile}_muzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_bs{batch_size}_sslw2_seed0',
    env=dict(
        stop_value=int(1e6),
        env_name=env_name,
        obs_shape=(16, 4, 4),
        obs_type='dict_encoded_board',
        raw_reward_type='raw',  # 'merged_tiles_plus_log_max_tile_num'
        reward_normalize=False,
        reward_norm_scale=100,
        max_tile=int(2 ** 16),  # 2**11=2048, 2**16=65536
        num_of_possible_chance_tile=num_of_possible_chance_tile,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=(16, 4, 4),
            action_space_size=action_space_size,
            image_channel=16,
            # NOTE: whether to use the self_supervised_learning_loss. default is False
            self_supervised_learning_loss=True,
        ),
        mcts_ctree=True,
        gumbel_algo=False,
        cuda=True,
        game_segment_length=200,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        td_steps=10,
        discount_factor=0.999,
        manual_temperature_decay=True,
        threshold_training_steps_for_final_temperature=int(1e5),
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=3e-3,
        # (float) Weight decay for training policy network.
        weight_decay=1e-4,
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
atari_muzero_config = EasyDict(atari_muzero_config)
main_config = atari_muzero_config

atari_muzero_create_config = dict(
    env=dict(
        type='game_2048',
        import_names=['zoo.game_2048.envs.game_2048_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)
atari_muzero_create_config = EasyDict(atari_muzero_create_config)
create_config = atari_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
