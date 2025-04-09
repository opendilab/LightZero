from easydict import EasyDict
from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map
norm_type = 'BN'
env_id = 'PongNoFrameskip-v4'  # You can specify any Atari game here
action_space_size = atari_env_action_space_map[env_id]

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
update_per_collect = None
replay_ratio = 0.25
batch_size = 256
max_env_step = int(2e5)
reanalyze_ratio = 0.

# =========== for debug ===========
# collector_env_num = 2
# n_episode = 2
# evaluator_env_num = 2
# num_simulations = 2
# update_per_collect = 2
# batch_size = 2
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_muzero_config = dict(
    exp_name=f'data_muzero_shm/{env_id[:-14]}_atari_stack4_muzero_ns{num_simulations}_upc{update_per_collect}-rr{replay_ratio}_seed0',
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        observation_shape=(4, 64, 64),  # (4, 96, 96)
        frame_stack_num=4,
        gray_scale=True,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        # manager=dict(shared_memory=False, ),
        manager=dict(shared_memory=True, ),
        # TODO: debug
        # collect_max_episode_steps=int(50),
        # eval_max_episode_steps=int(50),
    ),
    policy=dict(
        analysis_sim_norm=False,
        cal_dormant_ratio=False,
        model=dict(
            observation_shape=(4, 64, 64),  # (4, 96, 96)
            image_channel=1,
            frame_stack_num=4,
            gray_scale=True,
            action_space_size=action_space_size,
            downsample=True,
            self_supervised_learning_loss=True,  # default is False
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            use_sim_norm=True,
            use_sim_norm_kl_loss=False,
            model_type='conv'
        ),
        cuda=True,
        env_type='not_board_games',
        game_segment_length=400,
        random_collect_episode_num=0,
        use_augmentation=True,
        use_priority=False,
        replay_ratio=replay_ratio,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='SGD',
        piecewise_decay_lr_scheduler=True,
        learning_rate=0.2,
        target_update_freq=100,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        ssl_loss_weight=2,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),
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
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)
atari_muzero_create_config = EasyDict(atari_muzero_create_config)
create_config = atari_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
