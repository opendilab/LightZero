from easydict import EasyDict
from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map
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
batch_size = 256
replay_ratio = 0.25
max_env_step = int(5e5)

# ============= The key different params for ReZero =============
reuse_search = True
collect_with_pure_policy = True
buffer_reanalyze_freq = 1
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_muzero_config = dict(
    exp_name=f'data_rezero_mz/{env_id[:-14]}_rezero_muzero_ns{num_simulations}_upc{update_per_collect}_brf{buffer_reanalyze_freq}_seed0',
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        obs_shape=(4, 96, 96),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=(4, 96, 96),
            frame_stack_num=4,
            action_space_size=action_space_size,
            downsample=True,
            self_supervised_learning_loss=True,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
        ),
        cuda=True,
        env_type='not_board_games',
        use_augmentation=True,
        update_per_collect=update_per_collect,
        replay_ratio=replay_ratio,
        batch_size=batch_size,
        optim_type='SGD',
        lr_piecewise_constant_decay=True,
        learning_rate=0.2,
        num_simulations=num_simulations,
        ssl_loss_weight=2,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        # ============= The key different params for ReZero =============
        reuse_search=reuse_search,
        collect_with_pure_policy=collect_with_pure_policy,
        buffer_reanalyze_freq=buffer_reanalyze_freq,
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
    collector=dict(
        type='episode_muzero',
        import_names=['lzero.worker.muzero_collector'],
    )
)
atari_muzero_create_config = EasyDict(atari_muzero_create_config)
create_config = atari_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_rezero
    train_rezero([main_config, create_config], seed=0, max_env_step=max_env_step)
