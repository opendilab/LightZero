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
replay_ratio = 0.25
batch_size = 256
max_env_step = int(5e5)
reanalyze_ratio = 0.
history_length=4
# history_length=2

# collector_env_num = 4
# n_episode = 4
# evaluator_env_num = 3
# num_simulations = 3
# update_per_collect = 200
# replay_ratio = 0.
# batch_size = 2
# max_env_step = int(5e5)
# reanalyze_ratio = 0.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_muzero_config = dict(
    exp_name=f'data_lz/data_muzero_history/{env_id[:-14]}_muzero_HL{history_length}_final-LN_ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}_stack1_seed0',
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        observation_shape=(3, 64, 64),
        frame_stack_num=1,
        gray_scale=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        collect_max_episode_steps=int(2e4),
        eval_max_episode_steps=int(1e4),
        # debug
        # collect_max_episode_steps=int(1200),
        # eval_max_episode_steps=int(1200),
        # collect_max_episode_steps=int(50),
        # eval_max_episode_steps=int(50),
    ),
    policy=dict(
        model=dict(
            observation_shape=(3, 64, 64),
            image_channel=3,
            frame_stack_num=1,
            gray_scale=False,
            action_space_size=action_space_size,
            downsample=True,
            # self_supervised_learning_loss=True,
            self_supervised_learning_loss=False,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            # norm_type='LN',
            model_type='conv_history',
            history_length=history_length,
            num_unroll_steps=5,
        ),
        history_length=history_length,
        num_unroll_steps=5,
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        cuda=True,
        env_type='not_board_games',
        game_segment_length=400,
        # game_segment_length=20, # TODO： debug
        random_collect_episode_num=0,
        # use_augmentation=True,
        use_augmentation=False,
        update_per_collect=update_per_collect,
        replay_ratio=replay_ratio,
        batch_size=batch_size,
        optim_type='SGD',
        piecewise_decay_lr_scheduler=True,
        learning_rate=0.2,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        ssl_loss_weight=2,
        n_episode=n_episode,
        eval_freq=int(1e3),
        # eval_freq=int(1e2), # TODO
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
        type='muzero_history',
        import_names=['lzero.policy.muzero_history'],
    ),
)
atari_muzero_create_config = EasyDict(atari_muzero_create_config)
create_config = atari_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)