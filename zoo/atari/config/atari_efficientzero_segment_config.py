from easydict import EasyDict
from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map

# env_id = 'PongNoFrameskip-v4'  # You can specify any Atari game here
env_id = 'MsPacmanNoFrameskip-v4'  # You can specify any Atari game here

action_space_size = atari_env_action_space_map[env_id]

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
update_per_collect = None
replay_ratio = 0.25
# replay_ratio = 0.1 # 50M envsteps, 5M train iter
# replay_ratio = 0.02 # 50M envsteps, 1M train iter


num_segments = 8
game_segment_length = 20
# game_segment_length = 400

collect_num_simulations = 25
eval_num_simulations = 50

collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
batch_size = 256
max_env_step = int(5e5)
reanalyze_ratio = 0.
num_unroll_steps = 5

# Defines the frequency of reanalysis. E.g., 1 means reanalyze once per epoch, 2 means reanalyze once every two epochs.
# buffer_reanalyze_freq = 1
buffer_reanalyze_freq = 1/50
# buffer_reanalyze_freq = 1/10000
# Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
reanalyze_batch_size = 160
# The partition of reanalyze. E.g., 1 means reanalyze_batch samples from the whole buffer, 0.5 means samples from the first half of the buffer.
reanalyze_partition=0.75
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_efficientzero_config = dict(
    exp_name=f'data_efficientzero/{env_id[:-14]}_efficientzero_stack4_H{num_unroll_steps}_seed0',
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        frame_stack_num=4,
        # observation_shape=[4, 64, 64],
        # gray_scale=True,
        observation_shape=(12, 64, 64),
        gray_scale=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=(12, 64, 64),
            image_channel=3,
            gray_scale=False,

            # observation_shape=[4, 64, 64],
            # image_channel=1,
            # gray_scale=True,

            frame_stack_num=4,
            action_space_size=action_space_size,
            downsample=True,
            self_supervised_learning_loss=True,  # default is False
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            reward_support_range=(-50., 51., 1.),
            value_support_range=(-50., 51., 1.),
        ),
        cuda=True,
        env_type='not_board_games',
        num_segments=num_segments,
        train_start_after_envsteps=2000,
        game_segment_length=game_segment_length,
        use_augmentation=True,
        use_priority=False,
        # use_priority=True, # TODO(pu): test
        replay_ratio=replay_ratio,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        dormant_threshold=0.025,
        optim_type='SGD',
        piecewise_decay_lr_scheduler=True,
        learning_rate=0.2,
        target_update_freq=100,
        num_simulations=50, # for reanalyze
        collect_num_simulations=collect_num_simulations,
        eval_num_simulations=eval_num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        ssl_loss_weight=2,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,

        # ============= The key different params for reanalyze =============
        # Defines the frequency of reanalysis. E.g., 1 means reanalyze once per epoch, 2 means reanalyze once every two epochs.
        buffer_reanalyze_freq=buffer_reanalyze_freq,
        # Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
        reanalyze_batch_size=reanalyze_batch_size,
        # The partition of reanalyze. E.g., 1 means reanalyze_batch samples from the whole buffer, 0.5 means samples from the first half of the buffer.
        reanalyze_partition=reanalyze_partition,
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

if __name__ == "__main__":
    # Define a list of seeds for multiple runs
    # seeds = [0, 1, 2]  # You can add more seed values here
    seeds = [0]  # You can add more seed values here

    for seed in seeds:
        # Update exp_name to include the current seed
        main_config.exp_name = f'data_efficientzero_20250731/{env_id[:-14]}_efficientzero_stack4_rgb_H{num_unroll_steps}_seed{seed}'
        from lzero.entry import train_muzero_segment
        train_muzero_segment([main_config, create_config], seed=seed, max_env_step=max_env_step)
