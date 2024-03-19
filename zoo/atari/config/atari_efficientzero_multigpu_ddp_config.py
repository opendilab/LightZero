from easydict import EasyDict

# options={'PongNoFrameskip-v4', 'QbertNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4', 'BreakoutNoFrameskip-v4', ...}
env_id = 'PongNoFrameskip-v4'

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

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
gpu_num = 2
collector_env_num = 8
n_episode = int(8*gpu_num)
evaluator_env_num = 3
num_simulations = 50
update_per_collect = 1000
batch_size = 256
max_env_step = int(1e6)
reanalyze_ratio = 0.
eps_greedy_exploration_in_collect = False

# the following is debug config
# collector_env_num = 2
# n_episode = int(2*2)
# evaluator_env_num = 1
# num_simulations = 2
# update_per_collect = 2
# batch_size = 4
# max_env_step = int(1e6)
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_efficientzero_config = dict(
    exp_name=f'data_ez_ctree/{env_id[:-14]}_efficientzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_ddp_{gpu_num}gpu_seed0',
    env=dict(
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
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
        ),
        multi_gpu=True,
        cuda=True,
        env_type='not_board_games',
        game_segment_length=400,
        random_collect_episode_num=0,
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            # need to dynamically adjust the number of decay steps according to the characteristics of the environment and the algorithm
            type='linear',
            start=1.,
            end=0.05,
            decay=int(1e5),
        ),
        use_augmentation=True,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='SGD',
        lr_piecewise_constant_decay=True,
        learning_rate=0.2,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
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
    collector=dict(
        type='episode_muzero',
        import_names=['lzero.worker.muzero_collector'],
    )
)
atari_efficientzero_create_config = EasyDict(atari_efficientzero_create_config)
create_config = atari_efficientzero_create_config

if __name__ == "__main__":
    """
    Overview:
        This script should be executed with <nproc_per_node> GPUs.
        Run the following command to launch the script:
        python -m torch.distributed.launch --nproc_per_node=2 ./LightZero/zoo/atari/config/atari_efficientzero_multigpu_ddp_config.py
    """
    from ding.utils import DDPContext
    from lzero.entry import train_muzero
    from lzero.config.utils import lz_to_ddp_config

    seed_list = [0, 1, 2]  # list of seeds you want to use for training
    for seed in seed_list:
        with DDPContext():
            # Each iteration uses a different seed for training
            # Change exp_name according to current seed
            main_config.exp_name = f'data_ez_ctree/{env_id[:-14]}_efficientzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_ddp_{gpu_num}gpu_seed{seed}'
            main_config = lz_to_ddp_config(main_config)
            train_muzero([main_config, create_config], seed=seed, max_env_step=max_env_step)