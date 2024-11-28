# zoo/atari/config/atari_muzero_multitask_segment_8games_config.py

from easydict import EasyDict
from copy import deepcopy
from atari_env_action_space_map import atari_env_action_space_map

def create_config(
    env_id,
    action_space_size,
    collector_env_num,
    evaluator_env_num,
    n_episode,
    num_simulations,
    reanalyze_ratio,
    batch_size,
    num_unroll_steps,
    infer_context_length,
    norm_type,
    buffer_reanalyze_freq,
    reanalyze_batch_size,
    reanalyze_partition,
    num_segments
):

    return EasyDict(dict(
        env=dict(
            stop_value=int(5e5),  # Adjusted max_env_step based on user TODO
            env_id=env_id,
            observation_shape=(4, 96, 96),
            frame_stack_num=4,
            gray_scale=True,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
            full_action_space=True,
            # ===== only for debug =====
            # collect_max_episode_steps=int(50),
            # eval_max_episode_steps=int(50),
        ),
        policy=dict(
            multi_gpu=True, # ======== Very important for ddp =============
            learn=dict(
                learner=dict(
                    hook=dict(save_ckpt_after_iter=200000,),  # Adjusted checkpoint frequency
                ),
            ),
            grad_correct_params=dict(
                # Placeholder for gradient correction parameters if needed
            ),
            task_num=len(env_id_list),
            model=dict(
                num_res_blocks=2,  # NOTE: encoder for 4 game
                num_channels=256,
                reward_head_channels= 16,
                value_head_channels= 16,
                policy_head_channels= 16,
                fc_reward_layers= [32],
                fc_value_layers= [32],
                fc_policy_layers= [32],
                observation_shape=(4, 96, 96),
                frame_stack_num=4,
                gray_scale=True,
                action_space_size=action_space_size,
                norm_type=norm_type,
                model_type='conv',
                image_channel=1,
                downsample=True,
                self_supervised_learning_loss=True,
                discrete_action_encoding_type='one_hot',
                use_sim_norm=True,
                use_sim_norm_kl_loss=False,
                task_num=len(env_id_list),
            ),
            cuda=True,
            env_type='not_board_games',
            # train_start_after_envsteps=2000,
            train_start_after_envsteps=0,
            game_segment_length=20,  # Fixed segment length as per user config
            random_collect_episode_num=0,
            use_augmentation=True,
            use_priority=False,
            replay_ratio=0.25,
            num_unroll_steps=num_unroll_steps,
            # update_per_collect=2,  # TODO: debug
            update_per_collect=80,  # Consistent with UniZero config
            batch_size=batch_size,
            optim_type='SGD',
            td_steps=5,
            lr_piecewise_constant_decay=True,
            manual_temperature_decay=False,
            learning_rate=0.2,
            target_update_freq=100,
            num_segments=num_segments,
            num_simulations=num_simulations,
            policy_entropy_weight=5e-3, #TODO
            ssl_loss_weight=2,
            eval_freq=int(5e3),
            replay_buffer_size=int(5e5),  # Adjusted as per UniZero config
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            # ============= The key different params for reanalyze =============
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
        ),
    ))

def generate_configs(
    env_id_list,
    action_space_size,
    collector_env_num,
    evaluator_env_num,
    n_episode,
    num_simulations,
    reanalyze_ratio,
    batch_size,
    num_unroll_steps,
    infer_context_length,
    norm_type,
    seed,
    buffer_reanalyze_freq,
    reanalyze_batch_size,
    reanalyze_partition,
    num_segments
):
    configs = []
    exp_name_prefix = (
        f'data_muzero_mt_8games_ddp_8gpu/{len(env_id_list)}games_brf{buffer_reanalyze_freq}/'
        f'{len(env_id_list)}games_brf{buffer_reanalyze_freq}_1-encoder-{norm_type}-res2-channel256_gsl20_'
        f'{len(env_id_list)}-pred-head_mbs-512_upc80_H{num_unroll_steps}_seed{seed}/'
    )

    for task_id, env_id in enumerate(env_id_list):
        config = create_config(
            env_id,
            action_space_size,
            # collector_env_num if env_id not in ['PongNoFrameskip-v4', 'BoxingNoFrameskip-v4'] else 2,  # TODO: different collector_env_num for Pong and Boxing
            # evaluator_env_num if env_id not in ['PongNoFrameskip-v4', 'BoxingNoFrameskip-v4'] else 2,
            # n_episode if env_id not in ['PongNoFrameskip-v4', 'BoxingNoFrameskip-v4'] else 2,
            collector_env_num,
            evaluator_env_num,
            n_episode,
            num_simulations,
            reanalyze_ratio,
            batch_size,
            num_unroll_steps,
            infer_context_length,
            norm_type,
            buffer_reanalyze_freq,
            reanalyze_batch_size,
            reanalyze_partition,
            num_segments
        )
        config.policy.task_id = task_id
        config.exp_name = f"{exp_name_prefix}{env_id.split('NoFrameskip')[0]}_muzero-mt_seed{seed}"

        configs.append([task_id, [config, create_env_manager()]])

    return configs

def create_env_manager():
    return EasyDict(dict(
        env=dict(
            type='atari_lightzero',
            import_names=['zoo.atari.envs.atari_lightzero_env'],
        ),
        # env_manager=dict(type='subprocess'),
        env_manager=dict(type='base'),
        policy=dict(
            type='muzero_multitask',
            import_names=['lzero.policy.muzero_multitask'],
        ),
    ))

if __name__ == "__main__":
    # import sys
    # sys.path.insert(0, "/Users/puyuan/code/LightZero")
    # import lzero
    # print("lzero path:", lzero.__file__)

    # Define your list of environment IDs
    env_id_list = [
        'PongNoFrameskip-v4',
        'MsPacmanNoFrameskip-v4',
        'SeaquestNoFrameskip-v4',
        'BoxingNoFrameskip-v4',
        'AlienNoFrameskip-v4',
        'ChopperCommandNoFrameskip-v4',
        'HeroNoFrameskip-v4',
        'RoadRunnerNoFrameskip-v4',
    ]
    # env_id_list = [
    #     'PongNoFrameskip-v4',
    #     'MsPacmanNoFrameskip-v4',
    # ]

    action_space_size = 18  # Full action space, adjust if different per env
    seed = 0
    collector_env_num = 8
    evaluator_env_num = 3
    num_segments = 8
    n_episode = 8
    num_simulations = 50
    reanalyze_ratio = 0.0
    max_env_step = 5e5

    max_batch_size = 512
    batch_size = [int(min(64, max_batch_size / len(env_id_list))) for _ in range(len(env_id_list))]
    print(f'=========== batch_size: {batch_size} ===========')

    num_unroll_steps = 5
    infer_context_length = 4
    # norm_type = 'LN'
    norm_type = 'BN'

    buffer_reanalyze_freq = 1 / 50  # Adjusted as per UniZero config
    reanalyze_batch_size = 160
    reanalyze_partition = 0.75

    num_segments = 8

    # =========== TODO: debug ===========
    # collector_env_num = 2
    # evaluator_env_num = 2
    # num_segments = 2
    # n_episode = 2
    # num_simulations = 5
    # batch_size = [int(min(2, max_batch_size / len(env_id_list))) for _ in range(len(env_id_list))]


    # Generate configurations
    configs = generate_configs(
        env_id_list=env_id_list,
        action_space_size=action_space_size,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_episode=n_episode,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        batch_size=batch_size,
        num_unroll_steps=num_unroll_steps,
        infer_context_length=infer_context_length,
        norm_type=norm_type,
        seed=seed,
        buffer_reanalyze_freq=buffer_reanalyze_freq,
        reanalyze_batch_size=reanalyze_batch_size,
        reanalyze_partition=reanalyze_partition,
        num_segments=num_segments
    )

    """
    Overview:
        This script should be executed with <nproc_per_node> GPUs.
        Run the following command to launch the script:
        export NCCL_TIMEOUT=3600000
        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 ./zoo/atari/config/atari_muzero_multitask_segment_8games_ddp_config.py
        或者使用 torchrun:
        torchrun --nproc_per_node=8 ./zoo/atari/config/atari_muzero_multitask_segment_8games_ddp_config.py
    """
    from lzero.entry import train_muzero_multitask_segment_ddp
    from ding.utils import DDPContext
    with DDPContext():
        train_muzero_multitask_segment_ddp(configs, seed=seed, max_env_step=max_env_step)