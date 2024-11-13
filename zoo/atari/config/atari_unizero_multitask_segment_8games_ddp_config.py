from easydict import EasyDict
from copy import deepcopy
# from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map

def create_config(env_id, action_space_size, collector_env_num, evaluator_env_num, n_episode, num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type, buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition, num_segments, total_batch_size):
    return EasyDict(dict(
        env=dict(
            stop_value=int(1e6),
            env_id=env_id,
            observation_shape=(3, 64, 64),
            # observation_shape=(3, 96, 96),
            gray_scale=False,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
            full_action_space=True,
            # ===== only for debug =====
            # collect_max_episode_steps=int(30),
            # eval_max_episode_steps=int(30),
            # collect_max_episode_steps=int(150), # TODO: DEBUG
            # eval_max_episode_steps=int(150),
            # collect_max_episode_steps=int(500),
            # eval_max_episode_steps=int(500),
        ),
        policy=dict(
            multi_gpu=True, # ======== Very important for ddp =============
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=200000,),),),  # default is 10000
            grad_correct_params=dict(
                # for MoCo
                MoCo_beta=0.5,
                MoCo_beta_sigma=0.5,
                MoCo_gamma=0.1,
                MoCo_gamma_sigma=0.5,
                MoCo_rho=0,
                # for CAGrad
                calpha=0.5,
                rescale=1,
            ),
            task_num=len(env_id_list), # ======  在ddp中需要替换为每个rank对应的task数量  ======
            task_id=0,
            model=dict(
                observation_shape=(3, 64, 64),
                # observation_shape=(3, 96, 96),
                action_space_size=action_space_size,
                norm_type=norm_type,
                # num_res_blocks=1, # NOTE: encoder for 1 game
                # num_channels=64,
                num_res_blocks=2,  # NOTE: encoder for 4 game
                # num_channels=128,
                # num_res_blocks=4,  # NOTE: encoder for 8 game
                # num_res_blocks=2,  # NOTE: encoder for 8 game
                num_channels=256,
                world_model_cfg=dict(
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    # device='cpu',  # 'cuda',
                    device='cuda',  # 'cuda',
                    action_space_size=action_space_size,
                    # num_layers=2,  # NOTE
                    num_layers=4,  # NOTE: transformer
                    num_heads=8,

                    # num_layers=8,  # NOTE: gato-79M transformer
                    # num_heads=24,

                    embed_dim=768,
                    obs_type='image',
                    # env_num=max(collector_env_num, evaluator_env_num),
                    env_num=8,  # TODO: the max of all tasks
                    # collector_env_num=collector_env_num,
                    # evaluator_env_num=evaluator_env_num,
                    task_num=len(env_id_list), # ====== total_task_num ======
                    use_normal_head=True,
                    # use_normal_head=False,
                    use_softmoe_head=False,
                    # use_moe_head=True,
                    use_moe_head=False,
                    num_experts_in_moe_head=4,  # NOTE
                    # moe_in_transformer=True,
                    moe_in_transformer=False,  # NOTE
                    # multiplication_moe_in_transformer=True,
                    multiplication_moe_in_transformer=False,  # NOTE
                    num_experts_of_moe_in_transformer=4,
                    # num_experts_of_moe_in_transformer=2,
                ),
            ),
            total_batch_size=total_batch_size, #TODO=======
            allocated_batch_sizes=True,
            train_start_after_envsteps=int(0), # TODO
            use_priority=False,
            # print_task_priority_logs=False,
            # use_priority=True,  # TODO
            print_task_priority_logs=False,
            cuda=True,
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            game_segment_length=20,
            # update_per_collect=None,
            # update_per_collect=40, # TODO: 4/8games max-bs=64*4 8*20*0.25
            update_per_collect=80, # TODO: 26games max-bs=400, 8*20*1=160
            # update_per_collect=2, # TODO: 26games max-bs=400, 8*20*1=160
            replay_ratio=0.25,
            batch_size=batch_size,
            optim_type='AdamW',
            num_segments=num_segments,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            n_episode=n_episode,
            # replay_buffer_size=int(1e6),
            replay_buffer_size=int(5e5), # TODO
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
    ))

def generate_configs(env_id_list, action_space_size, collector_env_num, n_episode, evaluator_env_num, num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type, seed, buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition, num_segments, total_batch_size):
    configs = []
    # TODO
    # exp_name_prefix = f'data_unizero_mt_0711/{len(env_id_list)}games_{"-".join(env_id_list)}_1-head-softmoe4_1-encoder-{norm_type}_lsd768-nlayer4-nh8_seed{seed}/'
    # exp_name_prefix = f'data_unizero_mt_0716/{len(env_id_list)}games_4-head_1-encoder-{norm_type}_MoCo_lsd768-nlayer4-nh8_max-bs1500_seed{seed}/'
    # exp_name_prefix = f'data_unizero_mt_0719/{len(env_id_list)}games_pong-boxing-envnum2_4-head_1-encoder-{norm_type}_trans-ffw-moe4_lsd768-nlayer2-nh8_max-bs1500_upc1000_seed{seed}/'
    # exp_name_prefix = f'data_unizero_mt_0716/{len(env_id_list)}games_1-head_1-encoder-{norm_type}_trans-ffw-moe4_lsd768-nlayer4-nh8_max-bs1500_seed{seed}/'
    # exp_name_prefix = f'data_unizero_mt_0722_debug/{len(env_id_list)}games_1-encoder-{norm_type}_trans-ffw-moeV2-expert4_4-head_lsd768-nlayer2-nh8_max-bs2000_upc1000_seed{seed}/'
    # exp_name_prefix = f'data_unizero_mt_0722_profile/lineprofile_{len(env_id_list)}games_1-encoder-{norm_type}_4-head_lsd768-nlayer2-nh8_max-bs2000_upc1000_seed{seed}/'
    # exp_name_prefix = f'data_unizero_mt_segcollect_1104/{len(env_id_list)}games_1-encoder-{norm_type}-res2-channel128_gsl20_4-head_lsd768-nlayer4-nh8_max-bs64*4_upc40_seed{seed}/'
    # exp_name_prefix = f'data_unizero_mt_segcollect_1104/{len(env_id_list)}games_1-encoder-{norm_type}-res2-channel256_gsl20_8-head_lsd768-nlayer4-nh8_max-bs32*8_upc40_seed{seed}/'
    exp_name_prefix = f'data_unizero_mt_segcollect_ddp8gpu_fixlearnlog_1113_adaptivebs_100epoch-clip1-4/{len(env_id_list)}games_brf{buffer_reanalyze_freq}/{len(env_id_list)}games_brf{buffer_reanalyze_freq}_1-encoder-{norm_type}-res2-channel256_gsl20_{len(env_id_list)}-pred-head_lsd768-nlayer4-nh8_mbs-512-bs64_upc80_seed{seed}/'
    # exp_name_prefix = f'data_unizero_mt_segcollect_1104/{len(env_id_list)}games_1-encoder-{norm_type}_gsl20_8-head_lsd768-nlayer4-nh8_max-bs64*8_upc40_seed{seed}/'


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
            num_segments,
            total_batch_size
        )
        config.policy.task_id = task_id
        config.exp_name = exp_name_prefix + f"{env_id.split('NoFrameskip')[0]}_unizero-mt_seed{seed}"

        configs.append([task_id, [config, create_env_manager()]])
    return configs


def create_env_manager():
    return EasyDict(dict(
        env=dict(
            type='atari_lightzero',
            import_names=['zoo.atari.envs.atari_lightzero_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(
            type='unizero_multitask',
            import_names=['lzero.policy.unizero_multitask'],
        ),
    ))

if __name__ == "__main__":
    from lzero.entry import train_unizero_multitask_segment
    # TODO
    # env_id_list = [
    #     'PongNoFrameskip-v4',
    #     'MsPacmanNoFrameskip-v4',
    #     'SeaquestNoFrameskip-v4',
    #     'BoxingNoFrameskip-v4'
    # ]

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

    # 26games
    # env_id_list = [
    #     'PongNoFrameskip-v4',
    #     'MsPacmanNoFrameskip-v4',
    #     'SeaquestNoFrameskip-v4',
    #     'BoxingNoFrameskip-v4',
    #     'AlienNoFrameskip-v4',
    #     'ChopperCommandNoFrameskip-v4',
    #     'HeroNoFrameskip-v4',
    #     'RoadRunnerNoFrameskip-v4',

    #     'AmidarNoFrameskip-v4',
    #     'AssaultNoFrameskip-v4',
    #     'AsterixNoFrameskip-v4',
    #     'BankHeistNoFrameskip-v4',
    #     'BattleZoneNoFrameskip-v4',
    #     'CrazyClimberNoFrameskip-v4',
    #     'DemonAttackNoFrameskip-v4',
    #     'FreewayNoFrameskip-v4',
    #     'FrostbiteNoFrameskip-v4',
    #     'GopherNoFrameskip-v4',
    #     'JamesbondNoFrameskip-v4',
    #     'KangarooNoFrameskip-v4',
    #     'KrullNoFrameskip-v4',
    #     'KungFuMasterNoFrameskip-v4',
    #     'PrivateEyeNoFrameskip-v4',
    #     'UpNDownNoFrameskip-v4',
    #     'QbertNoFrameskip-v4',
    #     'BreakoutNoFrameskip-v4',
    # ]


    action_space_size = 18  # Full action space
    seed = 0
    collector_env_num = 8
    num_segments = 8
    n_episode = 8
    evaluator_env_num = 3
    num_simulations = 50
    # max_env_step = int(1e6)
    max_env_step = int(5e5) # TODO

    reanalyze_ratio = 0.
    # batch_size = [32, 32, 32, 32]
    # total_batch_size = 2048

    #应该根据一个样本sequence的占用显存量，和最大显存来设置
    total_batch_size = 512
    # total_batch_size = 3600
    batch_size = [int(min(64, total_batch_size / len(env_id_list))) for _ in range(len(env_id_list))]
    print(f'=========== batch_size: {batch_size} ===========')
    # batch_size = [int(64) for i in range(len(env_id_list))]

    num_unroll_steps = 10
    infer_context_length = 4
    norm_type = 'LN'
    # # norm_type = 'BN'  # bad performance now

    # Defines the frequency of reanalysis. E.g., 1 means reanalyze once per epoch, 1/10 means reanalyze once every ten epochs.
    # buffer_reanalyze_freq = 1/50 # TODO
    buffer_reanalyze_freq = 1/100000
    # buffer_reanalyze_freq = 1/10
    # Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
    reanalyze_batch_size = 160
    # The partition of reanalyze. E.g., 1 means reanalyze_batch samples from the whole buffer, 0.5 means samples from the first half of the buffer.
    reanalyze_partition = 0.75

    # ======== TODO: only for debug ========
    # collector_env_num = 2
    # num_segments = 2
    # n_episode = 2
    # evaluator_env_num = 2
    # num_simulations = 2
    # batch_size = [4, 4, 4, 4, 4, 4, 4, 4]

    configs = generate_configs(env_id_list, action_space_size, collector_env_num, n_episode, evaluator_env_num, num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type, seed, buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition, num_segments, total_batch_size)

    """
    Overview:
        This script should be executed with <nproc_per_node> GPUs.
        Run the following command to launch the script:
        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29503 ./zoo/atari/config/atari_unizero_multitask_segment_8games_ddp_config.py
        torchrun --nproc_per_node=8 ./zoo/atari/config/atari_unizero_multitask_segment_8games_ddp_config.py
    """
    from ding.utils import DDPContext
    from easydict import EasyDict
    with DDPContext():
        train_unizero_multitask_segment(configs, seed=seed, max_env_step=max_env_step)