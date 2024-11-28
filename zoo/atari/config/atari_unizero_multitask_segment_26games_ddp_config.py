from easydict import EasyDict

def create_config(env_id, action_space_size, collector_env_num, evaluator_env_num, n_episode, num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type, buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition, num_segments):
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
            collect_max_episode_steps=int(5e3), # TODO ===========
            eval_max_episode_steps=int(5e3), # TODO ===========
            # eval_max_episode_steps=int(1e4), # TODO ===========
            # ===== only for debug =====
            # collect_max_episode_steps=int(30),
            # eval_max_episode_steps=int(30),
            # collect_max_episode_steps=int(50),
            # eval_max_episode_steps=int(50),
            # collect_max_episode_steps=int(500),
            # eval_max_episode_steps=int(500),
        ),
        policy=dict(
            multi_gpu=True, # ======== Very important for ddp =============
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=50000,),),),  # replay_ratio=0.5时，大约对应100k envsteps 存储一次 default is 10000
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
            task_num=len(env_id_list),
            task_id=0,
            model=dict(
                observation_shape=(3, 64, 64),
                # observation_shape=(3, 96, 96),
                action_space_size=action_space_size,
                norm_type=norm_type,
                # num_res_blocks=1, # NOTE: encoder for 1 game
                # num_channels=64,
                # num_res_blocks=2,  # NOTE: encoder for 4 game
                # num_channels=128,
                # num_res_blocks=4,  # NOTE: encoder for 8 game
                num_res_blocks=2,  # NOTE: encoder for 8 game
                num_channels=256,
                world_model_cfg=dict(
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    # device='cpu',  # 'cuda',
                    device='cuda',  # 'cuda',
                    action_space_size=action_space_size,
                    # num_layers=2,  # NOTE
                    # NOTE: rl transformer
                    # batch_size=64 8games训练时，每张卡大约占12G cuda存储, mbs=512
                    # num_layers=4,  
                    # num_heads=8,   
                    # embed_dim=768,

                    # NOTE: gato-79M (small) transformer
                    # batch_size=64 8games训练时，每张卡大约占12*2=24G cuda存储
                    # num_layers=8,  
                    # num_heads=24,
                    # embed_dim=768,

                    # NOTE: gato-medium 修改版 transformer
                    # batch_size=32 ====== TODO======
                    # num_layers=12,  
                    # num_heads=24,
                    # embed_dim=768,

                    # NOTE: gato-medium 修改版 transformer
                    # batch_size=64 8games训练时，每张卡大约占12*2*4 cuda存储
                    # batch_size=32 8games训练时，每张卡大约占12*2*4/2 cuda存储
                    num_layers=8,  
                    num_heads=24,
                    embed_dim=1536,

                    # NOTE: gato-364M (medium) transformer
                    # batch_size=64 8games训练时，每张卡大约占12*3*4 cuda存储
                    # num_layers=12,  
                    # num_heads=12,
                    # embed_dim=1536,

                    # n_layer=12, 
                    # n_head=12,  # gpt2-base 124M parameters
                    # embed_dim=768,

                    obs_type='image',
                    # env_num=max(collector_env_num, evaluator_env_num),
                    env_num=8,  # TODO: the max of all tasks
                    # collector_env_num=collector_env_num,
                    # evaluator_env_num=evaluator_env_num,
                    task_num=len(env_id_list),
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
            # allocated_batch_sizes=True,#TODO=======
            allocated_batch_sizes=False,#TODO=======
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
            # update_per_collect=160, # TODO: 26games max-bs=400, 8*20*1=160
            update_per_collect=80, # TODO: 26games max-bs=400, 8*20*1=160
            # update_per_collect=2, # TODO: debug
            replay_ratio=0.25,
            batch_size=batch_size,
            optim_type='AdamW',
            num_segments=num_segments,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            n_episode=n_episode,
            replay_buffer_size=int(5e5), # TODO
            eval_freq=int(2e4),
            # eval_freq=int(1),
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

def generate_configs(env_id_list, action_space_size, collector_env_num, n_episode, evaluator_env_num, num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type, seed, buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition, num_segments):
    configs = []
    # TODO
    # exp_name_prefix = f'data_unizero_mt_ddp-8gpu-26game_1127/{len(env_id_list)}games_brf{buffer_reanalyze_freq}_nlayer4-nhead8_eval10min_seed{seed}/{len(env_id_list)}games_brf{buffer_reanalyze_freq}_1-encoder-{norm_type}-res2-channel256_gsl20_{len(env_id_list)}-pred-head_lsd768-nlayer4-nh8_mbs-512-bs64_upc80_seed{seed}/'
    
    # exp_name_prefix = f'data_unizero_mt_ddp-8gpu-26game_1127/{len(env_id_list)}games_brf{buffer_reanalyze_freq}_nlayer8-nhead24_seed{seed}/{len(env_id_list)}games_brf{buffer_reanalyze_freq}_1-encoder-{norm_type}-res2-channel256_gsl20_{len(env_id_list)}-pred-head_lsd768-nlayer8-nh24_mbs-512-bs64_upc80_seed{seed}/'
    exp_name_prefix = f'data_unizero_mt_ddp-8gpu-26game_1127/{len(env_id_list)}games_eval60min_brf{buffer_reanalyze_freq}_nlayer8-nh24-embed1536_seed{seed}/{len(env_id_list)}games_brf{buffer_reanalyze_freq}_1-encoder-{norm_type}-res2-channel256_gsl20_{len(env_id_list)}-pred-head_nlayer8-nh24-embed1536_mbs-450-bs32_upc80_seed{seed}/'


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

    # env_id_list = [
    #     'PongNoFrameskip-v4',
    #     'MsPacmanNoFrameskip-v4',
    #     'SeaquestNoFrameskip-v4',
    #     'BoxingNoFrameskip-v4',
    #     'AlienNoFrameskip-v4',
    #     'ChopperCommandNoFrameskip-v4',
    #     'HeroNoFrameskip-v4',
    #     'RoadRunnerNoFrameskip-v4',
    # ]

    # 26games
    env_id_list = [
        'PongNoFrameskip-v4',
        'MsPacmanNoFrameskip-v4',
        'SeaquestNoFrameskip-v4',
        'BoxingNoFrameskip-v4',
        'AlienNoFrameskip-v4',
        'ChopperCommandNoFrameskip-v4',
        'HeroNoFrameskip-v4',
        'RoadRunnerNoFrameskip-v4',

        'AmidarNoFrameskip-v4',
        'AssaultNoFrameskip-v4',
        'AsterixNoFrameskip-v4',
        'BankHeistNoFrameskip-v4',
        'BattleZoneNoFrameskip-v4',
        'CrazyClimberNoFrameskip-v4',
        'DemonAttackNoFrameskip-v4',
        'FreewayNoFrameskip-v4',
        'FrostbiteNoFrameskip-v4',
        'GopherNoFrameskip-v4', # 17
        'JamesbondNoFrameskip-v4', # 18
        'KangarooNoFrameskip-v4', # 19
        'KrullNoFrameskip-v4',
        'KungFuMasterNoFrameskip-v4',
        'PrivateEyeNoFrameskip-v4',
        'UpNDownNoFrameskip-v4',
        'QbertNoFrameskip-v4',
        'BreakoutNoFrameskip-v4',
    ]


    action_space_size = 18  # Full action space

    # TODO ==========
    import os 
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"
    # os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_TIMEOUT"] = "3600000000"

    for seed in [0,1]: # TODO
        collector_env_num = 8
        num_segments = 8
        n_episode = 8
        evaluator_env_num = 3
        num_simulations = 50
        max_env_step = int(5e5) # TODO
        reanalyze_ratio = 0.

        # for layer12        
        total_batch_size = 450
        batch_size = [int(min(32, total_batch_size/len(env_id_list))) for i in range(len(env_id_list))]

        # total_batch_size = 512
        # batch_size = [int(min(64, total_batch_size/len(env_id_list))) for i in range(len(env_id_list))]
        # print(f'=========== batch_size: {batch_size} ===========')

        num_unroll_steps = 10
        infer_context_length = 4
        norm_type = 'LN'
        # # norm_type = 'BN'  # bad performance now

        # Defines the frequency of reanalysis. E.g., 1 means reanalyze once per epoch, 2 means reanalyze once every two epochs.
        buffer_reanalyze_freq = 1/50 # TODO
        # buffer_reanalyze_freq = 1/20 # TODO
        # buffer_reanalyze_freq = 1/30 # TODO

        # buffer_reanalyze_freq = 1/100000
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
        # batch_size = [2 for i in range(26)]

        configs = generate_configs(env_id_list, action_space_size, collector_env_num, n_episode, evaluator_env_num, num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type, seed, buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition, num_segments)

        """
        Overview:
            This script should be executed with <nproc_per_node> GPUs.
            Run the following command to launch the script:
            export NCCL_TIMEOUT=3600000  # NOTE
            python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 ./zoo/atari/config/atari_unizero_multitask_segment_26games_ddp_config.py
            torchrun --nproc_per_node=8 ./zoo/atari/config/atari_unizero_multitask_segment_26games_ddp_config.py
        """
        from ding.utils import DDPContext
        from easydict import EasyDict
        with DDPContext():
            train_unizero_multitask_segment(configs, seed=seed, max_env_step=max_env_step)