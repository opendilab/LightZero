from easydict import EasyDict

import math

def compute_batch_config(env_id_list, effective_batch_size):
    n = len(env_id_list)
    
    # 根据环境数量设定有效 batch size 和每个环境的最大微 batch size
    gpu_num = 8 # TODO：修改
    if n<=8:
        max_micro_batch_one_gpu = 400
    else:
        max_micro_batch_one_gpu = 400

    # max_micro_batch = int(max_micro_batch_one_gpu / (n // gpu_num))

    div = max(1, math.ceil(n / gpu_num))  # 至少分到 1, 否则除 0
    max_micro_batch = max_micro_batch_one_gpu // div

    # 计算每个环境理论上应该分得的 batch size
    theoretical_env_batch = effective_batch_size / n
    
    if theoretical_env_batch > max_micro_batch:
        # 当每个环境按均分的 batch 大于允许的最大微 batch 时，
        # 则令每个环境的实际微 batch size 固定为 max_micro_batch
        micro_batch_size = max_micro_batch
        # 梯度累计步数 = ceil(每个环境理论 batch size / 最大微 batch size)
        grad_accumulate_steps = math.ceil(theoretical_env_batch / max_micro_batch)
    else:
        # 否则直接使用计算出的理论 batch size（这里向下取整以保证整数）
        micro_batch_size = int(theoretical_env_batch)
        grad_accumulate_steps = 1
    
    # 为每个环境分配相同的微 batch size
    batch_size = [micro_batch_size for _ in range(n)]
    
    # 打印一些调试信息（也可以记录到 log 中）
    print("环境数量: {}".format(n))
    print("有效 total batch size: {}".format(effective_batch_size))
    print("每个环境的理论 batch size: {:.2f}".format(theoretical_env_batch))
    print("每个环境的微 batch size: {}".format(micro_batch_size))
    print("梯度累积步数: {}".format(grad_accumulate_steps))
    
    return batch_size, grad_accumulate_steps

def create_config(env_id, action_space_size, collector_env_num, evaluator_env_num, n_episode,
                  num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length,
                  norm_type, buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition, num_segments,
                  total_batch_size, num_layers):
    return EasyDict(dict(
        env=dict(
            stop_value=int(1e6),
            env_id=env_id,
            observation_shape=(3, 64, 64),
            gray_scale=False,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False),
            full_action_space=True,
            collect_max_episode_steps=int(5e3),
            eval_max_episode_steps=int(5e3),
            # ===== only for debug =====
            # collect_max_episode_steps=int(200),
            # eval_max_episode_steps=int(200),
        ),
        policy=dict(
            multi_gpu=True,  # Very important for ddp
            only_use_moco_stats=False,
            use_moco=False,  # ==============TODO==============
            # use_moco=True,  # ==============TODO: moco==============
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=200000))),
            grad_correct_params=dict(
                MoCo_beta=0.5, MoCo_beta_sigma=0.5, MoCo_gamma=0.1, MoCo_gamma_sigma=0.5, MoCo_rho=0,
                calpha=0.5, rescale=1,
            ),
            # moco_version="v0",
            moco_version="v1",  # ==============TODO: moco==============
            total_task_num=len(env_id_list),
            task_num=len(env_id_list),
            task_id=0,
            model=dict(
                observation_shape=(3, 64, 64),
                action_space_size=action_space_size,
                norm_type=norm_type,
                num_res_blocks=2,
                num_channels=256,
                # num_channels=512, # ==============TODO==============
                continuous_action_space=False,
                world_model_cfg=dict(
                    # use_global_pooling=True,
                    use_global_pooling=False,

                    final_norm_option_in_obs_head='LayerNorm', # ==============TODO:orig==============
                    final_norm_option_in_encoder='LayerNorm',
                    predict_latent_loss_type='mse', # TODO: for latent state layer_norm
                                        
                    # final_norm_option_in_obs_head='SimNorm',
                    # final_norm_option_in_encoder='SimNorm',
                    # predict_latent_loss_type='group_kl', # TODO: only for latent state sim_norm
                   
                    # share_head=True, # TODO
                    share_head=False, # TODO

                    analysis_dormant_ratio_weight_rank=True,  # ==============TODO==============
                    # analysis_dormant_ratio_weight_rank=False, # TODO
                    # analysis_dormant_ratio_interval=100,
                    analysis_dormant_ratio_interval=5000,
                    # analysis_dormant_ratio_interval=20,

                    continuous_action_space=False,
                                        
                    task_embed_option=None,   # ==============TODO:orig==============
                    use_task_embed=False, # ==============TODO==============

                    # task_embed_option='concat_task_embed', 
                    # use_task_embed=True, # ==============TODO: taskembed128==============
                    # task_embed_dim=128,

                    use_shared_projection=False,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    device='cuda',
                    action_space_size=action_space_size,
                    # batch_size=64 8games训练时，每张卡大约占 12*3=36G cuda显存
                    # num_layers=12,
                    # num_heads=24,

                    num_layers=num_layers,
                    # num_layers=8,
                    # num_layers=12, # todo
                    num_heads=24,

                    embed_dim=768,
                    obs_type='image',
                    env_num=8,
                    task_num=len(env_id_list),
                    encoder_type='vit', # =======TODO: vit=======
                    # encoder_type='resnet', # ==============TODO:orig==============

                    use_normal_head=True,
                    use_softmoe_head=False,
                    use_moe_head=False,
                    num_experts_in_moe_head=4,

                    moe_in_transformer=False,
                    # multiplication_moe_in_transformer=False, # ==============TODO:orig==============
                    multiplication_moe_in_transformer=True, # =======TODO: moe8=======
                    n_shared_experts=1,
                    num_experts_per_tok=1,
                    num_experts_of_moe_in_transformer=8,

                    # LoRA 参数：
                    moe_use_lora=False, # TDO
                    lora_r= 0,
                    lora_alpha =1,
                    lora_dropout= 0.0,
                ),
            ),
            use_task_exploitation_weight=False, # TODO
            task_complexity_weight=False, # TODO
            total_batch_size=total_batch_size,
            allocated_batch_sizes=False,
            train_start_after_envsteps=int(0), # TODO: DEBUG
            # train_start_after_envsteps=int(2000),
            use_priority=False,
            print_task_priority_logs=False,
            cuda=True,
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            game_segment_length=20,
            # update_per_collect=160, # TODO: replay_ratio=1  20*8*1=160 not-use now
            update_per_collect=80, # TODO: replay_ratio=0.5  20*8*0.5=80 atari8-nlayer8 atari26
            # update_per_collect=40, # TODO: replay_ratio=0.25  20*8*0.25=40  atari8-nlayer4
            # update_per_collect=2, # TODO: only for debug
            replay_ratio=0.25,
            batch_size=batch_size,
            optim_type='AdamW',
            # cos_lr_scheduler=True, # TODO
            cos_lr_scheduler=False,
            num_segments=num_segments,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            n_episode=n_episode,
            replay_buffer_size=int(5e5),
            # eval_freq=int(1e4), # TODO: 8games
            eval_freq=int(2e4),  # TODO: 26games
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
        ),
    ))

def generate_configs(env_id_list, action_space_size, collector_env_num, n_episode, evaluator_env_num,
                     num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length,
                     norm_type, seed, buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition,
                     num_segments, total_batch_size, num_layers):
    configs = []
    # ===== only for debug =====
    # exp_name_prefix = f'data_unizero_atari_mt_20250522_debug/atari_{len(env_id_list)}games_orig_simnorm-kl_vit_moe8_moco-v1_tran-nlayer{num_layers}_brf{buffer_reanalyze_freq}_not-share-head_seed{seed}/'


    # ========= TODO: global BENCHMARK_NAME =========
    # exp_name_prefix = f'data_unizero_atari_mt_20250527/atari_{len(env_id_list)}games_orig_simnorm-kl_vit_moe8_tran-nlayer{num_layers}_brf{buffer_reanalyze_freq}_not-share-head_seed{seed}/'

    # exp_name_prefix = f'data_unizero_atari_mt_20250522/atari_{len(env_id_list)}games_orig_tran-nlayer{num_layers}_brf{buffer_reanalyze_freq}_not-share-head_seed{seed}/'
    # exp_name_prefix = f'data_unizero_atari_mt_20250527/atari_{len(env_id_list)}games_orig_simnorm-kl_vit_moco-v2_tran-nlayer{num_layers}_brf{buffer_reanalyze_freq}_not-share-head_seed{seed}/'

    exp_name_prefix = f'data_unizero_atari_mt_20250601/atari_{len(env_id_list)}games_orig_vit_ln-mse_moe8_tran-nlayer{num_layers}_brf{buffer_reanalyze_freq}_not-share-head_seed{seed}/'

    # exp_name_prefix = f'data_unizero_atari_mt_20250521/atari_{len(env_id_list)}games_orig_simnorm-kl_vit_moe8_taskembed128_tran-nlayer{num_layers}_rr1_brf{buffer_reanalyze_freq}_not-share-head_seed{seed}/'

    # exp_name_prefix = f'data_unizero_atari_mt_20250521/atari_{len(env_id_list)}games_orig_tran-nlayer{num_layers}_rr1_brf{buffer_reanalyze_freq}_not-share-head_seed{seed}/'

    # exp_name_prefix = f'data_unizero_atari_mt_20250521/atari_{len(env_id_list)}games_orig_simnorm-kl_vit_moe8_moco_tran-nlayer4_rr025_brf{buffer_reanalyze_freq}_not-share-head_seed{seed}/'

    # exp_name_prefix = f'data_lz/data_unizero_atari_mt_20250508/atari_{len(env_id_list)}games_orig_simnorm_tran-nlayer{num_layers}_brf{buffer_reanalyze_freq}_not-share-head_seed{seed}/'

    # exp_name_prefix = f'data_lz/data_unizero_atari_mt_20250508/atari_{len(env_id_list)}games_vit_simnorm_tran-nlayer{num_layers}-moe8_brf{buffer_reanalyze_freq}_not-share-head_seed{seed}/'
  

    for task_id, env_id in enumerate(env_id_list):
        config = create_config(
            env_id, action_space_size, collector_env_num, evaluator_env_num, n_episode, num_simulations,
            reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type,
            buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition, num_segments, total_batch_size, num_layers
        )
        config.policy.task_id = task_id
        config.exp_name = exp_name_prefix + f"{env_id.split('NoFrameskip')[0]}_seed{seed}"
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
    """
    Overview:
        This script should be executed with <nproc_per_node> GPUs.
        Run the following command to launch the script:

        =========== volce atari8 =========================
        cd /fs-computility/niuyazhe/puyuan/code/LightZero/
        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 /fs-computility/niuyazhe/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py 2>&1 | tee /fs-computility/niuyazhe/puyuan/code/LightZero/log/20250509/uz_mt_atari26_orig_vit_ln-mse_moe8_nlayer8_brf002_seed12.log


        =========== cpfs atari8 =========================
        cd /cpfs04/user/puyuan/code/LightZero/
        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 /cpfs04/user/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py 2>&1 | tee /cpfs04/user/puyuan/code/LightZero/log/20250522_cpfs/uz_mt_atari26_orig_simnorm-kl_vit_moe8_nlayer8_brf002_seed01.log

        python -m torch.distributed.launch --nproc_per_node=2 --master_port=29502 /cpfs04/user/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py 2>&1 | tee /cpfs04/user/puyuan/code/LightZero/log/20250522_cpfs/uz_mt_atari8_orig_simnorm-kl_vit_moe8_moco-v1_nlayer4_brf0_seed01.log

        # python -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 /cpfs04/user/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py 2>&1 | tee /cpfs04/user/puyuan/code/LightZero/log/20250522_cpfs/uz_mt_atari8_orig_simnorm-kl_vit_moe8_moco-v0_nlayer4_brf0_seed01.log
        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 /cpfs04/user/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py 2>&1 | tee /cpfs04/user/puyuan/code/LightZero/log/20250522_cpfs/uz_mt_atari8_orig_simnorm-kl_vit_moco-v0_nlayer4_brf0_seed01.log
        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 /cpfs04/user/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py 2>&1 | tee /cpfs04/user/puyuan/code/LightZero/log/20250522_cpfs/uz_mt_atari8_orig_simnorm-kl_vit_moco-v1_nlayer4_brf0_seed01.log


        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 /cpfs04/user/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py 2>&1 | tee /cpfs04/user/puyuan/code/LightZero/log/20250522_cpfs/uz_mt_atari8_orig_simnorm-kl_vit_moe8_nlayer8_brf002_seed01.log


        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 /cpfs04/user/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py 2>&1 | tee /cpfs04/user/puyuan/code/LightZero/log/20250522_cpfs/uz_mt_atari26_orig_simnorm-kl_vit_moe8_taskembed128_nlayer8_seed01.log


        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 /cpfs04/user/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py 2>&1 | tee /cpfs04/user/puyuan/code/LightZero/log/20250522_cpfs/uz_mt_atari8_orig_nlayer4_seed01.log

        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 /cpfs04/user/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py 2>&1 | tee /cpfs04/user/puyuan/code/LightZero/log/20250522_cpfs/uz_mt_atari8_orig_simnorm-kl_nlayer4_seed01.log


        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 /cpfs04/user/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py 2>&1 | tee /cpfs04/user/puyuan/code/LightZero/log/20250522_cpfs/uz_mt_atari8_orig_simnorm-kl_vit_moe8_taskembed128_nlayer8_seed01.log

        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 /cpfs04/user/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py
        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 /cpfs04/user/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py 2>&1 | tee /cpfs04/user/puyuan/code/LightZero/log/20250522_cpfs/uz_mt_atari26_orig_simnorm-kl_vit_moe8_taskembed128_nlayer8_rr1_seed01.log

        =========== oss atari26 =========================
        cd /oss/niuyazhe/puyuan/data/data_lz_202505/
        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 /cpfs04/user/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py 2>&1 | tee /cpfs04/user/puyuan/code/LightZero/log/20250522_oss/uz_mt_atari26_orig_nlayer8_seed01.log

        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 /cpfs04/user/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py 2>&1 | tee /cpfs04/user/puyuan/code/LightZero/log/20250522_oss/uz_mt_atari26_orig_simnorm-kl_vit_moe8_taskembed128_nlayer8_seed01.log

        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 /cpfs04/user/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py 2>&1 | tee /cpfs04/user/puyuan/code/LightZero/log/20250522_oss/uz_mt_atari26_orig_nlayer8_rr1_seed01.log

        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 /cpfs04/user/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py 2>&1 | tee /cpfs04/user/puyuan/code/LightZero/log/20250522_oss/uz_mt_atari8_orig_nlayer8_rr05_seed01.log

        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 /cpfs04/user/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py 2>&1 | tee /cpfs04/user/puyuan/code/LightZero/log/20250522_oss/uz_mt_atari8_orig_simnorm-kl_vit_moe8_taskembed128_nlayer4_rr025_seed0.log

        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 /cpfs04/user/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py 2>&1 | tee /cpfs04/user/puyuan/code/LightZero/log/20250522_oss/uz_mt_atari8_orig_simnorm-kl_vit_moe8_moco_nlayer4_rr025_seed0.log

        torchrun --nproc_per_node=8 ./zoo/atari/config/atari_unizero_multitask_segment_8games_ddp_config.py
    """

    from lzero.entry import train_unizero_multitask_segment_ddp
    from ding.utils import DDPContext
    import os


    num_games = 26 # 26 # 8
    num_layers = 8 # ==============TODO==============
    action_space_size = 18
    collector_env_num = 8
    num_segments = 8
    n_episode = 8
    evaluator_env_num = 3
    num_simulations = 50
    max_env_step = int(4e5)
    reanalyze_ratio = 0.0

    if num_games==8:
        env_id_list = [
            'PongNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SeaquestNoFrameskip-v4', 'BoxingNoFrameskip-v4',
            'AlienNoFrameskip-v4', 'ChopperCommandNoFrameskip-v4', 'HeroNoFrameskip-v4', 'RoadRunnerNoFrameskip-v4',
        ]
    elif num_games==26:
        # List of Atari games used for multi-task learning
        env_id_list = [
            'PongNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SeaquestNoFrameskip-v4', 'BoxingNoFrameskip-v4',
            'AlienNoFrameskip-v4', 'ChopperCommandNoFrameskip-v4', 'HeroNoFrameskip-v4', 'RoadRunnerNoFrameskip-v4',
            'AmidarNoFrameskip-v4', 'AssaultNoFrameskip-v4', 'AsterixNoFrameskip-v4', 'BankHeistNoFrameskip-v4',
            'BattleZoneNoFrameskip-v4', 'CrazyClimberNoFrameskip-v4', 'DemonAttackNoFrameskip-v4', 'FreewayNoFrameskip-v4',
            'FrostbiteNoFrameskip-v4', 'GopherNoFrameskip-v4', 'JamesbondNoFrameskip-v4', 'KangarooNoFrameskip-v4',
            'KrullNoFrameskip-v4', 'KungFuMasterNoFrameskip-v4', 'PrivateEyeNoFrameskip-v4', 'UpNDownNoFrameskip-v4',
            'QbertNoFrameskip-v4', 'BreakoutNoFrameskip-v4',
        ]

    if len(env_id_list) == 8:
        if num_layers == 4:
            # effective_batch_size =  1024 # nlayer4 需要设置replay_ratio=0.25对应的upc=40
            effective_batch_size =  512 # nlayer4 需要设置replay_ratio=0.25对应的upc=40 moco

        elif num_layers == 8:
            effective_batch_size = 512 # nlayer8 需要设置replay_ratio=0.5对应的upc=80

    elif len(env_id_list) == 26:
        # effective_batch_size = 832  # cnn-encoder
        # effective_batch_size = 1024  # base-vit-encoder transformer-nlayer4  or cnn-encoder
        effective_batch_size = 512  # base-vit-encoder transformer-nlayer4 transformer-nlayer8 需要设置replay_ratio=0.5对应的upc
        # effective_batch_size = 256   # large-vit-encoder
    elif len(env_id_list) == 18:
        effective_batch_size = 512 * 3  # 1536 
    else:
        raise ValueError("不支持的环境数量: {}".format(n))

    batch_sizes, grad_acc_steps = compute_batch_config(env_id_list, effective_batch_size)
    total_batch_size =  effective_batch_size # 当前无效

    num_unroll_steps = 10
    infer_context_length = 4
    # infer_context_length = 5 # ==============TODO==============

    norm_type = 'LN'
    buffer_reanalyze_freq = 1 / 50
    # buffer_reanalyze_freq = 1 / 1000000
    reanalyze_batch_size = 160
    reanalyze_partition = 0.75

    # ======== TODO: only for debug ========
    # env_id_list = [
    #         'PongNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SeaquestNoFrameskip-v4'
    #     ]
    # num_layers = 1 # ==============TODO==============
    # collector_env_num = 2
    # num_segments = 2
    # n_episode = 2
    # evaluator_env_num = 2
    # num_simulations = 1
    # reanalyze_batch_size = 2
    # num_unroll_steps = 5
    # infer_context_length = 2
    # batch_sizes = [2 for _ in range(len(env_id_list))]
    # total_batch_size =  2*len(env_id_list)


    import torch.distributed as dist
    for seed in [1,2]:
        configs = generate_configs(env_id_list, action_space_size, collector_env_num, n_episode, evaluator_env_num,
                                   num_simulations, reanalyze_ratio, batch_sizes, num_unroll_steps, infer_context_length,
                                   norm_type, seed, buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition,
                                   num_segments, total_batch_size, num_layers)

        with DDPContext():
            train_unizero_multitask_segment_ddp(configs, seed=seed, max_env_step=max_env_step, benchmark_name= "atari" )
            # ======== TODO: only for debug ========
            # train_unizero_multitask_segment_ddp(configs[:2], seed=seed, max_env_step=max_env_step) # train on the first four tasks

            dist.destroy_process_group()
    
