from easydict import EasyDict

import math

def compute_batch_config(env_id_list, effective_batch_size):
    n = len(env_id_list)
    
    # 根据环境数量设定有效 batch size 和每个环境的最大微 batch size
    gpu_num = 8
    max_micro_batch_one_gpu = 400
    max_micro_batch = int(max_micro_batch_one_gpu / (n // gpu_num))

    
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
                  total_batch_size):
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
            # collect_max_episode_steps=int(40),
            # eval_max_episode_steps=int(40),
        ),
        policy=dict(
            multi_gpu=True,  # Very important for ddp
            only_use_moco_stats=False,
            use_moco=False,  # ==============TODO==============
            # use_moco=True,  # ==============TODO==============
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=200000))),
            grad_correct_params=dict(
                MoCo_beta=0.5, MoCo_beta_sigma=0.5, MoCo_gamma=0.1, MoCo_gamma_sigma=0.5, MoCo_rho=0,
                calpha=0.5, rescale=1,
            ),
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
                    use_global_pooling=False,

                    final_norm_option_in_obs_head='LayerNorm',
                    final_norm_option_in_encoder='LayerNorm',
                    predict_latent_loss_type='mse', # TODO: for latent state layer_norm
                                        
                    # final_norm_option_in_obs_head='SimNorm',
                    # final_norm_option_in_encoder='SimNorm',
                    # predict_latent_loss_type='group_kl', # TODO: only for latent state sim_norm
                   
                    # predict_latent_loss_type='group_kl', # TODO: only for latent state sim_norm
                    # share_head=True, # TODO
                    share_head=False, # TODO

                    # analysis_dormant_ratio_weight_rank=True, # TODO
                    analysis_dormant_ratio_weight_rank=False, # TODO
                    dormant_threshold=0.025,
                    continuous_action_space=False,

                    task_embed_option=None,   # ==============TODO: none ==============
                    use_task_embed=False, # ==============TODO==============

                    # task_embed_option='concat_task_embed',   # ==============TODO: none ==============
                    # use_task_embed=True, # ==============TODO==============
                    # task_embed_dim=128,
                    # # task_embed_dim=96,

                    use_shared_projection=False,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    device='cuda',
                    action_space_size=action_space_size,
                    # batch_size=64 8games训练时，每张卡大约占 12*3=36G cuda显存
                    # num_layers=12,
                    # num_heads=24,

                    num_layers=4,  # TODO=======
                    # num_layers=8,
                    num_heads=24,

                    # ===== only for debug =====
                    # num_layers=1,
                    # num_heads=8,

                    embed_dim=768,
                    obs_type='image',
                    env_num=8,
                    task_num=len(env_id_list),

                    encoder_type='vit',
                    # encoder_type='resnet',

                    use_normal_head=True,
                    use_softmoe_head=False,
                    use_moe_head=False,
                    num_experts_in_moe_head=4,

                    moe_in_transformer=False,
                    # multiplication_moe_in_transformer=False,
                    multiplication_moe_in_transformer=True, # TODO=======
                    n_shared_experts=1,
                    num_experts_per_tok=1,
                    num_experts_of_moe_in_transformer=8,

                   # LoRA 参数：
                    # moe_use_lora=False, # TODO
                    moe_use_lora=True, # TODO

                    curriculum_stage_num=curriculum_stage_num,
                    lora_target_modules=["attn", "feed_forward"],
                    lora_r=64, # TODO
                    lora_alpha=32,
                    lora_dropout=0.1,
                    lora_scale_init=1,

                    min_stage0_iters=50000, # 50k
                    max_stage_iters=20000, # 20k
                ),
            ),
            use_task_exploitation_weight=False, # TODO
            # use_task_exploitation_weight=True, # TODO
            target_return =target_return_dict[env_id],
            balance_pipeline=True,
            # task_complexity_weight=False, # TODO
            task_complexity_weight=True, # TODO: 这个选项打开时统计所有环境的norm mean

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
            update_per_collect=80, # TODO
            # update_per_collect=2, # TODO
            replay_ratio=0.25,
            batch_size=batch_size,
            optim_type='AdamW',
            # cos_lr_scheduler=True,
            cos_lr_scheduler=False,
            num_segments=num_segments,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            n_episode=n_episode,
            replay_buffer_size=int(5e5),
            # eval_freq=int(1e4),
            eval_freq=int(1e4),
            # eval_freq=int(2),
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
                     num_segments, total_batch_size):
    configs = []
    # ===== only for debug =====
    # exp_name_prefix = f'data_lz/data_unizero_atari_mt_balance_20250509/atari_{len(env_id_list)}games_balance-total-stage{curriculum_stage_num}_vit-encoder-ps8_trans-nlayer8_brf{buffer_reanalyze_freq}_not-share-head_seed{seed}/'
    # exp_name_prefix = f'data_lz/data_unizero_atari_mt_balance_20250509/atari_{len(env_id_list)}games_balance-total-stage{curriculum_stage_num}_no-encoder-scale_cnn-encoder_moe8_trans-nlayer8_brf{buffer_reanalyze_freq}_not-share-head_seed{seed}/'
    # exp_name_prefix = f'data_lz/data_unizero_atari_mt_balance_20250514/atari_{len(env_id_list)}games_balance-total-stage{curriculum_stage_num}_vit-ln_moe8_trans-nlayer4_brf{buffer_reanalyze_freq}_not-share-head_seed{seed}/'
    # exp_name_prefix = f'data_unizero_atari_mt_balance_20250730/atari_{len(env_id_list)}games_balance-total-stage{curriculum_stage_num}_stage-50k-20k_vit-small-ln_trans-nlayer4-moe8_attn-mlp-lora_no-lora-scale_brf{buffer_reanalyze_freq}_not-share-head_seed{seed}/'
    exp_name_prefix = f'data_unizero_atari_mt_balance_20250730/atari_{len(env_id_list)}games_balance-total-stage{curriculum_stage_num}_stage-50k-20k_vit-small-ln_trans-nlayer4-moe8_encoder-backbone-attn-mlp-lora_no-lora-scale_brf{buffer_reanalyze_freq}_not-share-head_seed{seed}/'

    for task_id, env_id in enumerate(env_id_list):
        config = create_config(
            env_id, action_space_size, collector_env_num, evaluator_env_num, n_episode, num_simulations,
            reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type,
            buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition, num_segments, total_batch_size
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
        cd /mnt/nfs/zhangjinouwen/puyuan/LightZero
        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_balance_config.py 2>&1 | tee ./log/202507/uz_mt_nlayer4_atari8_balance-totalstage5_encoder-backbone.log

        cd /cpfs04/user/puyuan/code/LightZero/
        python -m torch.distributed.launch --nproc_per_node=6 --master_port=29502 /cpfs04/user/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_balance_config.py 2>&1 | tee ./log/20250522_cpfs/uz_mt_nlayer4_atari8_vit-small_moe8-lora_balance-totalstage5_stage-50k-20k_s0.log

        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 /fs-computility/ai-shen/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_balance_config.py 2>&1 | tee ./log/20250509/uz_mt_nlayer4_atari26_vit-ln_moe8_balance-totalstage9.log

        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29503 /fs-computility/ai-shen/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_balance_config.py 2>&1 | tee ./log/uz_mt_balance_atari26_vit-ln_moe8_totalstage5.log
        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29503 /fs-computility/ai-shen/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_balance_config.py 2>&1 | tee ./log/20250509/uz_mt_nlayer8_atari8_vit-ln_moe8_balance-totalstage5.log


        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29503 /fs-computility/ai-shen/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_balance_config.py 2>&1 | tee ./log/uz_mt_balance_atari8_no-encoder-grad-scale_cnn-encoder_moe8_totalstage5_20250509.log

        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29503 /fs-computility/ai-shen/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_balance_config.py 2>&1 | tee ./log/uz_mt_atari26_cnn-encoder_totalstage9_balance20250505.log

        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29503 /fs-computility/ai-shen/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_balance_config.py 2>&1 | tee ./log/uz_mt_atari8_vit-base-encoder-ps8_totalstage3_balance_20250501_debug.log
        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29503 /fs-computility/ai-shen/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_balance_config.py 2>&1 | tee ./log/uz_mt_atari26_vit-large-encoder-ps8-simnorm_totalstage5_balance20250501.log

    """

    from lzero.entry import train_unizero_multitask_segment_ddp
    from ding.utils import DDPContext
    import os


    # env_id_list = [
    #     'PongNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SeaquestNoFrameskip-v4', 'BoxingNoFrameskip-v4',
    #     'AlienNoFrameskip-v4', 'ChopperCommandNoFrameskip-v4', 'HeroNoFrameskip-v4', 'RoadRunnerNoFrameskip-v4',
    # ]
    # # List of Atari games used for multi-task learning
    # env_id_list = [
    #     'PongNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SeaquestNoFrameskip-v4', 'BoxingNoFrameskip-v4',
    #     'AlienNoFrameskip-v4', 'ChopperCommandNoFrameskip-v4', 'HeroNoFrameskip-v4', 'RoadRunnerNoFrameskip-v4',
    #     'AmidarNoFrameskip-v4', 'AssaultNoFrameskip-v4', 'AsterixNoFrameskip-v4', 'BankHeistNoFrameskip-v4',
    #     'BattleZoneNoFrameskip-v4', 'CrazyClimberNoFrameskip-v4', 'DemonAttackNoFrameskip-v4', 'FreewayNoFrameskip-v4',
    #     'FrostbiteNoFrameskip-v4', 'GopherNoFrameskip-v4', 'JamesbondNoFrameskip-v4', 'KangarooNoFrameskip-v4',
    #     'KrullNoFrameskip-v4', 'KungFuMasterNoFrameskip-v4', 'PrivateEyeNoFrameskip-v4', 'UpNDownNoFrameskip-v4',
    #     'QbertNoFrameskip-v4', 'BreakoutNoFrameskip-v4',
    # ]

    def get_atari_target_return_dict(ratio=1.0):
        """
        根据 Human 分数和传入的比例参数 ratio 计算每个 Atari 游戏的 target_return。
        
        参数：
            ratio: 控制 target_return 大小的比例因子，默认为 1.0
        
        返回：
            包含 Atari 游戏 target_return 的字典，key 为环境名称，value 为计算后的目标分数（整数）。
        """
        human_scores = {
            # 8games
            'PongNoFrameskip-v4': 14.6, # 0
            'MsPacmanNoFrameskip-v4': 6951.6, # 1
            'SeaquestNoFrameskip-v4': 42054.7, # 2
            'BoxingNoFrameskip-v4': 12.1, # 3
            'AlienNoFrameskip-v4': 7127.7, # 4
            'ChopperCommandNoFrameskip-v4': 7387.8, # 5
            'HeroNoFrameskip-v4': 30826.4, # 6
            'RoadRunnerNoFrameskip-v4': 7845.0, # 7
            # 后续 Atari 26games 的额外项
            'AmidarNoFrameskip-v4': 1719.5, # 8
            'AssaultNoFrameskip-v4': 742.0, # 9
            'AsterixNoFrameskip-v4': 8503.3, # 10
            'BankHeistNoFrameskip-v4': 753.1, # 11
            'BattleZoneNoFrameskip-v4': 37187.5, # 12
            'CrazyClimberNoFrameskip-v4': 35829.4, # 13
            'DemonAttackNoFrameskip-v4': 1971.0,  # 14
            'FreewayNoFrameskip-v4': 29.6, # 15
            'FrostbiteNoFrameskip-v4': 4334.7, # 16
            'GopherNoFrameskip-v4': 2412.5, # 17
            'JamesbondNoFrameskip-v4': 302.8, # 18
            'KangarooNoFrameskip-v4': 3035.0, # 19
            'KrullNoFrameskip-v4': 2665.5, # 20
            'KungFuMasterNoFrameskip-v4': 22736.3, # 21
            'PrivateEyeNoFrameskip-v4': 69571.3, # 22
            'UpNDownNoFrameskip-v4': 11693.2, # 23
            'QbertNoFrameskip-v4': 13455.0, # 24
            'BreakoutNoFrameskip-v4': 30.5, # 25
        }

        # target score
        target_scores = {
            # 8games
            # 'PongNoFrameskip-v4': 14.6, # 0 expert
            'PongNoFrameskip-v4': 20, # 0 expert
            # 'MsPacmanNoFrameskip-v4': 1500.6, # 1 
            'MsPacmanNoFrameskip-v4': 6951.6, # 1
            # 'SeaquestNoFrameskip-v4': 1000.7, # 2
            'SeaquestNoFrameskip-v4': 42054.7, # 2 expert
            'BoxingNoFrameskip-v4': 12.1, # 3 expert
            # 'AlienNoFrameskip-v4': 1000.7, # 4
            'AlienNoFrameskip-v4': 7127.7, # 4 expert
            # 'ChopperCommandNoFrameskip-v4': 3000.8, # 5
            # 'HeroNoFrameskip-v4': 3082.4, # 6
            'ChopperCommandNoFrameskip-v4': 7387.8, # 5 expert
            'HeroNoFrameskip-v4': 30826.4, # 6 expert
            'RoadRunnerNoFrameskip-v4': 7845.0, # 7 expert
            # 后续 Atari 26games 的额外项
            'AmidarNoFrameskip-v4': 100.5, # 8
            'AssaultNoFrameskip-v4': 742.0, # 9
            'AsterixNoFrameskip-v4': 1503.3, # 10
            'BankHeistNoFrameskip-v4': 753.1, # 11
            'BattleZoneNoFrameskip-v4': 12187.5, # 12
            'CrazyClimberNoFrameskip-v4': 15829.4, # 13
            'DemonAttackNoFrameskip-v4': 1971.0,  # 14
            'FreewayNoFrameskip-v4': 29.6, # 15
            'FrostbiteNoFrameskip-v4': 334.7, # 16
            'GopherNoFrameskip-v4': 2412.5, # 17
            'JamesbondNoFrameskip-v4': 302.8, # 18
            'KangarooNoFrameskip-v4': 3035.0, # 19
            'KrullNoFrameskip-v4': 2665.5, # 20
            'KungFuMasterNoFrameskip-v4': 12736.3, # 21
            'PrivateEyeNoFrameskip-v4': 1001.3, # 22
            'UpNDownNoFrameskip-v4': 11693.2, # 23
            'QbertNoFrameskip-v4': 13455.0, # 24
            'BreakoutNoFrameskip-v4': 30.5, # 25
        }


        # 计算每个游戏的 target_return
        # return {env: int(round(score * ratio)) for env, score in human_scores.items()}
        return {env: int(round(score * ratio)) for env, score in target_scores.items()}


    global target_return_dict 
    # global BENCHMARK_NAME
    # BENCHMARK_NAME='atari'

    # 示例：以 ratio=1 使用
    target_return_dict = get_atari_target_return_dict(ratio=1)
    # target_return_dict = get_atari_target_return_dict(ratio=0.5)
    num_games = 8 # 26 # 8

    # 分别定义 Atari 游戏列表（8games 和 26games）
    if num_games==3:
            env_id_list = [
            'PongNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SeaquestNoFrameskip-v4'
        ]
    elif num_games==8:
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

    global curriculum_stage_num
    # TODO ==============
    # curriculum_stage_num=3
    curriculum_stage_num=5
    # curriculum_stage_num=9

    action_space_size = 18
    collector_env_num = 8
    num_segments = 8
    n_episode = 8
    evaluator_env_num = 3
    num_simulations = 50
    max_env_step = int(4e5)
    reanalyze_ratio = 0.0
    
    if len(env_id_list) == 8:
        effective_batch_size = 512
    elif len(env_id_list) == 26:
        # effective_batch_size = 832  # cnn-encoder
        effective_batch_size = 512   # base-vit-encoder 
        # effective_batch_size = 256   # base-vit-encoder  large-vit-encoder
    elif len(env_id_list) == 18:
        effective_batch_size = 512 * 3  # 1536 
    else:
        raise ValueError("不支持的环境数量: {}".format(n))

    batch_sizes, grad_acc_steps = compute_batch_config(env_id_list, effective_batch_size)
    total_batch_size =  effective_batch_size # 当前无效

    num_unroll_steps = 10
    infer_context_length = 4
    norm_type = 'LN'
    # buffer_reanalyze_freq = 1 / 50
    buffer_reanalyze_freq = 1 / 1000000
    reanalyze_batch_size = 160
    reanalyze_partition = 0.75

    # ======== TODO: only for debug ========
    # collector_env_num = 2
    # num_segments = 2
    # n_episode = 2
    # evaluator_env_num = 2
    # num_simulations = 1
    # reanalyze_batch_size = 2
    # num_unroll_steps = 5
    # infer_context_length = 2
    # batch_sizes = [4 for _ in range(len(env_id_list))]

    from lzero.entry import train_unizero_multitask_balance_segment_ddp

    for seed in [0]:
        configs = generate_configs(env_id_list, action_space_size, collector_env_num, n_episode, evaluator_env_num,
                                   num_simulations, reanalyze_ratio, batch_sizes, num_unroll_steps, infer_context_length,
                                   norm_type, seed, buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition,
                                   num_segments, total_batch_size)

        with DDPContext():
            train_unizero_multitask_balance_segment_ddp(configs, seed=seed, max_env_step=max_env_step, benchmark_name="atari")
            # ======== TODO: only for debug ========
            # train_unizero_multitask_segment_ddp(configs[:2], seed=seed, max_env_step=max_env_step) # train on the first four tasks