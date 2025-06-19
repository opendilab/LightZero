from easydict import EasyDict

import math

# -------------------------------------------------
# 1. 重新实现 compute_batch_config
# -------------------------------------------------
def compute_batch_config(
        env_id_list,
        effective_batch_size: int,
        gpu_num: int = 8,
        max_micro_batch_one_gpu: int = 400,
):
    """
    Args:
        env_id_list (list[str]): 所有任务的环境 id
        effective_batch_size (int): 希望一次反向传播等价的全局 batch
        gpu_num (int): 实际使用的 GPU 数量
        max_micro_batch_one_gpu (int): 单卡能接受的最大 micro-batch
    Returns:
        batch_sizes (list[int]): 每个 env 的 micro-batch
        grad_acc_steps (int): 梯度累积步数
    """
    n_env = len(env_id_list)
    # 每张卡要同时跑多少个 env
    envs_per_gpu = max(1, math.ceil(n_env / gpu_num))
    # 针对“多 env 共用一张卡”的情况缩小 micro-batch 上限
    max_micro_batch = max(1, max_micro_batch_one_gpu // envs_per_gpu)

    # 先按均分做一个“候选 micro-batch”
    candidate = max(1, effective_batch_size // n_env)
    micro_batch = min(candidate, max_micro_batch)

    # 梯度累积步数 = ceil(全局 batch / (micro * n_env))
    grad_acc_steps = max(1, math.ceil(effective_batch_size / (micro_batch * n_env)))

    # 再向下微调 micro-batch，让
    #     micro_batch * n_env * grad_acc_steps <= effective_batch_size
    # 尽量贴合而不超额
    while micro_batch * n_env * grad_acc_steps > effective_batch_size:
        micro_batch -= 1
        if micro_batch == 0:      # 理论上不会发生，防御一下
            micro_batch = 1
            break

    batch_sizes = [micro_batch] * n_env

    # —— 调试信息 —— #
    real_total = micro_batch * n_env * grad_acc_steps
    print(
        f"[BatchConfig] envs={n_env}, target_total={effective_batch_size}, "
        f"micro={micro_batch}, grad_acc={grad_acc_steps}, real_total={real_total}"
    )

    return batch_sizes, grad_acc_steps

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
                    analysis_dormant_ratio_interval=100, # TODO
                    # analysis_dormant_ratio_interval=5000,
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
    # exp_name_prefix = f'data_unizero_atari_mt_20250605/atari_{len(env_id_list)}games_orig_moco_tran-nlayer{num_layers}_brf{buffer_reanalyze_freq}_not-share-head_seed{seed}/'

    exp_name_prefix = f'data_unizero_atari_mt_20250612/atari_{len(env_id_list)}games_vit-small_moe8_tbs512_tran-nlayer{num_layers}_brf{buffer_reanalyze_freq}_not-share-head_seed{seed}/'
    # exp_name_prefix = f'data_unizero_atari_mt_20250612/atari_{len(env_id_list)}games_orig_moco_tran-nlayer{num_layers}_brf{buffer_reanalyze_freq}_not-share-head_seed{seed}/'

    # exp_name_prefix = f'data_unizero_atari_mt_20250611/atari_{len(env_id_list)}games_orig_vit_moe8_tbs256_tran-nlayer{num_layers}_brf{buffer_reanalyze_freq}_not-share-head_seed{seed}/'

    # exp_name_prefix = f'data_unizero_atari_mt_20250605/atari_{len(env_id_list)}games_orig_taskembed128_tran-nlayer{num_layers}_brf{buffer_reanalyze_freq}_not-share-head_seed{seed}/'
    # exp_name_prefix = f'data_unizero_atari_mt_20250605/atari_{len(env_id_list)}games_orig_simnorm-kl_tran-nlayer{num_layers}_brf{buffer_reanalyze_freq}_not-share-head_seed{seed}/'

    # exp_name_prefix = f'data_unizero_atari_mt_20250605/atari_{len(env_id_list)}games_orig_ln-mse_tran-nlayer{num_layers}_brf{buffer_reanalyze_freq}_not-share-head_seed{seed}/'

    # exp_name_prefix = f'data_unizero_atari_mt_20250605/atari_{len(env_id_list)}games_orig_ln-mse_moco-memeff_tran-nlayer{num_layers}_brf{buffer_reanalyze_freq}_not-share-head_seed{seed}/'

    # exp_name_prefix = f'data_unizero_atari_mt_20250601/atari_{len(env_id_list)}games_orig_vit_ln-mse_moco-memeff_tran-nlayer{num_layers}_brf{buffer_reanalyze_freq}_not-share-head_seed{seed}/'

    # exp_name_prefix = f'data_unizero_atari_mt_20250601/atari_{len(env_id_list)}games_orig_vit_ln-mse_moe8_moco-memeff_tran-nlayer{num_layers}_brf{buffer_reanalyze_freq}_not-share-head_seed{seed}/'

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
        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 /fs-computility/niuyazhe/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py 2>&1 | tee /fs-computility/niuyazhe/puyuan/code/LightZero/log/20250509/uz_mt_atari8_orig_ln-mse_moe8_moco_nlayer8_brf002_seed12.log

        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 /fs-computility/niuyazhe/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py 2>&1 | tee /fs-computility/niuyazhe/puyuan/code/LightZero/log/20250509/uz_mt_atari26_orig_vit_ln-mse_moe8_nlayer8_brf002_seed12.log


        =========== cpfs atari8 =========================
        cd /cpfs04/user/puyuan/code/LightZero/
        python -m torch.distributed.launch --nproc_per_node=4 --master_port=29501 /cpfs04/user/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py 2>&1 | tee /cpfs04/user/puyuan/code/LightZero/log/20250522_cpfs/uz_mt_atari8_orig_moco-v1_lop_nlayer8_brf0_seed2.log

        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 /cpfs04/user/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py 2>&1 | tee /cpfs04/user/puyuan/code/LightZero/log/20250522_cpfs/uz_mt_atari8_orig_vit_moe8_lop_nlayer8_brf0_seed1.log

        python -m torch.distributed.launch --nproc_per_node=6 --master_port=29502 /cpfs04/user/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py 2>&1 | tee /cpfs04/user/puyuan/code/LightZero/log/20250522_cpfs/uz_mt_atari8_orig_taskembed128_lop_nlayer8_brf0_seed1.log

        python -m torch.distributed.launch --nproc_per_node=6 --master_port=29502 /cpfs04/user/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py 2>&1 | tee /cpfs04/user/puyuan/code/LightZero/log/20250522_cpfs/uz_mt_atari8_orig_simnorm-kl_lop_nlayer8_brf0_seed1.log


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


    num_games = 8 # 26 # 8
    num_layers = 4 # ==============TODO==============
    action_space_size = 18
    collector_env_num = 8
    num_segments = 8
    n_episode = 8
    evaluator_env_num = 3
    num_simulations = 50
    max_env_step = int(4e5)
    reanalyze_ratio = 0.0

    
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

    if len(env_id_list) == 8:
        if num_layers == 4:
            # effective_batch_size =  1024 # nlayer4 需要设置replay_ratio=0.25对应的upc=40
            effective_batch_size =  512 # nlayer4 需要设置replay_ratio=0.25对应的upc=40 moco
        elif num_layers == 8:
            effective_batch_size = 512 # nlayer8 需要设置replay_ratio=0.5对应的upc=80
            # effective_batch_size = 256 # moco nlayer8 需要设置replay_ratio=0.5对应的upc=80

    elif len(env_id_list) == 26:
        # effective_batch_size = 832  # cnn-encoder
        # effective_batch_size = 1024  # base-vit-encoder transformer-nlayer4  or cnn-encoder
        effective_batch_size = 512  # base-vit-encoder transformer-nlayer4 transformer-nlayer8 需要设置replay_ratio=0.5对应的upc
        # effective_batch_size = 256   # large-vit-encoder
    elif len(env_id_list) == 18:
        effective_batch_size = 512 * 3  # 1536 
    elif len(env_id_list) == 3:
        effective_batch_size = 10  # debug
    else:
        raise ValueError("不支持的环境数量: {}".format(n))

    batch_sizes, grad_acc_steps = compute_batch_config(env_id_list, effective_batch_size)
    total_batch_size =  effective_batch_size # 当前无效

    num_unroll_steps = 10
    infer_context_length = 4
    # infer_context_length = 5 # ==============TODO==============

    norm_type = 'LN'
    # buffer_reanalyze_freq = 1 / 50
    buffer_reanalyze_freq = 1 / 1000000
    reanalyze_batch_size = 160
    reanalyze_partition = 0.75

    # ======== TODO: only for debug ========
    # num_games=3
    # env_id_list = [
    #         'PongNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SeaquestNoFrameskip-v4'
    #     ]
    # num_layers = 1 # ==============TODO==============
    # collector_env_num = 2
    # num_segments = 2
    # n_episode = 2
    # evaluator_env_num = 2
    # num_simulations = 5
    # reanalyze_batch_size = 2
    # num_unroll_steps = 5
    # infer_context_length = 2
    # batch_sizes = [20 for _ in range(len(env_id_list))]
    # total_batch_size =  20*len(env_id_list)
    # max_env_step = 300

    import torch.distributed as dist
    # for seed in [1]:
    for seed in [0]:
        configs = generate_configs(env_id_list, action_space_size, collector_env_num, n_episode, evaluator_env_num,
                                   num_simulations, reanalyze_ratio, batch_sizes, num_unroll_steps, infer_context_length,
                                   norm_type, seed, buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition,
                                   num_segments, total_batch_size, num_layers)

        with DDPContext():
            train_unizero_multitask_segment_ddp(configs, seed=seed, max_env_step=max_env_step, benchmark_name= "atari" )
            # ======== TODO: only for debug ========
            # train_unizero_multitask_segment_ddp(configs[:2], seed=seed, max_env_step=max_env_step) # train on the first four tasks
            print(f"seed: {seed} done!")
            dist.destroy_process_group()
    
