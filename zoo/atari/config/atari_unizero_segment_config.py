from easydict import EasyDict
from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map


def main(env_id, seed):
    action_space_size = atari_env_action_space_map[env_id]

    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    collector_env_num = 8
    num_segments = 8

    game_segment_length = 20
    num_unroll_steps = 10
    infer_context_length = 4

    # game_segment_length = 50 # TODO
    # num_unroll_steps = 16
    # infer_context_length = 8

    # game_segment_length = 400 # TODO

    evaluator_env_num = 3
    num_simulations = 50

    if env_id == 'ALE/Pong-v5':
        max_env_step = int(5e5) # TODO pong
    else:
        # max_env_step = int(4e5)
        max_env_step = int(10e6) # TODO

    # batch_size = 2 # only for debug

    batch_size = 64 # for decode-loss
    replay_ratio = 0.25

    # batch_size = 256
    # replay_ratio = 0.1

    num_layers = 2


    # Defines the frequency of reanalysis. E.g., 1 means reanalyze once per epoch, 2 means reanalyze once every two epochs.
    # buffer_reanalyze_freq = 1/50
    buffer_reanalyze_freq = 1/5000000000

    # Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
    reanalyze_batch_size = 160
    # The partition of reanalyze. E.g., 1 means reanalyze_batch samples from the whole buffer, 0.5 means samples from the first half of the buffer.
    reanalyze_partition = 0.75
    norm_type ="LN"

    # ====== only for debug =====
    # collector_env_num = 2
    # num_segments = 2
    # evaluator_env_num = 2
    # num_simulations = 5
    # batch_size = 5
    # buffer_reanalyze_freq = 1/1000000
    # replay_ratio = 1

    # ==============================================================
    # end of the most frequently changed config specified by the user
    # ==============================================================

    atari_unizero_config = dict(
        env=dict(
            frame_skip=1, # TODO
            stop_value=int(1e6),
            env_id=env_id,
            observation_shape=(3, 64, 64),
            gray_scale=False,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
            collect_max_episode_steps=int(1e4),
            eval_max_episode_steps=int(1e4),
            # collect_max_episode_steps=int(5e3),
            # eval_max_episode_steps=int(5e3),
            # TODO: only for debug
            # collect_max_episode_steps=int(20),
            # eval_max_episode_steps=int(20),
        ),
        policy=dict(
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000000, ), ), ),  # default is 10000
            model=dict(
                observation_shape=(3, 64, 64),
                action_space_size=action_space_size,
                reward_support_range=(-300., 301., 1.),
                # reward_support_range=(-50., 51., 1.),
                value_support_range=(-300., 301., 1.),
                norm_type=norm_type,
                # num_res_blocks=1,
                # num_channels=64,
                num_res_blocks=2,
                num_channels=128,
                world_model_cfg=dict(
                    # latent_recon_loss_weight=1,
                    # perceptual_loss_weight=1,

                    latent_recon_loss_weight=0.5,
                    perceptual_loss_weight=0.5,

                    # latent_recon_loss_weight=0.1,
                    # perceptual_loss_weight=0.1,

                    # latent_recon_loss_weight=0,
                    # perceptual_loss_weight=0,  # TODO

                    # use_new_cache_manager=True, # TODO
                    use_new_cache_manager=False,

                    norm_type=norm_type,
                    final_norm_option_in_obs_head='LayerNorm',
                    final_norm_option_in_encoder='LayerNorm',
                    # predict_latent_loss_type='mse', # TODO: only for latent state layer_norm
                    predict_latent_loss_type='cos_sim', # TODO: only for latent state layer_norm

                    # final_norm_option_in_obs_head='SimNorm',
                    # final_norm_option_in_encoder='SimNorm',
                    # predict_latent_loss_type='group_kl', # TODO: only for latent state sim_norm

                    # analysis_dormant_ratio_weight_rank=True, # TODO

                    analysis_dormant_ratio_weight_rank=False, # TODO
                    dormant_threshold=0.025,
                    task_embed_option=None,   # ==============TODO: none ==============
                    use_task_embed=False, # ==============TODO==============
                    use_shared_projection=False,
                    support_size=601,
                    policy_entropy_weight=5e-3,
                    continuous_action_space=False,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,  # NOTE: each timestep has 2 tokens: obs and action
                    context_length=2 * infer_context_length,
                    device='cuda',
                    action_space_size=action_space_size,
                    num_layers=num_layers,
                    num_heads=8,
                    embed_dim=768,
                    obs_type='image',
                    env_num=max(collector_env_num, evaluator_env_num),
                    num_simulations=num_simulations,
                    game_segment_length=game_segment_length,
                    # use_priority=False,
                    use_priority=True,
                    rotary_emb=False,
                    encoder_type='resnet',
                    use_normal_head=True,
                    use_softmoe_head=False,
                    use_moe_head=False,
                    num_experts_in_moe_head=4,
                    moe_in_transformer=False,
                    multiplication_moe_in_transformer=False,
                    num_experts_of_moe_in_transformer=4,
                    # LoRA 参数：
                    lora_r= 0,
                    lora_alpha =1,
                    lora_dropout= 0.0,
                    optim_type='AdamW_mix_lr_wdecay', # only for tsne plot

                    # ==================== [FIXED LOCATION] Policy Stability Fixes ====================
                    # These fixes belong in world_model_cfg because they are read by world_model.py

                    # Fix1: Clip policy logits to prevent explosion
                    use_policy_logits_clip=True,        # Set to False to disable this fix
                    policy_logits_clip_min=-5.0,        # STRENGTHENED: Min value (was -5.0)
                    policy_logits_clip_max=5.0,         # STRENGTHENED: Max value (was 5.0)

                    # Fix3: Re-smooth target_policy from buffer before training
                    use_target_policy_resmooth=False,    
                    # use_target_policy_resmooth=True,    # ⚠️ MUST BE TRUE! (was False)
                    target_policy_resmooth_eps=0.05,    # INCREASED: Re-smooth epsilon (was 0.05)

                    # [NEW] Fix5: Policy loss temperature scaling
                    use_policy_loss_temperature=True,   # Scale policy logits in loss computation
                    policy_loss_temperature=1.5,        # Temperature for softening target
                    # =================================================================================

                ),
            ),
            optim_type='AdamW_mix_lr_wdecay',
            weight_decay=1e-2, # TODO: encoder 5*wd, transformer wd, head 0
            learning_rate=0.0001,

            # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
            model_path=None, # TODO
            # model_path="/mnt/shared-storage-user/puyuan/code_20250828/LightZero/data_unizero_st_refactor1024/Qbert/Qbert_uz_300k-reset-head_head-wd_recon-perc-w05_cossimloss_nokvcachemanager_ch128-res2_aug_targetentropy-alpha-100k-098-06-clipmin5e-3-lr1e-3-encoder-clip30-10-100k_brf2e-10-rbs160-rp0.75_nlayer2_numsegments-8_gsl50_rr0.25_Htrain16-Hinfer8_bs64_seed0/ckpt/ckpt_best.pth.tar",


            # (bool) 是否启用自适应策略熵权重 (alpha)
            use_adaptive_entropy_weight=True,
            # (float) 自适应alpha优化器的学习率
            # adaptive_entropy_alpha_lr=1e-4,
            adaptive_entropy_alpha_lr=1e-3,
            target_entropy_start_ratio =0.98,
            # target_entropy_end_ratio =0.9,
            # target_entropy_end_ratio =0.7,
            # target_entropy_decay_steps = 100000, # 例如，在100k次迭代后达到最终值 需要与replay ratio协同调整

            # target_entropy_end_ratio =0.6,
            # target_entropy_decay_steps = 100000, # 例如，在100k次迭代后达到最终值 需要与replay ratio协同调整
            
            target_entropy_end_ratio = 0.05,
            target_entropy_decay_steps = 500000, # 例如，在100k次迭代后达到最终值 需要与replay ratio协同调整

            # target_entropy_end_ratio =0.5, # TODO=====
            # target_entropy_decay_steps = 400000, # 例如，在100k次迭代后达到最终值 需要与replay ratio协同调整


            # ==================== START: Encoder-Clip Annealing Config ====================
            # (bool) 是否启用 encoder-clip 值的退火。
            use_encoder_clip_annealing=True,
            # (str) 退火类型。可选 'linear' 或 'cosine'。
            encoder_clip_anneal_type='cosine',
            # (float) 退火的起始 clip 值 (训练初期，较宽松)。
            encoder_clip_start_value=30.0,
            # (float) 退火的结束 clip 值 (训练后期，较严格)。
            encoder_clip_end_value=10.0,
            # (int) 完成从起始值到结束值的退火所需的训练迭代步数。
            # encoder_clip_anneal_steps=400000,  # 例如，在400k次迭代后达到最终值
            encoder_clip_anneal_steps=100000,  # 例如，在100k次迭代后达到最终值

            # encoder_clip_end_value=2.0,
            # # (int) 完成从起始值到结束值的退火所需的训练迭代步数。
            # encoder_clip_anneal_steps=400000,  # 例如，在400k次迭代后达到最终值

            # ==================== START: label smooth ====================
            policy_ls_eps_start=0.05, #TODO============= good start in Pong and MsPacman
            policy_ls_eps_end=0.01,
            policy_ls_eps_decay_steps=50000, # 50k
            label_smoothing_eps=0.1,  #TODO============= for value

            # ==================== [FIXED LOCATION] Policy-Level Fixes ====================
            # These fixes belong at policy level because they are read by unizero.py

            # Fix2: Keep label smoothing high throughout training
            use_continuous_label_smoothing=True,  # Set to False to use original decay
            continuous_ls_eps=0.1,                # INCREASED: Fixed epsilon (was 0.05)

            # Fix4: Enhanced monitoring for policy logits and target policy entropy
            use_enhanced_policy_monitoring=True,  # Set to False to disable extra logging
            # =============================================================================

            # ==================== [新增] 范数监控频率 ====================
            # 每隔多少个训练迭代步数，监控一次模型参数的范数。设置为0则禁用。
            # monitor_norm_freq=10000,
            monitor_norm_freq=5000, # TODO
            # monitor_norm_freq=2,  # only for debug

            # use_augmentation=False,
            use_augmentation=True,

            manual_temperature_decay=False,
            threshold_training_steps_for_final_temperature=int(2.5e4),
            # use_priority=False,
            use_priority=True,
            priority_prob_alpha=1,
            priority_prob_beta=1,
            num_unroll_steps=num_unroll_steps,
            update_per_collect=None,
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            num_simulations=num_simulations,
            num_segments=num_segments,
            td_steps=5,
            train_start_after_envsteps=0, # only for debug
            # train_start_after_envsteps=2000,
            game_segment_length=game_segment_length,
            grad_clip_value=5,

            # grad_clip_value=0.5,
            # cos_lr_scheduler=True,
            # inal_learning_rate=4e-5, # dreamerv3

            replay_buffer_size=int(5e5), # TODO
            eval_freq=int(5e3),
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
    atari_unizero_config = EasyDict(atari_unizero_config)
    main_config = atari_unizero_config

    atari_unizero_create_config = dict(
        env=dict(
            type='atari_lightzero',
            import_names=['zoo.atari.envs.atari_lightzero_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(
            type='unizero',
            import_names=['lzero.policy.unizero'],
        ),
    )
    atari_unizero_create_config = EasyDict(atari_unizero_create_config)
    create_config = atari_unizero_create_config

    # ============ use muzero_segment_collector instead of muzero_collector =============
    from lzero.entry import train_unizero_segment
    main_config.exp_name = f'data_unizero_st_refactor1105/{env_id[3:-3]}/{env_id[3:-3]}_uz_claudefix-true_tprnot_fixdownnorm_head-wd_recon-perc-w05_cossimloss5_nokvcachemanager_targetentropy-alpha-400k-098-005-clipmin5e-3-lr1e-3_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'

    # main_config.exp_name = f'data_unizero_st_refactor1105/{env_id[3:-3]}/{env_id[3:-3]}_uz_claudefix-true_fixdownnorm_250k-reset-head_head-wd_recon-perc-w05_cossimloss2_nokvcachemanager_targetentropy-alpha-400k-098-005-clipmin5e-3-lr1e-3_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'

    # main_config.exp_name = f'data_unizero_st_refactor1105/{env_id[3:-3]}/{env_id[3:-3]}_uz_fixdownnorm_250k-reset-head_head-wd_recon-perc-w05_cossimloss5_nokvcachemanager_targetentropy-alpha-100k-098-06-clipmin5e-3-lr1e-3_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'


    # main_config.exp_name = f'data_unizero_st_refactor1105/{env_id[3:-3]}/{env_id[3:-3]}_uz_fixdownnorm_cos4e-5_gcv05_encoder-400k-2_250k-reset-head_head-wd_recon-perc-w05_cossimloss_nokvcachemanager_targetentropy-alpha-100k-098-06-clipmin5e-3-lr1e-3_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'


    # main_config.exp_name = f'data_unizero_st_refactor1105/{env_id[3:-3]}/{env_id[3:-3]}_uz_from-300k-iter-ckpt_10k-300k-reset-head_head-wd_recon-perc-w05_cossimloss_nokvcachemanager_targetentropy-alpha-100k-098-06-clipmin5e-3-lr1e-3_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'

    # main_config.exp_name = f'data_unizero_st_refactor1105/{env_id[3:-3]}/{env_id[3:-3]}_uz_from-300k-iter-ckpt_300k-reset-head_head-wd_recon-perc-w05_cossimloss_nokvcachemanager_ch128-res2_aug_targetentropy-alpha-100k-098-06-clipmin5e-3-lr1e-3-encoder-clip30-10-100k_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'

    # main_config.exp_name = f'data_unizero_st_refactor1024/{env_id[3:-3]}/{env_id[3:-3]}_uz_head-wd_recon-perc-w1_cossimloss_nokvcachemanager_ch128-res2_aug_targetentropy-alpha-100k-098-06-clipmin1e-4-lr1e-3-encoder-clip30-10-100k_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'

    # main_config.exp_name = f'data_unizero_st_refactor1010/{env_id[3:-3]}/{env_id[3:-3]}_uz_ch128-res2_targetentropy-alpha-100k-098-07-encoder-clip30-10-400k_label-smooth_resnet-encoder_priority_adamw-wd1e-2-encoder1-trans1-head1_ln-inner-ln_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'
    train_unizero_segment([main_config, create_config], seed=seed, model_path=main_config.policy.model_path, max_env_step=max_env_step)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process different environments and seeds.')
    parser.add_argument('--env', type=str, help='The environment to use', default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    args = parser.parse_args()



    # 测试的atari8中的4个base环境
    # args.env = 'PongNoFrameskip-v4' # 反应型环境 密集奖励
    # args.env = 'MsPacmanNoFrameskip-v4' # 记忆规划型环境 稀疏奖励

    # args.env = 'ALE/Pong-v5' # 记忆规划型环境 稀疏奖励

    # args.env = 'ALE/MsPacman-v5' # 记忆规划型环境 稀疏奖励


    # args.env = 'SeaquestNoFrameskip-v4'  # 记忆规划型环境 稀疏奖励
    # args.env = 'ALE/Seaquest-v5' # 记忆规划型环境 稀疏奖励

    # args.env = 'HeroNoFrameskip-v4' # 记忆规划型环境 稀疏奖励

    # args.env = 'AlienNoFrameskip-v4'

    # 下面是atari8以外的2个代表环境
    # args.env = 'QbertNoFrameskip-v4' # 记忆规划型环境 稀疏奖励
    args.env = 'ALE/Qbert-v5' # 记忆规划型环境 稀疏奖励

    # args.env = 'SpaceInvadersNoFrameskip-v4' # 记忆规划型环境 稀疏奖励

    # 下面是已经表现不错的
    # args.env = 'BoxingNoFrameskip-v4' # 反应型环境 密集奖励
    # args.env = 'ChopperCommandNoFrameskip-v4'
    # args.env = 'RoadRunnerNoFrameskip-v4'

    main(args.env, args.seed)

    """
    tmux new -s uz-st-refactor-boxing

    export CUDA_VISIBLE_DEVICES=1
    cd /mnt/shared-storage-user/puyuan/code_20250828/LightZero/
    /mnt/shared-storage-user/puyuan/lz/bin/python /mnt/shared-storage-user/puyuan/code_20250828/LightZero/zoo/atari/config/atari_unizero_segment_config.py 2>&1 | tee /mnt/shared-storage-user/puyuan/code_20250828/LightZero/log/202511/20251105_uz_st_qbert_nokvcachemanager_from-0_250k-reset_cos4e-5_gcv05_encoder-400k-2_fixdownnorm.log

    /mnt/shared-storage-user/puyuan/lz/bin/python /mnt/shared-storage-user/puyuan/code_20250828/LightZero/zoo/atari/config/atari_unizero_segment_config.py 2>&1 | tee /mnt/shared-storage-user/puyuan/code_20250828/LightZero/log/202511/20251105_uz_st_qbert_nokvcachemanager_10k-300k-reset.log

    # conda activate /mnt/nfs/zhangjinouwen/puyuan/conda_envs/lz
    # cd /mnt/nfs/zhangjinouwen/puyuan/LightZero
    # python /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/config/atari_unizero_segment_config.py 2>&1 | tee /mnt/nfs/zhangjinouwen/puyuan/LightZero/log/202510/20251023_uz_st_pong_kvcachemanager-yes.log
    """
