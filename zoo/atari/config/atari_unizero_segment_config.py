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

    evaluator_env_num = 3
    num_simulations = 50

    if env_id == 'ALE/Pong-v5':
        max_env_step = int(5e5) # TODO pong
    else:
        max_env_step = int(10e6) # TODO

    batch_size = 128 # for decode-loss # TODO
    replay_ratio = 0.25

    num_layers = 2


    # Defines the frequency of reanalysis. E.g., 1 means reanalyze once per epoch, 2 means reanalyze once every two epochs.
    buffer_reanalyze_freq = 1/5000000000

    # Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
    reanalyze_batch_size = 160
    # The partition of reanalyze. E.g., 1 means reanalyze_batch samples from the whole buffer, 0.5 means samples from the first half of the buffer.
    reanalyze_partition = 0.75
    norm_type ="LN"

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
        ),
        policy=dict(
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=100000000000, ), ), ),  # default is 10000
            model=dict(
                observation_shape=(3, 64, 64),
                action_space_size=action_space_size,
                reward_support_range=(-300., 301., 1.),
                value_support_range=(-300., 301., 1.),
                norm_type=norm_type,
                num_res_blocks=2,
                num_channels=128,
                world_model_cfg=dict(
                    latent_recon_loss_weight=0.1, # TODO
                    perceptual_loss_weight=0.1,
                    use_new_cache_manager=False, # TODO

                    norm_type=norm_type,
                    final_norm_option_in_obs_head='LayerNorm',
                    final_norm_option_in_encoder='LayerNorm',
                    predict_latent_loss_type='mse', # TODO: only for latent state layer_norm

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
                    # LoRA parameters:
                    lora_r= 0,
                    lora_alpha =1,
                    lora_dropout= 0.0,
                    optim_type='AdamW_mix_lr_wdecay', # only for tsne plot

                    # ==================== [FIXED LOCATION] Policy Stability Fixes ====================
                    # These fixes belong in world_model_cfg because they are read by world_model.py

                    # Fix1: Clip policy logits to prevent explosion
                    # RECOMMENDED: Enable this as a safety net against catastrophic logits
                    use_policy_logits_clip=True,
                    policy_logits_clip_min=-20.0,        # STRENGTHENED: Tighter clip (was -5.0)
                    policy_logits_clip_max=20.0,         # STRENGTHENED: Tighter clip (was 5.0)

                    # Fix2: Unified policy label smoothing (applied in unizero.py)
                    # This is the PRIMARY smoothing mechanism
                    use_continuous_label_smoothing=True,  # Enable continuous smoothing
                    continuous_ls_eps=0.05,                # INCREASED: More aggressive smoothing (was 0.05)

                    # Fix3: Re-smooth target_policy from buffer before training
                    # ⚠️ DEPRECATED: This is now handled by Fix2 in unizero.py
                    # Setting to False to avoid redundant smoothing
                    use_target_policy_resmooth=False,
                    target_policy_resmooth_eps=0.05,    # Ignored when use_target_policy_resmooth=False

                    # Fix5: Policy loss temperature scaling
                    # RECOMMENDED: Enable for smoother gradients
                    use_policy_loss_temperature=True,
                    policy_loss_temperature=1.5,        # Temperature for softening policy distribution
                    # =================================================================================

                ),
            ),
            optim_type='AdamW_mix_lr_wdecay',
            weight_decay=1e-2, # TODO: encoder 5*wd, transformer wd, head 0
            learning_rate=0.0001,

            # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
            model_path=None, # TODO

            # (bool) Whether to enable adaptive policy entropy weight (alpha)
            use_adaptive_entropy_weight=True,
            # (float) Learning rate for adaptive alpha optimizer
            adaptive_entropy_alpha_lr=1e-3,
            target_entropy_start_ratio=0.98,
            target_entropy_end_ratio=0.05,
            target_entropy_decay_steps=500000,  # Complete decay after 500k iterations (needs coordination with replay ratio)


            # ==================== START: Encoder-Clip Annealing Config ====================
            # (bool) Whether to enable annealing for encoder-clip values.
            use_encoder_clip_annealing=True,
            # (str) Annealing type. Options: 'linear' or 'cosine'.
            encoder_clip_anneal_type='cosine',
            # (float) Starting clip value for annealing (looser in early training).
            encoder_clip_start_value=30.0,
            # (float) Ending clip value for annealing (stricter in later training).
            encoder_clip_end_value=10.0,
            # (int) Training iteration steps required to complete annealing from start to end value.
            encoder_clip_anneal_steps=100000,  # e.g., reach final value after 100k iterations


            # ==================== Head-Clip (Dynamic, like Encoder-Clip) ====================
            # Dynamic Head Clipping consistent with Encoder-Clip principles
            # Monitor head output (logits) range and scale entire head module weights when exceeding threshold
            use_head_clip=True,  # Enable Head-Clip
            head_clip_config=dict(
                enabled=True,  # TODO
                # Specify heads that need clipping
                enabled_heads=['policy'],  # Can add 'value', 'rewards'

                # Detailed configuration for each head
                head_configs=dict(
                    policy=dict(
                        use_annealing=True,     # Enable threshold annealing
                        anneal_type='cosine',   # 'cosine' or 'linear'
                        start_value=30.0,       # Loose in early phase (allow larger logits range)
                        end_value=10.0,         # Strict in later phase (tighten to reasonable range)
                        anneal_steps=100000,    # Complete annealing in 100k iterations (before performance degradation)
                    ),
                ),

                # Monitoring configuration
                monitor_freq=1,      # Check every iteration
                log_freq=10000,      # Print log every 10000 iterations (TODO)
            ),
            # ========================================================================================


            # ==================== START: Label Smoothing ====================
            policy_ls_eps_start=0.05,  # TODO: Good starting value for Pong and MsPacman
            policy_ls_eps_end=0.01,
            policy_ls_eps_decay_steps=50000,  # 50k
            label_smoothing_eps=0.1,  # TODO: For value
            # ========================================================================================


            # Enhanced monitoring for policy logits and target policy entropy
            use_enhanced_policy_monitoring=True,  # Set to False to disable extra logging
            # ==================== Norm Monitoring Frequency ====================
            # How often (in training iteration steps) to monitor model parameter norms. Set to 0 to disable.
            monitor_norm_freq=5000,  # TODO
            use_augmentation=True,
            manual_temperature_decay=False,
            threshold_training_steps_for_final_temperature=int(2.5e4),
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
            target_update_theta=0.05,
            train_start_after_envsteps=0,  # Only for debug
            game_segment_length=game_segment_length,
            grad_clip_value=5,
            replay_buffer_size=int(5e5),  # TODO

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
    main_config.exp_name = f'data_unizero_st_1226/{env_id[3:-3]}/{env_id[3:-3]}_uz_head-clip-p_target005_allhead4_targetentropy-alpha-500k-098-005-min005_mse-loss2_rec01_poli-clip10_pol-smo-005_pol-loss-tmp-1.5_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'
    # main_config.exp_name = f'data_unizero/{env_id[3:-3]}/{env_id[3:-3]}_uz_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'

    train_unizero_segment([main_config, create_config], seed=seed, model_path=main_config.policy.model_path, max_env_step=max_env_step)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process different environments and seeds.')
    parser.add_argument('--env', type=str, help='The environment to use', default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    args = parser.parse_args()

    # Test environments from atari8 base set
    # args.env = 'ALE/Pong-v5'               # Memory-planning environment with sparse rewards
    # args.env = 'ALE/Qbert-v5'               # Memory-planning environment with sparse rewards
    args.env = 'ALE/MsPacman-v5'               # Memory-planning environment with sparse rewards

    main(args.env, args.seed)

    """
    tmux new -s uz-st-refactor-boxing

    export CUDA_VISIBLE_DEVICES=1
    cd /mnt/shared-storage-user/puyuan/code_20250828/LightZero/
    /mnt/shared-storage-user/puyuan/lz/bin/python /mnt/shared-storage-user/puyuan/code_20250828/LightZero/zoo/atari/config/atari_unizero_segment_config.py 2>&1 | tee /mnt/shared-storage-user/puyuan/code_20250828/LightZero/log/202511/20251105_uz_st_qbert_nokvcachemanager_from-0_250k-reset_cos4e-5_gcv05_encoder-400k-2_fixdownnorm.log

    /mnt/shared-storage-user/puyuan/lz/bin/python /mnt/shared-storage-user/puyuan/code_20250828/LightZero/zoo/atari/config/atari_unizero_segment_config.py 2>&1 | tee /mnt/shared-storage-user/puyuan/code_20250828/LightZero/log/202511/20251105_uz_st_qbert_nokvcachemanager_10k-300k-reset.log
    """
