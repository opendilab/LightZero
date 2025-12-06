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
    evaluator_env_num = 3
    num_simulations = 50
    max_env_step = int(5e5)
    batch_size = 64
    num_layers = 2
    replay_ratio = 0.25
    num_unroll_steps = 10
    infer_context_length = 4
    collect_num_simulations = 50
    eval_num_simulations = 50
    num_channels=64
    num_res_blocks=1

    # Defines the frequency of reanalysis. E.g., 1 means reanalyze once per epoch, 2 means reanalyze once every two epochs.
    buffer_reanalyze_freq = 1/100000
    # Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
    reanalyze_batch_size = 160
    # The partition of reanalyze. E.g., 1 means reanalyze_batch samples from the whole buffer, 0.5 means samples from the first half of the buffer.
    reanalyze_partition = 0.75
    norm_type ="LN"

    # ====== only for debug =====
    # collector_env_num = 2
    # num_segments = 2
    # evaluator_env_num = 2
    # num_simulations = 10
    # batch_size = 5
    # ==============================================================
    # end of the most frequently changed config specified by the user
    # ==============================================================

    atari_unizero_config = dict(
        env=dict(
            stop_value=int(1e6),
            env_id=env_id,
            observation_shape=(3, 96, 96),
            gray_scale=False,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
            # TODO: only for debug
            # collect_max_episode_steps=int(50),
            # eval_max_episode_steps=int(50),
        ),
        reward_model=dict(
            type='rnd_unizero',
            intrinsic_reward_type='add',
            input_type='obs',  # options: ['obs', 'latent_state', 'obs_latent_state']
            activation_type='LeakyReLU',
            enable_image_logging=True,
            
            # —— 新增：自适应权重调度 —— #
            use_intrinsic_weight_schedule=False,     # 打开自适应权重
            intrinsic_weight_mode='cosine',         # 'cosine' | 'linear' | 'constant'
            intrinsic_weight_warmup=10000,           # 前多少次 estimate 权重=0
            intrinsic_weight_ramp=20000,            # 从min升到max所需的 estimate 数
            intrinsic_weight_min=0.0,               
            intrinsic_weight_max=0.025, 
            
            obs_shape=(3, 96, 96),
            latent_state_dim=256,
            hidden_size_list=[128, 256, 256],
            output_dim=512,
            learning_rate=3e-4,
            weight_decay=1e-4,
            input_norm=True,
            input_norm_clamp_max=5,
            input_norm_clamp_min=-5,
            
            intrinsic_norm=True,
            intrinsic_norm_clamp_min=-30,
            intrinsic_norm_clamp_max=30,
            
            extrinsic_sign=False,
            extrinsic_norm=False,
            extrinsic_norm_clamp_min=-5,
            extrinsic_norm_clamp_max=5,
            adjust_value_with_intrinsic=False,
            discount_factor=0.997,
            
        ),
        policy=dict(
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000000, ), ), ),  # default is 10000
            model=dict(
                num_channels=num_channels,
                num_res_blocks=num_res_blocks,
                observation_shape=(3, 96, 96),
                action_space_size=action_space_size,
                reward_support_range=(-300., 301., 1.),
                value_support_range=(-300., 301., 1.),
                norm_type=norm_type,
                world_model_cfg=dict(
                    use_new_cache_manager=False,
                    norm_type=norm_type,
                    final_norm_option_in_obs_head='LayerNorm',
                    final_norm_option_in_encoder='LayerNorm',
                    predict_latent_loss_type='mse',
                    # predict_latent_loss_type='cos_sim', # only for latent state layer_norm
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
                    use_priority=False,
                    rotary_emb=False,
                    optim_type='AdamW_mix_lr_wdecay',
                ),
            ),
            collect_num_simulations=collect_num_simulations,
            eval_num_simulations=eval_num_simulations,
            optim_type='AdamW_mix_lr_wdecay',
            weight_decay=1e-2, # TODO: encoder 5*wd, transformer wd, head 0
            learning_rate=0.0001,
            # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
            model_path=None,
            
            # (bool) 是否启用自适应策略熵权重 (alpha)
            use_adaptive_entropy_weight=True,
            # (float) 自适应alpha优化器的学习率
            adaptive_entropy_alpha_lr=1e-3,
            target_entropy_start_ratio =0.98,
            target_entropy_end_ratio =0.7,
            target_entropy_decay_steps = 50000, # 例如，在300k次迭代后达到最终值
            # ==================== START: Encoder-Clip Annealing Config ====================
            # (bool) 是否启用 encoder-clip 值的退火。
            use_encoder_clip_annealing=False,
            # (str) 退火类型。可选 'linear' 或 'cosine'。
            encoder_clip_anneal_type='cosine',
            # (float) 退火的起始 clip 值 (训练初期，较宽松)。
            encoder_clip_start_value=30.0,
            # (float) 退火的结束 clip 值 (训练后期，较严格)。
            encoder_clip_end_value=10.0,
            # (int) 完成从起始值到结束值的退火所需的训练迭代步数。
            encoder_clip_anneal_steps=100000,  # 例如，在300k次迭代后达到最终值
            
            # ==================== START: label smooth ====================
            policy_ls_eps_start=0.05, #good start in Pong and MsPacman
            policy_ls_eps_end=0.01,
            policy_ls_eps_decay_steps=0.0, # 50k
            label_smoothing_eps=0.0,  #for value

            # ==================== [新增] 范数监控频率 ====================
            # 每隔多少个训练迭代步数，监控一次模型参数的范数。设置为0则禁用。
            monitor_norm_freq=500000,
            
            use_augmentation=False,
            # use_augmentation=True,
            manual_temperature_decay=False,
            threshold_training_steps_for_final_temperature=int(2.5e4),
            use_priority=False,
            priority_prob_alpha=1,
            priority_prob_beta=1,
            num_unroll_steps=num_unroll_steps,
            update_per_collect=None,
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            num_simulations=num_simulations,
            num_segments=num_segments,
            td_steps=5,
            train_start_after_envsteps=0,
            game_segment_length=game_segment_length,
            grad_clip_value=5,
            replay_buffer_size=int(1e6),
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
            # ============= RND specific settings =============
            use_rnd_model=True,
            random_collect_data=True,
            use_momentum_representation_network=True,
            target_model_for_intrinsic_reward_update_type='assign',
            target_update_freq_for_intrinsic_reward=1000,
            target_update_theta_for_intrinsic_reward=0.005,
            bp_update_sync=True,
            multi_gpu=False,
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
    from lzero.entry import train_unizero_segment_with_reward_model
    main_config.exp_name = (f'./data_lz/data_unizero_atari_rnd_orig/{env_id[:-14]}_obs_latent_w_10/rnd_{main_config.reward_model.intrinsic_reward_type}_'
                            f'{main_config.reward_model.input_type}_wmax_{main_config.reward_model.intrinsic_weight_max}_input_norm_{main_config.reward_model.input_norm}_intrinsic_norm_{main_config.reward_model.intrinsic_norm}_use_intrinsic_weight_schedule_{main_config.reward_model.use_intrinsic_weight_schedule}/'
                            f'{main_config.policy.model.world_model_cfg.predict_latent_loss_type}_adaptive_entropy_{main_config.policy.use_adaptive_entropy_weight}_use_priority_{main_config.policy.use_priority}_encoder_clip_{main_config.policy.use_encoder_clip_annealing}_label_smoothing_{main_config.policy.label_smoothing_eps}_use_aug_{main_config.policy.use_augmentation}_ncha_{num_channels}_nres_{num_res_blocks}/') 
    # main_config.exp_name = (
    #     f'./data_lz/data_unizero_atari_rnd/{env_id[:-14]}/'
    #     f'{env_id[:-14]}_rnd_w_{main_config.reward_model.intrinsic_reward_weight}_uz_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_'
    #     f'nlayer{num_layers}_numsegments-{num_segments}_'
    #     f'gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_'
    #     f'bs{batch_size}_seed{seed}'
    # )
    train_unizero_segment_with_reward_model(
        [main_config, create_config],
        seed=seed,
        model_path=main_config.policy.model_path,
        max_env_step=max_env_step,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process different environments and seeds.')
    # parser.add_argument('--env', type=str, help='The environment to use', default='PongNoFrameskip-v4')
    parser.add_argument('--env', type=str, help='The environment to use', default='VentureNoFrameskip-v4')
    parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    args = parser.parse_args()

    main(args.env, args.seed)
