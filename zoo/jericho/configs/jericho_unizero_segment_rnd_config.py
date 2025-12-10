import os
import argparse

from easydict import EasyDict


def main(env_id: str = 'detective.z5', seed: int = 0) -> None:
    # ------------------------------------------------------------------
    # Base configurations
    # ------------------------------------------------------------------
    env_configurations = {
        'detective.z5': (12, 100),
        'omniquest.z5': (25, 100),
        'acorncourt.z5': (45, 50),
        'zork1.z5': (55, 500),
    }

    # Set action_space_size and max_steps based on env_id
    action_space_size, max_steps = env_configurations.get(env_id, (10, 50))  # Default values if env_id not found

    # ==============================================================
    # Frequently changed configurations (user-specified)
    # ==============================================================
    # Model name or path - configurable according to the predefined model paths or names
    encoder_option = 'legacy'        # ['qwen', 'legacy']. Legacy uses the bge encoder

    if encoder_option == 'qwen':
        model_name: str = 'Qwen/Qwen3-0.6B'
    elif encoder_option == 'legacy':
        model_name: str = 'BAAI/bge-base-en-v1.5'
    else:
        raise ValueError(f"Unsupported encoder option: {encoder_option}")  


    collector_env_num = 8
    game_segment_length = 50
    evaluator_env_num = 5
    num_segments = 8
    num_simulations = 50
    max_env_step = int(5e5)
    batch_size = 64
    num_unroll_steps = 10
    infer_context_length = 4
    num_layers = 2
    replay_ratio = 0.1
    embed_dim = 768

    # Reanalysis parameters
    buffer_reanalyze_freq = 1 / 100000
    reanalyze_batch_size = 160
    reanalyze_partition = 0.75
    
    collect_num_simulations = 50
    eval_num_simulations = 50
    norm_type = "LN"
    use_rnd_model = True
    rnd_weights = 1.0
    observation_shape = 512
    rnd_random_collect_episode_num = 50
    update_proportion = 0.25
    intrinsic_weight_max = 0.5

    # =========== Debug configurations ===========
    # collector_env_num = 2
    # num_segments = 2
    # max_steps = 20
    # game_segment_length = 20
    # evaluator_env_num = 2
    # num_simulations = 5
    # max_env_step = int(5e5)
    # batch_size = 10
    # num_unroll_steps = 5
    # infer_context_length = 2
    # num_layers = 1
    # replay_ratio = 0.05
    # embed_dim = 24
    # rnd_random_collect_episode_num = 2

    # ------------------------------------------------------------------
    # Construct Jericho Unizero Segment configuration dictionary
    # ------------------------------------------------------------------
    jericho_unizero_config = dict(
        env=dict(
            stop_value=int(1e6),
            max_steps=max_steps,
            observation_shape=observation_shape,
            max_action_num=action_space_size,
            tokenizer_path=model_name,
            max_seq_len=512,
            game_path=f"./zoo/jericho/envs/z-machine-games-master/jericho-game-suite/{env_id}",
            for_unizero=True,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False),
            use_cache=True,
            cache_size=100000,
        ),
        policy=dict(
            learn=dict(
                learner=dict(
                    hook=dict(
                        save_ckpt_after_iter=1000000,
                    ),
                ),
            ),
            model=dict(
                observation_shape=observation_shape,
                action_space_size=action_space_size,
                encoder_option=encoder_option,
                encoder_url=model_name,
                model_type="mlp",
                reward_support_range=(-300., 301., 1.),
                value_support_range=(-300., 301., 1.),
                norm_type=norm_type,
                world_model_cfg=dict(
                    final_norm_option_in_obs_head='LayerNorm',
                    final_norm_option_in_encoder='LayerNorm',
                    predict_latent_loss_type='mse',
                    policy_entropy_weight=5e-3,
                    support_size=601,
                    continuous_action_space=False,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,  # each timestep represents 2 tokens: observation and action
                    context_length=2 * infer_context_length,
                    device="cuda",
                    action_space_size=action_space_size,
                    num_layers=num_layers,
                    num_heads=24,
                    embed_dim=embed_dim,
                    obs_type="text",
                    env_num=max(collector_env_num, evaluator_env_num),
                    decode_loss_mode='None', # Controls where to compute reconstruction loss: after_backbone, before_backbone, or None.
                    latent_recon_loss_weight=0.1,
                    use_priority=False,
                    rotary_emb=False,
                    optim_type='AdamW',
                    game_segment_length=game_segment_length,
                ),
            ),
            collect_num_simulations=collect_num_simulations,
            eval_num_simulations=eval_num_simulations,
            optim_type='AdamW',
            weight_decay=1e-2, 
            learning_rate=0.0001,
            action_type="varied_action_space",
            
            # (bool) 是否启用自适应策略熵权重 (alpha)
            use_adaptive_entropy_weight=False,
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
            use_priority=False,
            priority_prob_alpha=1,
            priority_prob_beta=1,
            td_steps=5,
            
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            reanalyze_ratio=0,
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            fixed_temperature_value=0.25,
            manual_temperature_decay=False,
            num_simulations=num_simulations,
            num_segments=num_segments,
            train_start_after_envsteps=0,
            game_segment_length=game_segment_length,
            replay_buffer_size=int(1e6),
            eval_freq=int(5e3),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
            
            # RND
            use_rnd_model=use_rnd_model,
            rnd_weights=rnd_weights,
            rnd_random_collect_episode_num=rnd_random_collect_episode_num,
            use_momentum_representation_network=True,
            target_model_for_intrinsic_reward_update_type='assign',
            target_update_freq_for_intrinsic_reward=1000,
            target_update_theta_for_intrinsic_reward=0.005,
            reward_model=dict(
                device='cuda',
                type='rnd_unizero',
                intrinsic_reward_type='add',
                input_type='latent_state',  # options: ['obs', 'latent_state', 'obs_latent_state']
                activation_type='LeakyReLU',
                enable_image_logging=True,
                
                update_proportion=update_proportion, # 每次计算loss只有部分值参与计算，防止rnd网络更新太快
                # —— 新增：自适应权重调度 —— #
                use_intrinsic_weight_schedule=False,     # 打开自适应权重
                intrinsic_weight_mode='cosine',         # 'cosine' | 'linear' | 'constant'
                intrinsic_weight_warmup=10000,           # 前多少次 estimate 权重=0
                intrinsic_weight_ramp=20000,            # 从min升到max所需的 estimate 数
                intrinsic_weight_min=0.0,               
                intrinsic_weight_max=intrinsic_weight_max, 
                
                obs_shape=observation_shape,
                latent_state_dim=768,
                hidden_size_list=[768, 512],
                output_dim=512,
                
                input_norm=True,
                input_norm_clamp_max=5,
                input_norm_clamp_min=-5,
                
                intrinsic_norm=True,
                intrinsic_norm_clamp_max=10,
                
                extrinsic_sign=False,
                extrinsic_norm=False,
                extrinsic_norm_clamp_min=-5,
                extrinsic_norm_clamp_max=5,
                discount_factor=0.997,
                
            ),
            bp_update_sync=True,
            multi_gpu=False,
            
        ),
    )
    jericho_unizero_config = EasyDict(jericho_unizero_config)

    # ------------------------------------------------------------------
    # Create configuration for module import
    # ------------------------------------------------------------------
    jericho_unizero_create_config = dict(
        env=dict(
            type="jericho",
            import_names=["zoo.jericho.envs.jericho_env"],
        ),
        env_manager=dict(type="base"),  # Use base env manager to avoid subprocess bugs.
        policy=dict(
            type="unizero",
            import_names=["lzero.policy.unizero"],
        ),
    )
    jericho_unizero_create_config = EasyDict(jericho_unizero_create_config)

    main_config = jericho_unizero_config
    create_config = jericho_unizero_create_config

    # Construct experiment name using key parameters
    # ============ use muzero_segment_collector instead of muzero_collector =============
    from lzero.entry import train_unizero_segment_with_reward_model
    intrinsic_reward_type = main_config.policy.reward_model.intrinsic_reward_type
    input_type = main_config.policy.reward_model.input_type
    intrinsic_weight_max = main_config.policy.reward_model.intrinsic_weight_max
    input_norm = main_config.policy.reward_model.input_norm
    intrinsic_norm = main_config.policy.reward_model.intrinsic_norm
    use_intrinsic_weight_schedule = main_config.policy.reward_model.use_intrinsic_weight_schedule
    main_config.exp_name = (f'./data_lz_rnd/jericho/{env_id}/rnd_loss_w_{rnd_weights}_{intrinsic_reward_type}_'
                            f'{input_type}_wmax_{intrinsic_weight_max}_input_norm_{input_norm}_intrinsic_norm_{intrinsic_norm}_use_intrinsic_weight_schedule_{use_intrinsic_weight_schedule}')
    train_unizero_segment_with_reward_model(
        [main_config, create_config],
        seed=seed,
        model_path=main_config.policy.model_path,
        max_env_step=max_env_step,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process environment configuration for Unizero Segment training."
    )
    parser.add_argument(
        '--env',
        type=str,
        help="The environment to use, e.g., detective.z5",
        default='detective.z5'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help="The seed to use",
        default=0
    )
    args = parser.parse_args()

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main(args.env, args.seed)