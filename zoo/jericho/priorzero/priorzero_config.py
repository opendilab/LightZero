import os
from typing import Dict, Tuple
from easydict import EasyDict
import torch.distributed as dist

def get_priorzero_config(
    env_id: str = 'zork1.z5',
    seed: int = 0,
    exp_name: str = None,
    use_cot: bool = False,
) -> Tuple[EasyDict, EasyDict]:
    """
    Generate complete PriorZero configuration.

    Args:
        env_id: Jericho game ID
        seed: Random seed
        exp_name: Experiment name (auto-generated if None)
        enable_llm: Whether to enable LLM policy (if False, degrades to pure UniZero)
        enable_rft: Whether to enable RFT training (if False, only use SFT)

    Returns:
        main_config: Main configuration dictionary
        create_config: Creation configuration for DI-engine components
    """
    env_configurations = {
        'detective.z5': (12, 100),
        'omniquest.z5': (25, 100),
        'acorncourt.z5': (45, 50),
        'zork1.z5': (55, 500),
    }
    action_space_size, max_steps = env_configurations.get(env_id, (20, 100))
    wm_encoder_option = 'legacy' 
    wm_model_name = 'BAAI/bge-base-en-v1.5'  
    multi_gpu = False
    GPUs = 1
    
    collector_env_num = 4
    evaluator_env_num = 2
    n_episode = collector_env_num
    
    num_unroll_steps = 10
    infer_context_length = 4
    game_segment_length = 50
    num_layers = 2
    embed_dim = 768
    replay_ratio = 0.1
    batch_size = 64
    collect_num_simulations=25
    eval_num_simulations=25
    
    if multi_gpu:
        n_episode = int(GPUs * collector_env_num)
        batch_size = int(batch_size * GPUs)
        
    ## LLM 参数
    # llm_model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # Smaller model for faster iteration
    llm_model_name = "/mnt/afs/wanzunian/niuyazhe/xiongjyu/models/Qwen2.5-0.5B-Instruct"
    train_batch_size = 128   # Total batch size across all GPUs
    GPUS = 1
    micro_batch_size = 8    # Micro batch size per GPU
    gradient_accumulation_steps = train_batch_size // micro_batch_size // GPUS
    rft_loss_type = 'reinforce++'  # 'reinforce' | 'reinforce++' | 'ppo-simple-adv'
    history_length = 5
    llm_learn_num_samples = 256
    replay_buffer_size = int(1e5)
    
    env_config = dict(
        stop_value=int(1e6),
        max_steps=max_steps,
        observation_shape=512,  
        env_id=env_id,
        game_path=f"/mnt/afs/wanzunian/niuyazhe/xiongjyu/jericho/LightZero/zoo/jericho/envs/z-machine-games-master/jericho-game-suite/{env_id}",
        for_unizero=True,
        tokenizer_path=wm_model_name,
        max_action_num=action_space_size,
        max_seq_len=512,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(
            shared_memory=False,
        ),
        use_cache=True,
        cache_size=100000,
    )
    policy_config = dict(
        type='priorzero',
        multi_gpu=multi_gpu,  
        use_wandb=False,
        profile_cfg=dict(
            enable_cprofile=False,  # Enable cProfile for collect/train hot paths
            log_interval=100,        # Aggregate wall-time stats every N profiled sections
        ),
        learn=dict(
                learner=dict(
                    hook=dict(
                        save_ckpt_after_iter=1000000, 
                    ),
                ),
        ),
        model=dict(
            observation_shape=512,
            action_space_size=action_space_size,
            encoder_option=wm_encoder_option,
            encoder_url=wm_model_name,
            model_type="mlp",
            continuous_action_space=False,
            norm_type="LN",
            world_model_cfg=dict(
                norm_type="LN",
                final_norm_option_in_head="LayerNorm",
                final_norm_option_in_encoder="LayerNorm",
                predict_latent_loss_type='mse', 
                policy_entropy_weight=5e-2, 
                continuous_action_space=False,
                max_blocks=num_unroll_steps,  
                max_tokens=2 * num_unroll_steps,  
                context_length=2 * infer_context_length,  
                device="cuda",
                action_space_size=action_space_size,
                num_layers=num_layers,
                num_heads=24,
                embed_dim=embed_dim,
                obs_type="text",
                env_num=max(collector_env_num, evaluator_env_num),
                decode_loss_mode=None, 
                latent_recon_loss_weight=0,
                
                task_embed_option=None,
                moe_in_transformer=False,
                multiplication_moe_in_transformer=False,
                game_segment_length=game_segment_length,
            )
        ),
        update_per_collect=None,
        num_segments=collector_env_num,
        action_type="varied_action_space",
        model_path=None,
        num_unroll_steps=num_unroll_steps,
        reanalyze_ratio=0,
        replay_ratio=replay_ratio,
        batch_size=batch_size,
        learning_rate=3e-4,  
        weight_decay=1e-4,
        cos_lr_scheduler=False,
        fixed_temperature_value=0.25,
        manual_temperature_decay=False,
        n_episode=n_episode,
        train_start_after_envsteps=0,
        replay_buffer_size=replay_buffer_size,
        eval_freq=int(3e4),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        buffer_reanalyze_freq=1 / 1000000,
        reanalyze_batch_size=160,
        reanalyze_partition=0.75,
        device='cuda',
        
        collect_num_simulations=collect_num_simulations,
        eval_num_simulations=eval_num_simulations,
        game_segment_length=game_segment_length,
        off_policy_degree=0,
        enable_async_eval=False,
        
        optim_type='AdamW',
        grad_clip_value=10.0,
        value_loss_weight=0.25,
        policy_loss_weight=1.0,
        reward_loss_weight=1.0,

        use_adaptive_entropy_weight=False,
        adaptive_entropy_alpha_lr=1e-4,
        use_encoder_clip_annealing=False,
        encoder_clip_anneal_type='cosine',
        encoder_clip_start_value=30.0,
        encoder_clip_end_value=10.0,
        encoder_clip_anneal_steps=100000,
        use_priority=False,  # Prioritized experience replay
        priority_prob_alpha=0.6,
        priority_prob_beta=0.4,
        llm_policy_cfg=dict(
            # 是否使用大模型的相关参数
            enable_llm=True,
            enable_sft=False,
            enable_rft=True,
            sft_loss_weight=1,   # Weight of SFT loss in total loss
            rft_loss_weight=1, 
            prompt_log_interval=1000, # 隔多久step输出模型的回答和valid action进行对比
            
            # 模型相关参数
            pretrain_llm_path=llm_model_name,
            history_length=history_length,
            use_cot=use_cot,
            prompt_max_len=8192,
            generate_max_len=128,
            temperature = 1.0,
            top_p = 1.0,
            
            # 训练相关参数
            llm_learn_num_samples=llm_learn_num_samples,
            zero_stage=0,
            train_batch_size=train_batch_size,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=1e-5,
            weight_decay=0.01,
            
            # loss相关参数 
            rft_loss_type=rft_loss_type,
            rft_clip_epsilon=0.2,
            rft_kl_coef=0.01,
        
            # vllm 相关参数
            vllm_tensor_parallel_size=1,
            gpu_memory_utilization=0.2,
        ),
    )
    priorzero_config = dict(
        env=env_config,
        policy=policy_config,
        exp_name=exp_name,
        seed=seed
    )

    create_config = dict(
        env=dict(
            type="jericho",
            import_names=["zoo.jericho.envs.jericho_env"],
        ),
        env_manager=dict(
            type="base" 
        ),
        policy=dict(
            type="priorzero",
            import_names=["zoo.jericho.priorzero.priorzero_policy"],
        ),
        collector=dict(
            type="priorzero_segment",
            import_names=["zoo.jericho.priorzero.priorzero_collector"],
        ),
        evaluator=dict(
            type="priorzero",
            import_names=["zoo.jericho.priorzero.priorzero_evaluator"],
        ),
        replay_buffer=dict(
            type='game_buffer_muzero',
            import_names=['lzero.mcts.buffer.game_buffer_muzero'],
        ),
    )

    main_config = EasyDict(priorzero_config)
    create_config = EasyDict(create_config)
    return main_config, create_config


def get_priorzero_debug_config(
    env_id: str = 'zork1.z5',
    seed: int = 0,
    exp_name: str = None,
    use_cot: bool = False,
) -> EasyDict:
    
    main_config, create_config = get_priorzero_config(env_id=env_id, seed=seed, exp_name=exp_name)
    collector_env_num = 4
    evaluator_env_num = 1
    max_steps=10
    
    num_unroll_steps = 5
    infer_context_length = 2
    batch_size = 16
    collect_num_simulations=2
    eval_num_simulations=2
    num_layers=1
    game_segment_length = 20
    llm_learn_num_samples = 64
    
    create_config.collector_env_num = collector_env_num
    create_config.evaluator_env_num = evaluator_env_num
    create_config.max_steps = max_steps
    
    main_config.policy.model.world_model_cfg.max_blocks = num_unroll_steps
    main_config.policy.model.world_model_cfg.max_tokens = 2 * num_unroll_steps
    main_config.policy.model.world_model_cfg.context_length = 2 * infer_context_length
    main_config.policy.model.world_model_cfg.num_layers = num_layers
    main_config.policy.model.world_model_cfg.game_segment_length = game_segment_length
    main_config.policy.num_unroll_steps = num_unroll_steps
    main_config.policy.batch_size = batch_size
    main_config.policy.collect_num_simulations = collect_num_simulations
    main_config.policy.eval_num_simulations = eval_num_simulations
    main_config.policy.model.world_model_cfg.env_num = collector_env_num
    main_config.policy.num_segments = collector_env_num
    main_config.policy.collector_env_num = collector_env_num
    main_config.policy.update_per_collect = 2
    main_config.policy.game_segment_length = game_segment_length
    main_config.policy.llm_policy_cfg.llm_learn_num_samples = llm_learn_num_samples
    main_config.policy.llm_policy_cfg.use_cot = use_cot 
    
    return main_config, create_config
