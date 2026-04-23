import os
from typing import Dict, Tuple, Optional, Any
from easydict import EasyDict
import torch.distributed as dist
from dataclasses import dataclass, field

# ============================================================================
# Model Configuration Presets
# ============================================================================
MODEL_CONFIGS = {
    "qwen2.5-0.5b": {
        "model_name_or_path": "/mnt/afs/wanzunian/niuyazhe/xiongjyu/models/Qwen2.5-0.5B-Instruct",
        "vllm_tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.2,
        "description": "Qwen2.5-0.5B-Instruct (smallest, fastest)",
    },
    "qwen2.5-1.5b": {
        "model_name_or_path": "/mnt/shared-storage-user/puyuan/xiongjyu/models/Qwen2.5-1.5B-Instruct",
        "vllm_tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.2,
        "description": "Qwen2.5-1.5B-Instruct (balanced performance)",
    },
    "qwen2.5-3b": {
        # "model_name_or_path": "/mnt/afs/niuyazhe/workspace/xiongjyu/models/Qwen2.5-3B-Instruct",
        "model_name_or_path": "/mnt/shared-storage-user/puyuan/xiongjyu/models/Qwen2.5-3B-Instruct",
        "vllm_tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.25,
        "description": "Qwen2.5-3B-Instruct (better quality)",
    },
    "qwen2.5-7b": {
        "model_name_or_path": "/mnt/shared-storage-user/puyuan/model/Qwen2.5-7B-Instruct",
        "vllm_tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.35,
        "description": "Qwen2.5-7B-Instruct (high quality, needs 2+ GPUs)",
    },
    "qwen2.5-14b": {
        "model_name_or_path": "/mnt/shared-storage-user/puyuan/model/Qwen2.5-14B-Instruct",
        "vllm_tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.5,
        "description": "Qwen2.5-14B-Instruct (best quality, needs 4+ GPUs)",
    },
}

def get_available_models():
    """Get list of available model configurations"""
    return list(MODEL_CONFIGS.keys())

def get_model_config(model_key: str) -> Dict:
    """Get model configuration by key"""
    if model_key not in MODEL_CONFIGS:
        available = ", ".join(get_available_models())
        raise ValueError(
            f"Unknown model key: {model_key}\n"
            f"Available models: {available}"
        )
    return MODEL_CONFIGS[model_key]

def print_available_models():
    """Print all available model configurations"""
    print("\n" + "="*80)
    print("Available Model Configurations:")
    print("="*80)
    for key, config in MODEL_CONFIGS.items():
        print(f"\n  {key}:")
        print(f"    Path: {config['model_name_or_path']}")
        print(f"    Tensor Parallel Size: {config['vllm_tensor_parallel_size']}")
        print(f"    GPU Memory Utilization: {config['gpu_memory_utilization']}")
        print(f"    Description: {config['description']}")
    print("="*80 + "\n")

@dataclass
class PriorZeroLLMConfig:
    model_name_or_path: str = "Qwen2.5-3B-Instruct"
    enable_rft: bool = True
    enable_world_model: bool = True
    train_mode_dict: Optional[EasyDict] = field(default_factory=lambda: EasyDict({
        "mode": "full",  # "full" or "lora"
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_bias": "none",  # "none" / "all" / "lora_only"
        "lora_target_modules": (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    }))
    
    train_schedule: Optional[EasyDict] = field(default_factory=lambda: EasyDict({
        "alternate": True, # False 两者都训练（默认配置）；True: 严格交替训练：phase=wm 时仅训练 wm；phase=llm 时仅训练 llm
        "wm_update_iters": 2e3, # alternate=True. wm 的 train_iter 
        "llm_update_iters": 2e2, # alternate=True. llm 的 train_iter
        "start_phase": "wm",   # alternate=True. 从哪个阶段开始： "wm" 或 "llm"
        "llm_collect_mode": "no_collect" # wm_collect意味着llm训练过程收集数据使用 wm; wm_llm_collect意味着 llm 训练过程收集数据使用 llm 和 wm; no_collect 意味着 llm 训练过程不收集数据，直接使用 replay buffer 中的数据
    }))

    llm_prior_temperature: float = 2.0  # LLM prior 分布的温度参数
    mcts_root_logits_dict: Optional[EasyDict] = field(default_factory=lambda: EasyDict({
        "mode": "llm_plus_wm_logits",        # collect/eval阶段保持一致。"llm_logits"是仅用llm prior的logits; "wm_logits"是仅用 world_model 的policy给出的logits; "llm_plus_wm_logits"是两者的加权求和。
        "plus_method": "fixed",        # 当 plus_method = "fixed" 时，使用固定权重；否则使用自适应权重"adaptive"
        "wm_weight": 0.5,            # 当 plus_method = "fixed" 时，WM logits 的权重；LLMPrior 的权重 = 1 - WM_weight
        "llm_max_weight": 0.7,        # 当 plus_method = "adaptive" 时，LLM 的最大权重；WM 的最小权重 = 1 - llm_max_weight
        "llm_min_weight": 0.3,
        "max_envsteps": 1e5,           # 当 plus_method = "adaptive" 时，随着环境交互步数增加，逐渐降低 llm prior 的权重
    }))
    eval_dict: Optional[EasyDict] = field(default_factory=lambda: EasyDict({
        "world_model": True,              # 评估模式1：完全与 unizero 的 eval 一致；mcts 的根节点仅使用 WM 的logits
        "world_model_llm_prior": True,    # 评估模式2：基于 unizero 的 eval 过程, 但是 mcts 的根节点需要利用 llm 的先验；具体怎么利用取决于mcts_root_logits_dict.mode 参数
        "llm_prior": True,                # 评估模式3：仅使用 llm prior 进行 eval, 不需要 wm 进行评估
        "wm_eval_freq": 499,
        "llm_eval_freq": 49,
    }))
    
    attn_implementation: str = "flash_attention_2" 
    history_length: int = 25
    use_cot: bool = False
    cot_weight: float = 0.1 # 控制 cot前缀token的权重，由于重点是action:，所以前缀的token权重调低
    
    user_prompt_dict: Optional[EasyDict] = field(default_factory=lambda: EasyDict({
        "history_with_reward": True,   # 是否在 prompt 中加入历史交互的 reward 信息
        "observation_with_valid_actions": False,  # 是否在 prompt 中加入当前 observation 中可执行的 action 信息  
    }))
    
    prompt_max_len: int = 8192
    generate_max_len: int = 512
    bf16: bool = True

    # vLLM engines 
    enable_vllm: bool = True
    enable_prefix_caching: bool = False
    use_cuda_ipc: bool = False
    enable_vllm_is_correction: bool = False
    vllm_is_truncated_threshold:  Tuple[float, float] = (0.5, 5.0)
    use_mispo: bool = False
    mispo_token_truncated_threshold: Tuple[float, float] = (0.5, 2.0)
    mispo_traj_truncated_threshold: Tuple[float, float] = (0.8, 1.2)
    
    vllm_sync_backend: str = "nccl" # vLLM 同步参数使用的后端
    vllm_tensor_parallel_size: int = 1 # 每个vllm engine使用几张GPU张量并行 (Fixed: 1.5B model should use 1 GPU)

    gpu_memory_utilization: float = 0.3
    vllm_enable_sleep: bool = True # 是否可以休眠
    temperature: float = 1.0
    top_p: float = 0.95
    seed: int = 0
    reduction: str = "mean"
    
    # 训练相关参数
    deepspeed_enable_sleep: bool = True
    
    zero_stage: int = 2
    gradient_checkpointing: bool = False
    gradient_checkpointing_use_reentrant: bool = False
    max_norm: float = 1.0     # Gradient clipping
    ds_tensor_parallel_size: int = 1
    
    # 需要注意的是，buffer中取一条经验是 10个样本，因为包含10次交互； num_unroll_steps = 10
    train_batch_size: int = 128 # 总的train_size, 结果= micro_batch_size *  GPUS * gradient_accumulation_steps
    micro_train_batch_size: int = 4 # 一次micro_train_batch_size 用来计算梯度；只有一次 train_batch_size 才会更新参数
    max_rollout_staleness: int = 1 # off 次数，用来训练的数据和当前策略之间允许的最大差距

    learning_rate: float = 1e-6
    adam_betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01
    lr_scheduler: str = "cosine_with_min_lr"
    lr_warmup_ratio: float = 0.03
    max_steps: int = int(1e4)
    policy_loss_type: str = "ppo"   # 'ppo' / 'gspo'
    reward_func: Optional[EasyDict] = field(default_factory=lambda: EasyDict({
        "format_reward": True,
        "format_param": EasyDict(
            {"format_weight": 0.5, } # fmt_reward 的权重，应该在 [0, 1) 之间，因为advantage的权重是 1 - format_weight
        ),
    }))
    # advantage = target_value - pred_value 
    # advantage_global_batch_norm：意味着 llm训练阶段，所有训练数据的 advantage
    # advantage_batch_norm：意味着 llm 训练过程，train_batch_size之前取advantage
    advantage_type: str = "advantage_global_batch_norm"  # "advantage", "target_reward", "advantage_batch_norm", "advantage_running_norm" "advantage_global_batch_norm"
    eps_clip_low_high: Tuple[float, float] = (0.2, 0.2)
    rft_kl_coef: float = 0.01
    entropy_loss_coef: float = 0.0
    kl_estimator: str = "k3"
    
    llm_save_freq: int = 1000  # 每多少步保存一次 llm 模型,一步代表一次参数更新而不是梯度累积
    save_path: str = "" # 该参数将被 exp_name 目录覆盖
    
    value_norm_cfg: Optional[EasyDict] = field(default_factory=lambda: EasyDict({
        'enable_stability_optimizer': True,
        'value_norm_init_momentum': 0.9,        # Fast adaptation in early training
        'value_norm_final_momentum': 0.99,     # Slow, stable updates in later training
        'value_norm_warmup_steps': 100,           # Steps to transition from init to final momentum
        'value_norm_clip_percentile': 0.95,     # Clip outliers beyond this percentile
        'value_norm_clip_method': "soft",
        "value_norm_history_size": 1000,
    }))


def get_priorzero_config(
    env_id: str = 'detective.z5',
    seed: int = 0,
    exp_name: str = None,
    use_cot: bool = False,
    model_key: Optional[str] = "qwen2.5-3b",
    multi_gpu: bool = False,
) -> Tuple[EasyDict, EasyDict]:
    """
    Generate complete PriorZero configuration with automatic model configuration.

    Args:
        env_id: Jericho game ID
        seed: Random seed
        exp_name: Experiment name (auto-generated if None)
        use_cot: Whether to use Chain-of-Thought reasoning
        model_key: Model configuration key (e.g., 'qwen2.5-0.5b', 'qwen2.5-1.5b', 'qwen2.5-7b')
                  If None, uses default 'qwen2.5-1.5b'

    Returns:
        main_config: Main configuration dictionary
        create_config: Creation configuration for DI-engine components
        llm_config: LLM configuration with auto-configured model parameters
    """
    env_configurations = {
        'detective.z5': (12, 100),
        'omniquest.z5': (25, 100),
        'acorncourt.z5': (45, 50),
        'zork1.z5': (55, 500),
    }
    action_space_size, max_steps = env_configurations.get(env_id, (20, 100))
    wm_encoder_option = 'legacy' 
    # wm_model_name = 'BAAI/bge-base-en-v1.5'  
    # wm_model_name = '/mnt/afs/niuyazhe/workspace/xiongjyu/models/bge-base-en-v1.5'
    wm_model_name = '/mnt/shared-storage-user/puyuan/xiongjyu/models/bge-base-en-v1.5' 
    
    collector_env_num = 1
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
    replay_buffer_size = int(3e5)
    
    env_config = dict(
        stop_value=int(1e6),
        max_steps=max_steps,
        observation_shape=512,  
        env_id=env_id,
        game_path=f"/mnt/shared-storage-user/puyuan/xiongjyu/LightZero/zoo/jericho/envs/z-machine-games-master/jericho-game-suite/{env_id}",
        # game_path=f"/mnt/afs/niuyazhe/workspace/xiongjyu/LightZero/zoo/jericho/envs/z-machine-games-master/jericho-game-suite/{env_id}",
        # game_path=f"/mnt/shared-storage-user/puyuan/code/LightZero/zoo/jericho/envs/z-machine-games-master/jericho-game-suite/{env_id}",
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
        get_valid_actions_timeout=40
    )
    policy_config = dict(
        type='priorzero',
        multi_gpu=multi_gpu,  
        use_wandb=False,
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
    )

    llm_config = PriorZeroLLMConfig(use_cot=use_cot) # 需要修改 llm 相关的参数，修改以上类即可

    # Apply model configuration
    model_config = get_model_config(model_key)
    llm_config.model_name_or_path = model_config["model_name_or_path"]
    llm_config.vllm_tensor_parallel_size = model_config["vllm_tensor_parallel_size"]
    llm_config.gpu_memory_utilization = model_config["gpu_memory_utilization"]

    if exp_name is None:
        env_name = env_id.replace(".z5", "")
        if llm_config.enable_rft:
            exp_name = (
                f"data_priorzero/llm_rft/priorzero_{env_name}_{model_key}_train_{llm_config.train_mode_dict.mode}/"
                f"useCot_{llm_config.use_cot}_alternate_{llm_config.train_schedule.alternate}/"
                f"mcts_{llm_config.mcts_root_logits_dict.mode}_staleness_{llm_config.max_rollout_staleness}_tbs_{llm_config.train_batch_size}_use_mispo_{llm_config.use_mispo}"
            )
        else:
            exp_name = (
                f"data_priorzero/llm_frozen/priorzero_{env_name}_{model_key}_"
                f"train_{llm_config.train_mode_dict.mode}"
                f"useCot_{llm_config.use_cot}_seed{seed}"
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
            import_names=["zoo.jericho.priorzero.src.priorzero_policy"],
        ),
        collector=dict(
            type="priorzero_segment",
            import_names=["zoo.jericho.priorzero.src.priorzero_collector"],
        ),
        evaluator=dict(
            type="priorzero",
            import_names=["zoo.jericho.priorzero.src.priorzero_evaluator"],
        ),
        replay_buffer=dict(
            type='game_buffer_muzero',
            import_names=['lzero.mcts.buffer.game_buffer_muzero'],
        ),
    )
    main_config = EasyDict(priorzero_config)
    create_config = EasyDict(create_config)

    print(f"[Config] Model configuration applied:")
    print(f"  - Model: {model_key}")
    print(f"  - Path: {llm_config.model_name_or_path}")
    print(f"  - Train Mode: {llm_config.train_mode_dict.mode}")
    print(f"  - Tensor Parallel Size: {llm_config.vllm_tensor_parallel_size}")
    print(f"  - GPU Memory Utilization: {llm_config.gpu_memory_utilization}")
    if llm_config.train_mode_dict.mode == "lora":
        print(
            f"  - LoRA r/alpha/dropout: "
            f"{llm_config.train_mode_dict.lora_r}/"
            f"{llm_config.train_mode_dict.lora_alpha}/"
            f"{llm_config.train_mode_dict.lora_dropout}"
        )
        print(f"  - LoRA target modules: {', '.join(llm_config.train_mode_dict.lora_target_modules)}")

    return main_config, create_config, llm_config


def get_priorzero_debug_config(
    env_id: str = 'detective.z5',
    seed: int = 0,
    exp_name: str = None,
    use_cot: bool = False,
    model_key: Optional[str] = "qwen2.5-3b",
) -> EasyDict:

    main_config, create_config, llm_config = get_priorzero_config(
        env_id=env_id, seed=seed, exp_name=exp_name, use_cot=use_cot, model_key=model_key
    )
    max_steps = 20
    
    batch_size = 8
    collect_num_simulations=2
    eval_num_simulations=2
    num_layers=1
    game_segment_length = 50

    llm_config.train_batch_size = 8  # 总的train_size, 结果= micro_batch_size *  GPUS * gradient_accumulation_steps
    llm_config.micro_train_batch_size = 4
    llm_config.train_schedule.wm_update_iters=2
    llm_config.train_schedule.llm_update_iters=1

    create_config.max_steps = max_steps
    
    main_config.policy.model.world_model_cfg.num_layers = num_layers
    main_config.policy.model.world_model_cfg.game_segment_length = game_segment_length
    main_config.policy.batch_size = batch_size
    main_config.policy.collect_num_simulations = collect_num_simulations
    main_config.policy.eval_num_simulations = eval_num_simulations
    main_config.policy.update_per_collect = 2
    main_config.policy.game_segment_length = game_segment_length
    
    return main_config, create_config, llm_config
