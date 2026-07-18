"""
VL Configuration for PriorZero with Image Input

This module provides configuration for using Vision-Language (VL) models
to generate action priors for image-based environments (e.g., Atari).
"""
from typing import Dict, Tuple, Optional
from easydict import EasyDict
from dataclasses import dataclass, field


# ==============================================================================
# Game Descriptions for VL Prompts
# ==============================================================================
GAME_DESCRIPTIONS = {
    'PongNoFrameskip-v4': (
        "This is Pong. You control the right paddle. "
        "Move the paddle UP or DOWN to hit the ball past the opponent's paddle on the left. "
        "Score points when the opponent misses. First to 21 points wins."
    ),
    'BreakoutNoFrameskip-v4': (
        "This is Breakout. You control a paddle at the bottom of the screen. "
        "Move LEFT or RIGHT to bounce the ball upward and break the colored bricks. "
        "Each brick broken scores points. Don't let the ball fall below the paddle."
    ),
    'SpaceInvadersNoFrameskip-v4': (
        "This is Space Invaders. You control a cannon at the bottom of the screen. "
        "Move LEFT/RIGHT and FIRE to shoot the descending rows of aliens. "
        "Destroy all aliens before they reach the bottom. Use shields for cover."
    ),
    'QbertNoFrameskip-v4': (
        "This is Q*bert. You control Q*bert on a pyramid of cubes. "
        "Jump on each cube to change its color to the target color. "
        "Avoid enemies like Coily the snake. Change all cubes to complete the level."
    ),
    'MsPacmanNoFrameskip-v4': (
        "This is Ms. Pac-Man. Navigate the maze eating dots and power pellets. "
        "Avoid the ghosts unless you've eaten a power pellet, which lets you eat them. "
        "Clear all dots to advance to the next level."
    ),
    'LunarLander-v2': (
        "This is Lunar Lander. You control a spacecraft descending toward a landing pad (between two flags) at coordinate (0,0).\n"
        "Goal: Land gently and perfectly horizontal on the pad.\n"
        "\n"
        "Rewards & Penalties:\n"
        "- Closer to pad / slower speed = Positive reward.\n"
        "- Tilted (not horizontal) = Continuous penalty.\n"
        "- Side engine fire = -0.03 points/frame.\n"
        "- Main engine fire = -0.3 points/frame (10x more expensive!).\n"
        "- Crash = -100 points, Safe landing = +100 points."
    ),
}


# ==============================================================================
# VL Model Configuration Presets
# ==============================================================================
VL_MODEL_CONFIGS = {
    "Qwen2.5-VL-2b": {
        "model_name": "Qwen2.5-VL",
        "model_path": "/mnt/shared-storage-user/puyuan/model/Qwen2.5-VL-2B-Instruct",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.25,
        "description": "Qwen2.5-VL-2B-Instruct (smaller, faster)",
    },
    "Qwen2.5-VL-3b": {
        "model_name": "Qwen2.5-VL",
        "model_path": "/mnt/shared-storage-user/puyuan/model/Qwen2.5-VL-3B-Instruct",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.25,
        "description": "Qwen2.5-VL-3B-Instruct",
    },
    "Qwen2.5-VL-7b": {
        "model_name": "Qwen2.5-VL",
        "model_path": "/mnt/shared-storage-user/puyuan/model/Qwen2.5-VL-7B-Instruct",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.35,
        "description": "Qwen2.5-VL-7B-Instruct (better quality)",
    },
    "Qwen3-VL-2b": {
        "model_name": "Qwen3-VL",
        "model_path": "/mnt/shared-storage-user/puyuan/model/Qwen3-VL-2B-Instruct",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.25,
        "description": "Qwen3-VL-2B-Instruct (smaller, faster)",
    },
    "Qwen3-VL-8b": {
        "model_name": "Qwen3-VL",
        "model_path": "/mnt/shared-storage-user/puyuan/model/Qwen3-VL-8B-Instruct",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.25,
        "description": "Qwen3-VL-8B-Instruct",
    },
}


def get_available_vl_models():
    """Get list of available VL model configurations"""
    return list(VL_MODEL_CONFIGS.keys())


def get_vl_model_config(model_key: str) -> Dict:
    """Get VL model configuration by key"""
    if model_key not in VL_MODEL_CONFIGS:
        available = ", ".join(get_available_vl_models())
        raise ValueError(
            f"Unknown VL model key: {model_key}\n"
            f"Available models: {available}"
        )
    return VL_MODEL_CONFIGS[model_key]


def print_available_vl_models():
    """Print all available VL model configurations"""
    print("\n" + "="*80)
    print("Available VL Model Configurations:")
    print("="*80)
    for key, config in VL_MODEL_CONFIGS.items():
        print(f"\n  {key}:")
        print(f"    Path: {config['model_path']}")
        print(f"    Tensor Parallel Size: {config['tensor_parallel_size']}")
        print(f"    GPU Memory Utilization: {config['gpu_memory_utilization']}")
        print(f"    Description: {config['description']}")
    print("="*80 + "\n")


@dataclass
class PriorZeroVLConfig:
    """Configuration for VL-based PriorZero (image input)"""

    # VL model settings
    model_name_or_path: str = "Qwen2.5-VL-7b"

    vl_model_type: str = "qwen-vl"  # 'qwen-vl', 'llava', 'internvl'

    # Game description for prompts
    game_description: str = ""

    # Training settings (similar to LLM config)
    enable_sft: bool = False
    enable_rft: bool = True
    rft_loss_weight: float = 1.0

    # VL inference settings
    temperature: float = 1.0
    max_new_tokens: int = 128  # CoT reasoning + action selection, no need for 256
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.3

    # vLLM engines 
    enable_vllm: bool = True
    enable_prefix_caching: bool = True
    use_cuda_ipc: bool = False
    vllm_sync_backend: str = "nccl" # vLLM 同步参数使用的后端
    vllm_sync_with_ray: bool = False # 是否使用 ray 来同步 vLLM 参数
    vllm_tensor_parallel_size: int = 1 # 每个vllm engine使用几张GPU张量并行 (Fixed: 1.5B model should use 1 GPU)

    vllm_enable_sleep: bool = True # 是否可以休眠
    enable_vllm_is_correction: bool = False
    vllm_is_truncated_threshold: Tuple[float, float] = (0.5, 5.0)
    use_mispo: bool = False
    mispo_token_truncated_threshold: Tuple[float, float] = (0.5, 2.0)
    mispo_traj_truncated_threshold: Tuple[float, float] = (0.8, 1.2)
    top_p: float = 0.95
    seed: int = 0
    reduction: str = "mean"

    # User prompt settings
    user_prompt_dict: Optional[EasyDict] = field(default_factory=lambda: EasyDict({
        "history_with_reward": True,
        "observation_with_valid_actions": False,
    }))



    # Prior generation settings
    use_prior: bool = True  # Whether to use VL prior
    llm_prior_temperature: float = 2.0  # Temperature for prior distribution (aligned with LLM converged config)

    # MCTS root logits configuration
    mcts_root_logits_dict: Optional[EasyDict] = field(default_factory=lambda: EasyDict({
        "mode": "llm_plus_wm_logits",
        # "mode": "llm_logits",
        "plus_method": "fixed",
        "wm_weight": 0.5,
        "llm_max_weight": 0.7,
        "llm_min_weight": 0.3,
        "max_envsteps": 1e5,
    }))

    # Evaluation settings
    eval_dict: Optional[EasyDict] = field(default_factory=lambda: EasyDict({
        "world_model": True,
        "world_model_llm_prior": True,
        "llm_prior": True,
        "wm_eval_freq": 1000,
        "llm_eval_freq": 100,
        "eval_freq": int(20000),
    }))

    attn_implementation: str = "flash_attention_2" 
    use_cot: bool = True
    cot_weight: float = 0.1  # 控制 cot前缀token的权重，由于重点是action:，所以前缀的token权重调低
    prompt_max_len: int = 8192  # Image + prompt tokens;
    generate_max_len: int = 512  # CoT + action output
    bf16: bool = True

    history_length: int = 3  # Number of recent steps to include in context

    # VLM image mode: controls how many images are sent to the VL model
    # "current_only": only the current frame (default, backward compatible)
    # "first_and_current": first history frame + current frame (2 images max)
    # "all_history": all history frames + current frame (history_length+1 images max)
    vlm_image_mode: str = "current_only"

    # Prompt style: "concise" (shorter, better for small VLMs) or "legacy" (verbose, original)
    prompt_style: str = "legacy"

    # Training settings
    colocate_all_models: bool = True
    policy_model_num_gpus: int = 1
    reference_model_num_gpus: int = 1
    deepspeed_enable_sleep: bool = True

    zero_stage: int = 2
    gradient_checkpointing: bool = False
    gradient_checkpointing_use_reentrant: bool = False

    # Training mode (full or lora)
    train_mode_dict: Optional[EasyDict] = field(default_factory=lambda: EasyDict({
        "mode": "full",  # "full" or "lora"
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_bias": "none",
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
    max_norm: float = 1.0
    ds_tensor_parallel_size: int = 1
    ring_attn_size: int = 1

    # Batch sizes
    train_batch_size: int = 128
    micro_train_batch_size: int = 4
    broadcast_every: int = 1

    # Optimizer settings
    learning_rate: float = 1e-6
    adam_betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01
    lr_scheduler: str = "cosine_with_min_lr"
    lr_warmup_ratio: float = 0.03
    max_steps: int = int(1e4)

    # Loss settings
    policy_loss_type: str = "ppo"
    reward_func: Optional[EasyDict] = field(default_factory=lambda: EasyDict({
        "format_reward": True,
        "format_param": EasyDict(
            {"format_weight": 0.5, }
        ),
    }))
    advantage_type: str = "advantage_batch_norm"
    eps_clip_low_high: Tuple[float, float] = (0.2, 0.2)
    rft_kl_coef: float = 0.01
    entropy_loss_coef: float = 0.0
    kl_estimator: str = "k3"

    # Training schedule
    train_vl_after_wm_warm_step: int = int(1e2)
    vl_save_freq: int = 500
    save_path: str = ""

    # Alternating training schedule (matches LLM config)
    train_schedule: Optional[EasyDict] = field(default_factory=lambda: EasyDict({
        "alternate": True,
        "wm_update_iters": 1e3,
        "llm_update_iters": 1e2,
        "start_phase": "wm",
        "wm_warmup_updates": 0,
    }))

    enable_world_model: bool = True
    enable_rft: bool = True
    max_rollout_staleness: int = 1
    # vl_fixed: If True, VL policy model is frozen (inference only, no PPO training).
    # NOTE: vl_fixed=True is mutually exclusive with enable_rft=True (see validate()).
    vl_fixed: bool = False

    # Value normalization
    value_norm_cfg: Optional[EasyDict] = field(default_factory=lambda: EasyDict({
        'enable_stability_optimizer': True,
        'value_norm_init_momentum': 0.9,
        'value_norm_final_momentum': 0.99,
        'value_norm_warmup_steps': 100,
        'value_norm_clip_percentile': 0.95,
        'value_norm_clip_method': "soft",
        "value_norm_history_size": 1000,
    }))

    def validate(self) -> None:
        """
        Validate configuration consistency and raise on illegal combinations.

        Rules:
            1. enable_rft=True requires vl_fixed=False (PPO training needs a trainable policy model).
            2. When vl_fixed=True the VL inference engine is frozen AND PPO training is
               disabled — the pipeline is collect-only + WM training.
            3. train_schedule.alternate=True requires enable_world_model=True.
        """
        valid_image_modes = ("current_only", "first_and_current", "all_history")
        if self.vlm_image_mode not in valid_image_modes:
            raise ValueError(
                f"[PriorZeroVLConfig] Invalid vlm_image_mode='{self.vlm_image_mode}'.\n"
                f"Must be one of: {valid_image_modes}"
            )

        if self.enable_rft and self.vl_fixed:
            raise ValueError(
                "[PriorZeroVLConfig] Illegal config: enable_rft=True AND vl_fixed=True.\n"
                "  enable_rft=True  → PPO training is enabled, which requires a trainable policy model.\n"
                "  vl_fixed=True    → the VL policy model is frozen (no gradient update).\n"
                "These two flags are mutually exclusive. Either:\n"
                "  (a) Set vl_fixed=False to enable VL PPO training, or\n"
                "  (b) Set enable_rft=False to run WM-only training with a frozen VL prior."
            )

        if self.train_schedule.get("alternate", False) and not self.enable_world_model:
            raise ValueError(
                "[PriorZeroVLConfig] Illegal config: train_schedule.alternate=True but enable_world_model=False.\n"
                "Alternating schedule requires the World Model training phase."
            )

        if not self.enable_rft and not self.enable_world_model:
            raise ValueError(
                "[PriorZeroVLConfig] Illegal config: both enable_rft=False and enable_world_model=False.\n"
                "At least one training objective must be enabled."
            )


def get_priorzero_vl_config(
    env_id: str = 'PongNoFrameskip-v4',
    seed: int = 0,
    exp_name: str = None,
    vl_model_key: Optional[str] = None,
    use_prior: bool = True,
    multi_gpu: bool = False,
    quick_test: bool = False,
) -> Tuple[EasyDict, EasyDict, PriorZeroVLConfig]:
    """
    Generate complete PriorZero configuration with VL for image input.

    Args:
        env_id: Atari environment ID
        seed: Random seed
        exp_name: Experiment name
        vl_model_key: VL model key (e.g., 'qwen-vl-chat', 'llava-1.5-7b')
        use_prior: Whether to use VL prior
        multi_gpu: Whether to use multi-GPU training
        quick_test: Whether to use quick test configuration

    Returns:
        main_config: Main configuration dictionary
        create_config: Creation configuration
        vl_config: VL configuration
    """
    from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map

    # Detect environment type
    is_lunarlander = 'LunarLander' in env_id

    if is_lunarlander:
        action_space_size = 4
    else:
        action_space_size = atari_env_action_space_map[env_id]

    # Base configuration parameters
    if quick_test:
        collector_env_num = 2
        num_segments = 2
        game_segment_length = 200
        evaluator_env_num = 2
        num_simulations = 5
        collect_num_simulations = 5
        eval_num_simulations = 5
        batch_size = 8
        num_layers = 1
        replay_ratio = 0.1
    else:
        # collector_env_num = 8
        # num_segments = 8
        collector_env_num = 4
        num_segments = 4
        game_segment_length = 200
        evaluator_env_num = 3
        num_simulations = 25
        # collect_num_simulations = 25
        collect_num_simulations = 50
        eval_num_simulations = 25
        # eval_num_simulations = 50

        batch_size = 256
        # num_layers = 4
        num_layers = 2
        replay_ratio = 0.25

    num_unroll_steps = 10
    infer_context_length = 4

    # Episode step limits
    if is_lunarlander:
        collect_max_episode_steps = int(10000)
        eval_max_episode_steps = int(10000)
    else:
        collect_max_episode_steps = int(5e3)
        eval_max_episode_steps = int(5e3)

    # Environment configuration
    env_config = dict(
        stop_value=int(1e6),
        env_id=env_id,
        # observation_shape=(3, 64, 64),
        observation_shape=(3, 96, 96),
        image_size=96,
        gray_scale=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False,),
        collect_max_episode_steps=collect_max_episode_steps,
        eval_max_episode_steps=eval_max_episode_steps,
    )

    # Policy configuration
    policy_config = dict(
        type='priorzero',
        multi_gpu=multi_gpu,
        use_wandb=False,
        learn=dict(
            learner=dict(
                hook=dict(save_ckpt_after_iter=1000000,),
            ),
        ),
        model=dict(
            # observation_shape=(3, 64, 64),
            observation_shape=(3, 96, 96),

            action_space_size=action_space_size,
            # ====== [FIX] support range must cover LunarLander reward/value range (-200 ~ +300) ======
            reward_support_range=(-300., 301., 1.),
            value_support_range=(-300., 301., 1.),
            norm_type="BN",
            num_res_blocks=1,
            num_channels=64,
            world_model_cfg=dict(
                norm_type="BN",
                # final_norm_option_in_obs_head='LayerNorm',
                # final_norm_option_in_encoder='LayerNorm',
                # predict_latent_loss_type='mse',
                final_norm_option_in_encoder='SimNorm',
                final_norm_option_in_obs_head='SimNorm',
                predict_latent_loss_type='group_kl',
                support_size=601,
                policy_entropy_weight=5e-3,
                continuous_action_space=False,
                max_blocks=num_unroll_steps,
                max_tokens=2 * num_unroll_steps,
                context_length=2 * infer_context_length,
                device='cuda',
                action_space_size=action_space_size,
                num_layers=num_layers,
                # num_heads=8,
                # embed_dim=768,
                num_heads=4,
                embed_dim=256,
                obs_type='image',  # KEY: Image input with VL prior
                env_num=max(collector_env_num, evaluator_env_num),
                num_simulations=num_simulations,
                game_segment_length=game_segment_length,
                encoder_type='resnet',
                # use_priority=True,
                use_priority=False,
                use_normal_head=True,
                use_softmoe_head=False,
                use_moe_head=False,
                # optim_type='AdamW_mix_lr_wdecay',
                optim_type='AdamW',

                decode_loss_mode=None,
                latent_recon_loss_weight=0,
                task_embed_option=None,
                moe_in_transformer=False,
                multiplication_moe_in_transformer=False,
            )
        ),
        # ====== [FIX] optimizer: AdamW -> AdamW_mix_lr_wdecay (layered lr/wd for encoder/transformer/head) ======
        # optim_type='AdamW_mix_lr_wdecay',
        optim_type='AdamW',

        # weight_decay=1e-2,
        learning_rate=1e-4,
        num_unroll_steps=num_unroll_steps,
        update_per_collect=None,
        replay_ratio=replay_ratio,
        batch_size=batch_size,
        num_simulations=num_simulations,
        td_steps=5,
        train_start_after_envsteps=0,
        game_segment_length=game_segment_length,
        replay_buffer_size=int(5e5),
        eval_freq=int(2e4),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,

        num_segments=collector_env_num,
        action_type="varied_action_space",
        model_path=None,
        reanalyze_ratio=0,
        cos_lr_scheduler=False,
        fixed_temperature_value=0.25,
        manual_temperature_decay=False,
        n_episode=collector_env_num,
        buffer_reanalyze_freq=1 / 1000000,
        reanalyze_batch_size=160,
        reanalyze_partition=0.75,
        device='cuda',

        collect_num_simulations=collect_num_simulations,
        eval_num_simulations=eval_num_simulations,
        off_policy_degree=0,
        enable_async_eval=False,

        # ====== [FIX] grad clip: 10 -> 5, prevent gradient explosion ======
        grad_clip_value=5,
        value_loss_weight=0.25,
        policy_loss_weight=1.0,
        reward_loss_weight=1.0,

        # ====== [FIX] Adaptive entropy weight ======
        # use_adaptive_entropy_weight=True,
        use_adaptive_entropy_weight=False,
        adaptive_entropy_alpha_lr=1e-4,
        target_entropy_start_ratio=0.98,
        target_entropy_end_ratio=0.7,
        target_entropy_decay_steps=100000,
        # ====== [FIX] Encoder-clip annealing (prevents latent state norm from diverging) ======
        # use_encoder_clip_annealing=True,
        use_encoder_clip_annealing=False,
        encoder_clip_anneal_type='cosine',
        encoder_clip_start_value=30.0,
        encoder_clip_end_value=10.0,
        encoder_clip_anneal_steps=100000,
        # ====== [FIX] Priority Experience Replay ======
        # use_priority=True,
        use_priority=False,
        priority_prob_alpha=1,
        priority_prob_beta=1,
        # ====== [FIX] Label smoothing ======
        # policy_ls_eps_start=0.05,
        policy_ls_eps_start=0.0,
        policy_ls_eps_end=0.01,
        policy_ls_eps_decay_steps=50000,
        label_smoothing_eps=0.1,
        # ====== Monitor ======
        monitor_norm_freq=10000,
    )

    main_config = EasyDict(dict(
        env=env_config,
        policy=policy_config,
        exp_name=exp_name or f'data_priorzero_vl/{env_id}_seed{seed}',
        seed=seed
    ))

    if is_lunarlander:
        env_create_cfg = dict(
            type='lunarlander_image',
            import_names=['zoo.box2d.lunarlander.envs.lunarlander_image_env'],
        )
    else:
        env_create_cfg = dict(
            type='atari_lightzero',
            import_names=['zoo.atari.envs.atari_lightzero_env'],
        )

    create_config = EasyDict(dict(
        env=env_create_cfg,
        env_manager=dict(type='subprocess'),
        policy=dict(
            type='priorzero',
            import_names=['zoo.jericho.priorzero.src.priorzero_policy'],
        ),
        collector=dict(
            type='priorzero_segment',
            import_names=['zoo.jericho.priorzero.priorzero_collector_unified'],
        ),
        evaluator=dict(
            type='priorzero',
            import_names=['zoo.jericho.priorzero.src.priorzero_evaluator'],
        ),
        replay_buffer=dict(
            type='game_buffer_muzero',
            import_names=['lzero.mcts.buffer.game_buffer_muzero'],
        ),
    ))

    # VL configuration
    vl_config = PriorZeroVLConfig(use_prior=use_prior)

    # Set game description
    vl_config.game_description = GAME_DESCRIPTIONS.get(env_id, "")

    # Auto-configure VL model
    if use_prior:
        if vl_model_key is None:
            vl_model_key = "qwen-vl-chat"  # Default VL
            print(f"[Config] Using default VL model: {vl_model_key}")

        vl_model_config = get_vl_model_config(vl_model_key)
        vl_config.model_name_or_path = vl_model_config["model_path"]
        vl_config.vl_model_type = vl_model_config["model_name"]
        vl_config.tensor_parallel_size = vl_model_config["tensor_parallel_size"]
        vl_config.gpu_memory_utilization = vl_model_config["gpu_memory_utilization"]

        print(f"[Config] VL configuration applied:")
        print(f"  - Model: {vl_model_key}")
        print(f"  - Path: {vl_config.model_name_or_path}")
        print(f"  - Tensor Parallel Size: {vl_config.tensor_parallel_size}")
        print(f"  - GPU Memory Utilization: {vl_config.gpu_memory_utilization}")

        # Override VL config for quick_test to avoid stuck training
        if quick_test:
            # Reduce WM warmup so VL training phase can be reached sooner
            vl_config.train_schedule = EasyDict({
                "alternate": True,
                "wm_update_iters": 50,      # Reduced from 1000
                "llm_update_iters": 20,      # Reduced from 100
                "start_phase": "wm",
                "wm_warmup_updates": 0,
            })
            # Only run WM+VL prior eval in quick_test (skip slow pure-VL and pure-WM eval)
            vl_config.eval_dict = EasyDict({
                "world_model": False,
                "world_model_llm_prior": True,
                "llm_prior": False,
                "eval_freq": int(50),       # Reduced from 500
            })
    else:
        print(f"[Config] VL prior disabled (use_prior=False)")
        vl_config = None

    return main_config, create_config, vl_config


if __name__ == "__main__":
    # Test configuration generation
    print("PriorZero VL Configuration")
    print("=" * 80)

    # List available models
    print_available_vl_models()

    # Generate test config
    print("\nGenerating test configuration...")
    main_cfg, create_cfg, vl_cfg = get_priorzero_vl_config(
        env_id='PongNoFrameskip-v4',
        seed=0,
        vl_model_key='qwen-vl-chat',
        use_prior=True,
        quick_test=True,
    )

    print("\n✓ Configuration generated successfully")
    print(f"  - Experiment: {main_cfg.exp_name}")
    print(f"  - Environment: {main_cfg.env.env_id}")
    print(f"  - Observation shape: {main_cfg.policy.model.observation_shape}")
    print(f"  - obs_type: {main_cfg.policy.model.world_model_cfg.obs_type}")
    if vl_cfg:
        print(f"  - VL model: {vl_cfg.model_name_or_path}")
        print(f"  - Use prior: {vl_cfg.use_prior}")
