# priorzero_config.py
"""
[PRIORZERO] PriorZero Configuration

This module provides complete configuration for PriorZero algorithm.

Key Features:
- Complete UniZero world model configuration
- LLM policy configuration (ORZ-style)
- Action space mapping for text environments
- Flexible switches to enable/disable components

Author: PriorZero Team
Date: 2025-01-20
"""

import os
from typing import Dict, Tuple
from easydict import EasyDict


def get_jericho_action_mapping(env_id: str = 'zork1.z5') -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Get action mapping for Jericho environments.

    In Jericho, the action space is typically defined by the game's valid actions.
    For simplicity, we'll provide a basic mapping that can be extended.

    Args:
        env_id: Jericho game ID

    Returns:
        action_map: Mapping from action text to action index
        action_inv_map: Mapping from action index to action text
    """
    # Basic common actions for text adventure games
    # These should ideally be loaded from the environment's action space
    common_actions = [
        # Movement
        "go north", "go south", "go east", "go west",
        "go up", "go down", "go northeast", "go northwest",
        "go southeast", "go southwest",
        # Object interaction
        "take all", "drop all", "inventory", "look",
        "examine", "open", "close", "unlock",
        # Common verbs
        "read", "eat", "drink", "wear", "remove",
    ]

    # Create mapping
    action_map = {action.lower(): idx for idx, action in enumerate(common_actions)}
    action_inv_map = {idx: action for action, idx in action_map.items()}

    return action_map, action_inv_map


def get_priorzero_config(
    env_id: str = 'zork1.z5',
    seed: int = 0,
    exp_name: str = None,
    enable_llm: bool = True,
    enable_rft: bool = True,
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

    # ==============================================================================
    # 1. Basic Settings
    # ==============================================================================
    action_space_size = 20  # Default for Jericho
    max_steps = 100

    # World model encoder (for processing text observations)
    wm_encoder_option = 'legacy'  # Options: 'legacy', 'clip', 'custom'
    wm_model_name = 'BAAI/bge-base-en-v1.5'  # Sentence transformer for text encoding

    # LLM policy model
    llm_model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # Smaller model for faster iteration

    # Get action mappings
    action_map, action_inv_map = get_jericho_action_mapping(env_id)

    # ==============================================================================
    # 2. Environment Configuration
    # ==============================================================================
    env_config = dict(
        # Stop conditions
        stop_value=int(1e6),
        max_steps=max_steps,

        # Observation and action space
        observation_shape=768,  # BGE embedding dimension
        action_space_size=action_space_size,

        # Jericho-specific
        env_id=env_id,
        jericho_setting=dict(
            game_path=f"./z-machine-games-master/jericho-game-suite/{env_id}",
            tokenizer_path=wm_model_name,
        ),

        # Parallelization
        collector_env_num=4,
        evaluator_env_num=2,
        n_evaluator_episode=2,

        # Environment manager
        manager=dict(
            shared_memory=False,
            reset_timeout=60,  # Increased timeout for text env initialization
        ),
    )

    # ==============================================================================
    # 3. UniZero World Model Configuration
    # ==============================================================================
    world_model_config = dict(
        # Model type
        model_type='mlp',  # For vector observations (text embeddings)
        continuous_action_space=False,

        # Observation and action
        observation_shape=768,
        action_space_size=action_space_size,

        # World model architecture
        world_model_cfg=dict(
            # Encoder settings
            encoder_option=wm_encoder_option,
            encoder_url=wm_model_name,
            obs_type="text",  # Important: text-based observations

            # Transformer settings
            num_layers=4,  # Reduced for faster training
            num_heads=8,
            embed_dim=768,

            # Context and unroll
            context_length=8,  # Number of past transitions to condition on
            num_unroll_steps=10,  # Number of steps to unroll in training
            tokens_per_block=2,  # obs + action
            max_blocks=10,
            max_tokens=20,  # tokens_per_block * max_blocks

            # Regularization
            embed_pdrop=0.1,
            resid_pdrop=0.1,
            attn_pdrop=0.1,

            # Loss weights
            latent_recon_loss_weight=0.0,  # Latent reconstruction loss
            perceptual_loss_weight=0.0,
            policy_entropy_weight=0.0,  # Entropy regularization

            # Normalization
            final_norm_option_in_head="LayerNorm",
            final_norm_option_in_encoder="LayerNorm",
            predict_latent_loss_type='mse',  # or 'group_kl' with SimNorm

            # Device
            device="cuda",

            # Advanced settings
            gru_gating=False,
            attention='causal',
            support_size=101,  # For distributional RL

            # Analysis flags
            analysis_sim_norm=False,
            analysis_dormant_ratio_weight_rank=False,

            # Position encoding
            rotary_emb=False,  # Whether to use RoPE
            rope_theta=10000,
            max_seq_len=8192,

            # LoRA (optional, for world model)
            lora_r=0,  # Set > 0 to enable LoRA

            # Other
            decode_loss_mode=None,  # 'after_backbone', 'before_backbone', or None
            gamma=1.0,  # Discount factor
            dormant_threshold=0.025,
        ),

        # Distributional RL
        categorical_distribution=True,
        reward_support_range=(-10., 11., 1.),  # (min, max, step) for reward support
        value_support_range=(-50., 51., 1.),   # (min, max, step) for value support

        # Self-supervised learning
        self_supervised_learning_loss=True,

        # Model architecture details
        frame_stack_num=1,
        bias=True,
        res_connection_in_dynamics=True,
        norm_type='LN',  # LayerNorm for text
    )

    # ==============================================================================
    # 4. LLM Policy Configuration (ORZ-style)
    # ==============================================================================
    llm_policy_config = dict(
        # Model path
        pretrain_llm_path=llm_model_name,

        # LoRA for parameter-efficient fine-tuning
        use_lora=False,  # Set to True to enable LoRA
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,

        # Training
        llm_learning_rate=1e-6,
        llm_weight_decay=0.01,
        llm_loss_weight=0.5,   # Weight of SFT loss in total loss
        rft_loss_weight=0.3,   # Weight of RFT loss in total loss

        # Generation
        prompt_max_len=2048,
        generate_max_len=256,  # Max tokens for LLM output

        # Prompting strategy
        history_length=5,      # Number of recent (obs, action, reward) tuples to include
        use_cot=True,          # Whether to use Chain-of-Thought prompting

        # Training strategy
        sft_target='mcts_policy',  # 'mcts_policy' or 'oracle_policy'
        enable_rft=enable_rft,     # Whether to enable RFT with env rewards

        # vLLM settings
        vllm_tensor_parallel_size=1,
        gpu_memory_utilization=0.3,  # Adjust based on your GPU memory
    )

    # ==============================================================================
    # 5. Policy Configuration (Combines World Model + LLM)
    # ==============================================================================
    policy_config = dict(
        type='priorzero',

        # Model config (world model)
        model=world_model_config,

        # [PRIORZERO-NEW] LLM policy config
        llm_policy_cfg=llm_policy_config,

        # [PRIORZERO-NEW] Action mappings
        action_map=action_map,
        action_inv_map=action_inv_map,

        # MCTS settings
        num_simulations=25,
        collect_num_simulations=25,
        eval_num_simulations=25,

        # MCTS exploration
        root_dirichlet_alpha=0.3,
        root_noise_weight=0.25,

        # MCTS variants (set one to True to use that variant)
        sampled_algo=False,  # Sampled MuZero
        gumbel_algo=False,   # Gumbel MuZero
        mcts_ctree=True,     # Use C++ MCTS (faster)

        # Training settings
        batch_size=32,
        learning_rate=3e-4,  # World model learning rate
        weight_decay=1e-4,
        optim_type='AdamW',
        grad_clip_value=10.0,

        # Loss components
        value_loss_weight=1.0,
        policy_loss_weight=1.0,
        reward_loss_weight=1.0,

        # Adaptive entropy weight (for exploration)
        use_adaptive_entropy_weight=True,
        adaptive_entropy_alpha_lr=1e-4,

        # Encoder gradient clipping with annealing
        use_encoder_clip_annealing=True,
        encoder_clip_anneal_type='cosine',
        encoder_clip_start_value=30.0,
        encoder_clip_end_value=10.0,
        encoder_clip_anneal_steps=100000,

        # Training schedule
        num_unroll_steps=10,
        td_steps=5,
        train_start_after_envsteps=1000,
        update_per_collect=None,  # Will be set automatically
        replay_ratio=0.25,

        # Replay buffer
        replay_buffer_size=int(1e4),
        use_priority=True,  # Prioritized experience replay
        priority_prob_alpha=0.6,
        priority_prob_beta=0.4,

        # Evaluation
        eval_freq=500,

        # Game segments
        game_segment_length=200,
        num_segments=4,  # Must equal collector_env_num

        # Misc
        ignore_done=False,
        collect_with_pure_policy=False,
        monitor_extra_statistics=True,

        # Device
        cuda=True,
        device='cuda',
        multi_gpu=False,

        # Environment type
        env_type='not_board_games',
        action_type='fixed_action_space',
        battle_mode='play_with_bot_mode',

        # Data processing
        transform2string=False,
        gray_scale=False,
        use_augmentation=False,

        # Advanced
        use_rnd_model=False,  # Random Network Distillation for exploration
        analysis_sim_norm=False,
        sample_type='transition',
    )

    # ==============================================================================
    # 6. Replay Buffer Configuration
    # ==============================================================================
    replay_buffer_config = dict(
        type='game',
        replay_buffer_size=policy_config['replay_buffer_size'],
        batch_size=policy_config['batch_size'],
    )

    # ==============================================================================
    # 7. Main Configuration Assembly
    # ==============================================================================
    priorzero_config = dict(
        env=env_config,
        policy=policy_config,
        replay_buffer=replay_buffer_config,

        # Experiment settings
        exp_name=exp_name or f"priorzero_{env_id}_seed{seed}",
        seed=seed,
    )

    # ==============================================================================
    # 8. Create Configuration (for DI-engine component creation)
    # ==============================================================================
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
            type='game',
            import_names=['lzero.mcts.buffer.game_buffer_muzero'],
        ),
    )

    # ==============================================================================
    # 9. Convert to EasyDict for convenient access
    # ==============================================================================
    main_config = EasyDict(priorzero_config)
    create_config = EasyDict(create_config)

    # Set experiment path
    main_config.exp_name = f"data_priorzero/{main_config.exp_name}"

    return main_config, create_config


def get_priorzero_config_for_quick_test(env_id: str = 'zork1.z5', seed: int = 0):
    """
    Get a lightweight configuration for quick testing (reduced resources).

    This is useful for:
    - Debugging
    - CI/CD pipelines
    - Local development without powerful GPUs
    """
    main_config, create_config = get_priorzero_config(env_id, seed)

    # Reduce computational requirements
    main_config.env.collector_env_num = 2
    main_config.env.evaluator_env_num = 1
    main_config.env.n_evaluator_episode = 1

    main_config.policy.num_simulations = 10
    main_config.policy.batch_size = 8
    main_config.policy.game_segment_length = 50
    main_config.policy.num_segments = 2
    main_config.policy.replay_buffer_size = 1000

    main_config.policy.model.world_model_cfg.num_layers = 2
    main_config.policy.model.world_model_cfg.num_heads = 4
    main_config.policy.model.world_model_cfg.context_length = 4
    main_config.policy.model.world_model_cfg.num_unroll_steps = 5

    main_config.policy.llm_policy_cfg.prompt_max_len = 1024
    main_config.policy.llm_policy_cfg.generate_max_len = 128
    main_config.policy.llm_policy_cfg.history_length = 3

    main_config.exp_name = f"debug_{main_config.exp_name}"

    return main_config, create_config


# ==============================================================================
# Preset Configurations for Different Scenarios
# ==============================================================================

def get_config_pure_unizero(env_id: str = 'zork1.z5', seed: int = 0):
    """Get config for pure UniZero (without LLM)."""
    main_config, create_config = get_priorzero_config(
        env_id=env_id,
        seed=seed,
        enable_llm=False,
    )
    main_config.exp_name = f"pure_unizero_{env_id}_seed{seed}"
    main_config.policy.llm_policy_cfg.llm_loss_weight = 0.0
    main_config.policy.llm_policy_cfg.rft_loss_weight = 0.0
    return main_config, create_config


def get_config_llm_only_sft(env_id: str = 'zork1.z5', seed: int = 0):
    """Get config for LLM with only SFT (no RFT)."""
    main_config, create_config = get_priorzero_config(
        env_id=env_id,
        seed=seed,
        enable_rft=False,
    )
    main_config.exp_name = f"priorzero_sft_only_{env_id}_seed{seed}"
    return main_config, create_config


def get_config_with_lora(env_id: str = 'zork1.z5', seed: int = 0):
    """Get config with LoRA enabled for LLM (memory efficient)."""
    main_config, create_config = get_priorzero_config(env_id=env_id, seed=seed)
    main_config.policy.llm_policy_cfg.use_lora = True
    main_config.exp_name = f"priorzero_lora_{env_id}_seed{seed}"
    return main_config, create_config


# ==============================================================================
# Example Usage
# ==============================================================================

if __name__ == "__main__":
    # Test configuration generation
    print("="*80)
    print("Testing PriorZero Configuration Generation")
    print("="*80)

    # 1. Standard config
    print("\n1. Standard PriorZero Config:")
    main_cfg, create_cfg = get_priorzero_config(env_id='zork1.z5', seed=0)
    print(f"  Exp name: {main_cfg.exp_name}")
    print(f"  Action space size: {main_cfg.policy.model.action_space_size}")
    print(f"  LLM model: {main_cfg.policy.llm_policy_cfg.pretrain_llm_path}")
    print(f"  World model layers: {main_cfg.policy.model.world_model_cfg.num_layers}")
    print(f"  Num action mappings: {len(main_cfg.policy.action_map)}")

    # 2. Quick test config
    print("\n2. Quick Test Config:")
    test_cfg, _ = get_priorzero_config_for_quick_test()
    print(f"  Batch size: {test_cfg.policy.batch_size}")
    print(f"  Num simulations: {test_cfg.policy.num_simulations}")
    print(f"  Collector envs: {test_cfg.env.collector_env_num}")

    # 3. Pure UniZero config
    print("\n3. Pure UniZero Config:")
    unizero_cfg, _ = get_config_pure_unizero()
    print(f"  LLM loss weight: {unizero_cfg.policy.llm_policy_cfg.llm_loss_weight}")
    print(f"  RFT enabled: {unizero_cfg.policy.llm_policy_cfg.enable_rft}")

    # 4. Config with LoRA
    print("\n4. Config with LoRA:")
    lora_cfg, _ = get_config_with_lora()
    print(f"  Use LoRA: {lora_cfg.policy.llm_policy_cfg.use_lora}")
    print(f"  LoRA rank: {lora_cfg.policy.llm_policy_cfg.lora_r}")

    print("\n" + "="*80)
    print("✓ All configurations generated successfully!")
    print("="*80)
