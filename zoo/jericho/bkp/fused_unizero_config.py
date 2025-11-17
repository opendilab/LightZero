# fused_unizero_config.py

import os
from easydict import EasyDict

def get_priorzero_config(env_id: str = 'zork1.z5', seed: int = 0) -> EasyDict:
    """
    Generates the configuration for the PriorZero algorithm, merging UniZero and LLM settings.
    """
    # ==============================================================
    # 1. UniZero Base Configurations
    # ==============================================================
    action_space_size, max_steps = 20, 100 # Default for Jericho, can be overridden

    # World Model Encoder (can be different from the main policy LLM)
    wm_encoder_option = 'legacy'
    if wm_encoder_option == 'legacy':
        wm_model_name = 'BAAI/bge-base-en-v1.5'
    else:
        wm_model_name = 'Qwen/Qwen2-0.5B' # A smaller model for the world model encoder

    jericho_unizero_config = dict(
        env=dict(
            stop_value=int(1e6),
            max_steps=max_steps,
            observation_shape=768, # Embedding dimension
            max_action_num=action_space_size,
            tokenizer_path=wm_model_name,
            game_path=f"./zoo/jericho/envs/z-machine-games-master/jericho-game-suite/{env_id}",
            collector_env_num=8,
            evaluator_env_num=5,
            n_evaluator_episode=5,
            manager=dict(shared_memory=False),
        ),
        policy=dict(
            # This section now primarily configures the World Model and MCTS
            model=dict(
                observation_shape=768,
                action_space_size=action_space_size,
                encoder_option=wm_encoder_option,
                encoder_url=wm_model_name,
                model_type="mlp",
                world_model_cfg=dict(
                    final_norm_option_in_obs_head='LayerNorm',
                    final_norm_option_in_encoder='LayerNorm',
                    predict_latent_loss_type='mse',
                    policy_entropy_weight=5e-3,
                    continuous_action_space=False,
                    max_blocks=10, # num_unroll_steps
                    max_tokens=20,
                    context_length=8, # 2 * infer_context_length
                    device="cuda",
                    action_space_size=action_space_size,
                    num_layers=4,
                    num_heads=12,
                    embed_dim=768,
                    obs_type="text",
                    env_num=8,
                    decode_loss_mode='None',
                    latent_recon_loss_weight=0.1,
                ),
            ),
            # MCTS settings
            num_simulations=50,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.25,
            # World Model training settings
            batch_size=64,
            num_unroll_steps=10,
            td_steps=5,
            learning_rate=3e-4, # LR for World Model
            weight_decay=1e-4,
            # Replay Buffer settings
            replay_buffer_size=int(5e4),
            replay_ratio=0.25,
            # Other RL settings
            eval_freq=int(1e3),
            train_start_after_envsteps=2000,
        ),
    )

    # ==============================================================
    # 2. LLM Policy (ORZ-style) Configurations
    # ==============================================================
    llm_policy_config = dict(
        # Model path for the main LLM policy
        pretrain="Qwen/Qwen2.5-7B",
        # vLLM settings for efficient inference
        vllm_num_engines=jericho_unizero_config['env']['collector_env_num'],
        vllm_tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
        # LLM Policy training settings (RFT/PPO)
        llm_learning_rate=1e-6,
        llm_weight_decay=0.01,
        # Prompting
        prompt_max_len=4096,
        generate_max_len=512,
    )
    
    # Add LLM config to the main policy config
    jericho_unizero_config['policy']['llm_policy_cfg'] = llm_policy_config

    # ==============================================================
    # 3. Create Config for DI-engine
    # ==============================================================
    create_config = dict(
        env=dict(
            type="jericho",
            import_names=["zoo.jericho.envs.jericho_env"],
        ),
        env_manager=dict(type="base"),
        # We will create a custom policy class `PriorZeroPolicy`
        policy=dict(
            type="priorzero", # Register a new policy type
            import_names=["your_project.policy.priorzero_policy"], # Path to your custom policy
        ),
    )

    # ==============================================================
    # 4. Final Touches
    # ==============================================================
    main_config = EasyDict(jericho_unizero_config)
    create_config = EasyDict(create_config)
    
    main_config.exp_name = f"data_lz/priorzero/{env_id}_qwen7b_seed{seed}"
    
    return main_config, create_config