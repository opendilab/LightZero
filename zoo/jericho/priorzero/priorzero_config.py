# priorzero_config.py
import os
from easydict import EasyDict

def get_priorzero_config(env_id: str = 'zork1.z5', seed: int = 0) -> EasyDict:
    """
    为 PriorZero 算法生成配置，融合 UniZero 和 LLM 设置。
    """
    # ==============================================================
    # 1. UniZero 世界模型和 MCTS 的基础配置
    # ==============================================================
    action_space_size, max_steps = 20, 100  # Jericho 默认值, 可被环境覆盖

    # 世界模型的编码器 (可以与主策略LLM不同)
    wm_encoder_option = 'legacy'
    wm_model_name = 'BAAI/bge-base-en-v1.5'

    priorzero_config = dict(
        env=dict(
            stop_value=int(1e6),
            max_steps=max_steps,
            observation_shape=768,  # 嵌入维度
            action_space_size=action_space_size,
            env_id=env_id,
            # Jericho 环境特定配置
            jericho_setting=dict(
                game_path=f"./z-machine-games-master/jericho-game-suite/{env_id}",
                tokenizer_path=wm_model_name,
            ),
            collector_env_num=4, # 减少数量以便于本地测试
            evaluator_env_num=2,
            n_evaluator_episode=2,
            manager=dict(shared_memory=False),
        ),
        policy=dict(
            type='priorzero', # 注册我们的新策略
            # ==============================================================
            # 2. LLM 策略 (ORZ-style) 的配置
            # ==============================================================
            llm_policy_cfg=dict(
                # 主策略LLM的模型路径
                pretrain_llm_path="Qwen/Qwen1.5-1.8B-Chat", # 使用一个较小的模型进行演示
                # vLLM 设置
                vllm_tensor_parallel_size=1,
                gpu_memory_utilization=0.6,
                # LLM 策略训练 (RFT/SFT) 设置
                llm_learning_rate=1e-6,
                llm_weight_decay=0.01,
                llm_loss_weight=0.5, # LLM损失在总损失中的权重
                # Prompting
                prompt_max_len=2048,
                generate_max_len=128,
            ),
            # ==============================================================
            # 3. UniZero 世界模型和 MCTS 的配置 (从您提供的代码中提取)
            # ==============================================================
            model=dict(
                model_type='mlp',
                observation_shape=768,
                action_space_size=action_space_size,
                world_model_cfg=dict(
                    encoder_option=wm_encoder_option,
                    encoder_url=wm_model_name,
                    num_layers=4,
                    num_heads=12,
                    embed_dim=768,
                    obs_type="text",
                    context_length=8,
                    num_unroll_steps=10,
                    device="cuda",
                    # ... 其他 UniZero world_model_cfg 参数 ...
                ),
                # ... 其他 UniZero model 参数 ...
            ),
            # MCTS 设置
            num_simulations=25,
            collect_num_simulations=25,
            eval_num_simulations=25,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.25,
            # 世界模型训练设置
            batch_size=32,
            num_unroll_steps=10,
            td_steps=5,
            learning_rate=3e-4,  # 世界模型的学习率
            weight_decay=1e-4,
            optim_type='AdamW',
            grad_clip_value=20,
            # Replay Buffer 设置
            replay_buffer_size=int(1e4),
            # 其他 RL 设置
            eval_freq=int(500),
            train_start_after_envsteps=1000,
            # ... 其他 UniZero policy 参数 ...
            ignore_done=False,
            game_segment_length=200,
            num_segments=4, # 必须等于 collector_env_num
        ),
    )

    # ==============================================================
    # 4. DI-engine 的 Create Config
    # ==============================================================
    create_config = dict(
        env=dict(
            type="jericho",
            import_names=["zoo.jericho.envs.jericho_env"],
        ),
        env_manager=dict(type="base"),
        policy=dict(
            type="priorzero",
            import_names=["priorzero_policy"], # 指向我们的新策略文件
        ),
        collector=dict(
            type="priorzero_segment",
            import_names=["priorzero_collector"], # 指向我们的新收集器文件
        ),
        evaluator=dict(
            type="priorzero",
            import_names=["priorzero_evaluator"], # 指向我们的新评估器文件
        ),
        replay_buffer=dict(
            type='game',
            import_names=['lzero.mcts.buffer.game_buffer_muzero'],
        )
    )

    main_config = EasyDict(priorzero_config)
    create_config = EasyDict(create_config)
    
    main_config.exp_name = f"data_lz/priorzero/{env_id}_qwen1.8b_seed{seed}"
    
    return main_config, create_config