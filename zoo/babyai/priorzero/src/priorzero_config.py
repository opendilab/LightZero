import os
from typing import Dict, Tuple, Optional, Any
from easydict import EasyDict
import torch.distributed as dist
from dataclasses import dataclass, field

# ============================================================================
# Model Configuration Presets (shared with Jericho version)
# ============================================================================
MODEL_CONFIGS = {
    "qwen2.5-0.5b": {
        "model_name_or_path": "/mnt/shared-storage-user/puyuan/xiongjyu/models/Qwen2.5-0.5B-Instruct",
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
        "model_name_or_path": "/mnt/shared-storage-user/puyuan/xiongjyu/models/Qwen2.5-3B-Instruct",
        "vllm_tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.25,
        "description": "Qwen2.5-3B-Instruct (better quality)",
    },
    "qwen2.5-7b": {
        "model_name_or_path": "/mnt/shared-storage-user/puyuan/xiongjyu/models/Qwen2.5-7B-Instruct",
        "vllm_tensor_parallel_size": 2,
        # "vllm_tensor_parallel_size": 1,

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
    return list(MODEL_CONFIGS.keys())

def get_model_config(model_key: str) -> Dict:
    if model_key not in MODEL_CONFIGS:
        available = ", ".join(get_available_models())
        raise ValueError(f"Unknown model key: {model_key}\nAvailable models: {available}")
    return MODEL_CONFIGS[model_key]


@dataclass
class PriorZeroLLMConfig:
    model_name_or_path: str = "Qwen2.5-3B-Instruct"
    local_rank: int = -1
    enable_rft: bool = True
    enable_world_model: bool = True
    train_mode_dict: Optional[EasyDict] = field(default_factory=lambda: EasyDict({
        "mode": "full",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_bias": "none",
        "lora_target_modules": (
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ),
    }))

    train_schedule: Optional[EasyDict] = field(default_factory=lambda: EasyDict({
        "alternate": True,
        "wm_update_iters": 500,
        "llm_update_iters": 100,
        "start_phase": "wm",
        "wm_warmup_updates": 0,
    }))

    llm_prior_temperature: float = 1.0
    mcts_root_logits_dict: Optional[EasyDict] = field(default_factory=lambda: EasyDict({
        "mode": "llm_plus_wm_logits",
        "plus_method": "fixed",
        "wm_weight": 0.5,
        "llm_max_weight": 0.7,
        "llm_min_weight": 0.3,
        "max_envsteps": 1e5,
    }))
    eval_dict: Optional[EasyDict] = field(default_factory=lambda: EasyDict({
        "world_model": True,
        "world_model_llm_prior": True,
        "llm_prior": True,
        "wm_eval_freq": 2000,   # aligned with ScalingInter-RL: larger eval interval for 40-level multi-task
        "llm_eval_freq": 200,   # aligned with ScalingInter-RL: larger eval interval for 40-level multi-task
    }))

    attn_implementation: str = "flash_attention_2"
    history_length: int = 10
    use_cot: bool = True
    cot_weight: float = 0.1

    user_prompt_dict: Optional[EasyDict] = field(default_factory=lambda: EasyDict({
        "history_with_reward": True,
        "observation_with_valid_actions": True,
    }))

    # Total context budget consumed by line 662 of priorzero_datafactory.py as
    # `max_length = prompt_max_len - generate_max_len - 20`; BabyAI obs typically ≤ 512 tokens.
    prompt_max_len: int = 4096
    generate_max_len: int = 512
    bf16: bool = True

    enable_vllm: bool = True
    enable_prefix_caching: bool = False
    use_cuda_ipc: bool = False
    enable_vllm_is_correction: bool = False
    vllm_is_truncated_threshold: Tuple[float, float] = (0.5, 5.0)
    use_mispo: bool = False
    mispo_token_truncated_threshold: Tuple[float, float] = (0.5, 2.0)
    mispo_traj_truncated_threshold: Tuple[float, float] = (0.8, 1.2)

    vllm_sync_backend: str = "nccl"
    vllm_tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.3
    vllm_enable_sleep: bool = True
    temperature: float = 1.0
    top_p: float = 0.95
    seed: int = 0
    reduction: str = "mean"

    deepspeed_enable_sleep: bool = True
    zero_stage: int = 2
    gradient_checkpointing: bool = False
    gradient_checkpointing_use_reentrant: bool = False
    max_norm: float = 1.0
    ds_tensor_parallel_size: int = 1

    train_batch_size: int = 128
    micro_train_batch_size: int = 2
    max_rollout_staleness: int = 1

    learning_rate: float = 1e-6
    adam_betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01
    lr_scheduler: str = "cosine_with_min_lr"
    lr_warmup_ratio: float = 0.03
    max_steps: int = int(1e4)
    policy_loss_type: str = "ppo"
    reward_func: Optional[EasyDict] = field(default_factory=lambda: EasyDict({
        "format_reward": True,
        "format_param": EasyDict({"format_weight": 0.5}),
    }))
    advantage_type: str = "advantage_global_batch_norm"
    eps_clip_low_high: Tuple[float, float] = (0.2, 0.2)
    rft_kl_coef: float = 0.001
    entropy_loss_coef: float = 0.0
    kl_estimator: str = "k3"

    llm_save_freq: int = 1000
    save_path: str = ""

    value_norm_cfg: Optional[EasyDict] = field(default_factory=lambda: EasyDict({
        'enable_stability_optimizer': True,
        'value_norm_init_momentum': 0.9,
        'value_norm_final_momentum': 0.99,
        'value_norm_warmup_steps': 100,
        'value_norm_clip_percentile': 0.95,
        'value_norm_clip_method': "soft",
        "value_norm_history_size": 1000,
    }))


def get_priorzero_config(
    env_id: str = 'babyai',
    seed: int = 0,
    exp_name: str = None,
    use_cot: bool = True,
    model_key: Optional[str] = "qwen2.5-3b",
    multi_gpu: bool = False,
    env_addr: str = 'http://127.0.0.1:8000',
    use_high_level_actions: bool = True,
) -> Tuple[EasyDict, EasyDict]:

    action_space_size = 20  # upper bound for dynamic action space
    max_steps = 20  # aligned with ScalingInter-RL babyai_train.sh (max_rounds=20)
    wm_encoder_option = 'legacy'
    wm_model_name = '/mnt/shared-storage-user/puyuan/xiongjyu/models/bge-base-en-v1.5'

    # Aligned with ScalingInter-RL (HF: AgentGym/AgentGym-RL-Data-ID, train/babyai_train.json).
    # ScalingInter-RL trains on 18 out of 40 BabyAI levels (810 items, 45 seeds per level).
    # BabyAI level mapping: level_id = data_idx % 40 + 1, seed = data_idx // 40.
    # Using seed=0 (data_idx = level_id - 1) for PriorZero since it re-samples each episode.
    _SCALING_INTER_RL_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 19, 20, 21, 30, 31, 33, 36]
    train_data_idx_list = [lvl - 1 for lvl in _SCALING_INTER_RL_LEVELS]  # 18 levels, seed=0
    eval_data_idx_list = [lvl - 1 for lvl in _SCALING_INTER_RL_LEVELS]   # same 18 levels for eval

    collector_env_num = 1
    # Set evaluator_env_num == n_evaluator_episode so each env runs exactly one episode
    # (covers every eval level once and avoids the buggy `n_episode > env_num` refill path).
    evaluator_env_num = len(eval_data_idx_list)
    evaluator_env_num = 4


    n_episode = collector_env_num
    n_evaluator_episode = len(eval_data_idx_list)  # 18 episodes to cover all eval levels


    # only for debug
    # evaluator_env_num = 2
    # n_evaluator_episode = 2

    num_unroll_steps = 10
    infer_context_length = 4
    game_segment_length = 50
    num_layers = 2
    embed_dim = 768
    replay_ratio = 0.1
    batch_size = 64
    collect_num_simulations = 50
    eval_num_simulations = 50

    # only for debug
    # collect_num_simulations = 2
    # eval_num_simulations = 2

    replay_buffer_size = int(3e5)

    env_config = dict(
        stop_value=int(1e6),
        max_steps=max_steps,
        observation_shape=512,
        env_id=env_id,
        env_addr=env_addr,
        train_data_idx_list=train_data_idx_list,   # aligned with ScalingInter-RL
        eval_data_idx_list=eval_data_idx_list,      # aligned with ScalingInter-RL
        use_high_level_actions=use_high_level_actions,
        for_unizero=True,
        tokenizer_path=wm_model_name,
        max_action_num=action_space_size,
        max_seq_len=512,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=n_evaluator_episode,    # aligned with ScalingInter-RL: cover all eval tasks
        manager=dict(shared_memory=False),
    )
    policy_config = dict(
        type='priorzero',
        multi_gpu=multi_gpu,
        use_wandb=False,
        learn=dict(
            learner=dict(
                hook=dict(save_ckpt_after_iter=1000000),
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
        use_priority=False,
        priority_prob_alpha=0.6,
        priority_prob_beta=0.4,
    )

    llm_config = PriorZeroLLMConfig(use_cot=use_cot)

    model_config = get_model_config(model_key)
    llm_config.model_name_or_path = model_config["model_name_or_path"]
    llm_config.vllm_tensor_parallel_size = model_config["vllm_tensor_parallel_size"]
    llm_config.gpu_memory_utilization = model_config["gpu_memory_utilization"]

    if exp_name is None:
        # aligned with ScalingInter-RL: multi-task across 18 levels
        if llm_config.enable_rft:
            exp_name = (
                f"data_priorzero/babyai/llm_rft/priorzero_multitask_18levels_{model_key}_train_{llm_config.train_mode_dict.mode}/"
                f"useCot_{llm_config.use_cot}_alternate_{llm_config.train_schedule.alternate}/"
                f"mcts_{llm_config.mcts_root_logits_dict.mode}_staleness_{llm_config.max_rollout_staleness}_tbs_{llm_config.train_batch_size}_use_mispo_{llm_config.use_mispo}"
            )
        else:
            exp_name = (
                f"data_priorzero/babyai/llm_frozen/priorzero_multitask_18levels_{model_key}_"
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
            type="babyai",
            import_names=["zoo.babyai.priorzero.envs.babyai_env"],
        ),
        env_manager=dict(type="base"),
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

    train_level_ids = [idx % 40 + 1 for idx in train_data_idx_list]
    eval_level_ids = [idx % 40 + 1 for idx in eval_data_idx_list]
    import logging
    logging.getLogger("priorzero.main").info(
        f"[Config] model={model_key} | {len(train_data_idx_list)} train levels | {len(eval_data_idx_list)} eval levels | high_level={use_high_level_actions}"
    )

    return main_config, create_config, llm_config


def get_priorzero_debug_config(
    env_id: str = 'babyai',
    seed: int = 0,
    exp_name: str = None,
    use_cot: bool = True,
    model_key: Optional[str] = "qwen2.5-3b",
    env_addr: str = 'http://127.0.0.1:8000',
    use_high_level_actions: bool = True,
) -> EasyDict:

    main_config, create_config, llm_config = get_priorzero_config(
        env_id=env_id, seed=seed, exp_name=exp_name, use_cot=use_cot,
        model_key=model_key, env_addr=env_addr,
        use_high_level_actions=use_high_level_actions,
    )
    max_steps = 20
    batch_size = 8
    collect_num_simulations = 2
    eval_num_simulations = 2
    num_layers = 1
    game_segment_length = 50

    llm_config.train_batch_size = 8
    llm_config.micro_train_batch_size = 4
    llm_config.train_schedule.wm_update_iters = 2
    llm_config.train_schedule.llm_update_iters = 1
    llm_config.eval_dict.wm_eval_freq = 2
    llm_config.eval_dict.llm_eval_freq = 1

    main_config.env.max_steps = max_steps
    main_config.policy.model.world_model_cfg.num_layers = num_layers
    main_config.policy.model.world_model_cfg.game_segment_length = game_segment_length
    main_config.policy.batch_size = batch_size
    main_config.policy.collect_num_simulations = collect_num_simulations
    main_config.policy.eval_num_simulations = eval_num_simulations
    main_config.policy.update_per_collect = 2
    main_config.policy.game_segment_length = game_segment_length

    return main_config, create_config, llm_config
