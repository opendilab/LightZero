"""
BabyAI UniZero Baseline Config (Ablation)
==========================================
Pure UniZero world-model baseline for BabyAI multi-task (18 levels).
No LLM module, no llm-prior, no vLLM — only the world model + MCTS.

Corresponding LLM-prior experiment config:
    zoo/babyai/priorzero/src/priorzero_config.py  (get_priorzero_config)

All world-model hyperparameters (embed_dim, num_layers, num_heads, batch_size,
learning_rate, replay_buffer_size, num_simulations, game_segment_length, etc.)
are kept identical to the PriorZero config for a fair ablation comparison.

Entry point:
    lzero.entry.train_unizero_segment
"""
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from easydict import EasyDict


def main(
    env_id: str = 'babyai',
    seed: int = 0,
    env_addr: str = 'http://127.0.0.1:8000',
    use_high_level_actions: bool = True,
    max_env_step: int = int(5e5),
) -> None:

    # === Environment (aligned with PriorZero config) ===
    action_space_size = 20
    max_steps = 20
    wm_encoder_option = 'legacy'
    wm_model_name = '/mnt/shared-storage-user/puyuan/xiongjyu/models/bge-base-en-v1.5'

    _SCALING_INTER_RL_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 19, 20, 21, 30, 31, 33, 36]
    train_data_idx_list = [lvl - 1 for lvl in _SCALING_INTER_RL_LEVELS]
    eval_data_idx_list = [lvl - 1 for lvl in _SCALING_INTER_RL_LEVELS]

    # === Collector / Evaluator (aligned with PriorZero config) ===
    collector_env_num = 1
    evaluator_env_num = 4
    n_episode = collector_env_num
    n_evaluator_episode = len(eval_data_idx_list)  # 18

    # === World Model (aligned with PriorZero config) ===
    num_unroll_steps = 10
    infer_context_length = 4
    game_segment_length = 50
    num_layers = 2
    embed_dim = 768
    replay_ratio = 0.1
    batch_size = 64
    num_simulations = 50
    replay_buffer_size = int(3e5)

    # ------------------------------------------------------------------
    babyai_unizero_config = dict(
        env=dict(
            stop_value=int(1e6),
            max_steps=max_steps,
            observation_shape=512,
            env_id=env_id,
            env_addr=env_addr,
            train_data_idx_list=train_data_idx_list,
            eval_data_idx_list=eval_data_idx_list,
            use_high_level_actions=use_high_level_actions,
            for_unizero=True,
            tokenizer_path=wm_model_name,
            max_action_num=action_space_size,
            max_seq_len=512,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=n_evaluator_episode,
            manager=dict(shared_memory=False),
        ),
        policy=dict(
            multi_gpu=False,
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
                ),
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
            eval_freq=int(5e3),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            buffer_reanalyze_freq=1 / 1000000,
            reanalyze_batch_size=160,
            reanalyze_partition=0.75,
            device='cuda',
            num_simulations=num_simulations,
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
        ),
    )
    babyai_unizero_config = EasyDict(babyai_unizero_config)

    babyai_unizero_create_config = dict(
        env=dict(
            type="babyai",
            import_names=["zoo.babyai.priorzero.envs.babyai_env"],
        ),
        env_manager=dict(type="base"),
        policy=dict(
            type="unizero",
            import_names=["lzero.policy.unizero"],
        ),
    )
    babyai_unizero_create_config = EasyDict(babyai_unizero_create_config)

    main_config = babyai_unizero_config
    create_config = babyai_unizero_create_config

    main_config.exp_name = (
        f"data_unizero/babyai/babyai_unizero_18levels_"
        f"nlayer{num_layers}_edim{embed_dim}_gsl{game_segment_length}_"
        f"rr{replay_ratio}_bs{batch_size}_sim{num_simulations}_seed{seed}"
    )

    from lzero.entry import train_unizero_segment

    train_unizero_segment(
        [main_config, create_config],
        seed=seed,
        model_path=main_config.policy.model_path,
        max_env_step=max_env_step,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BabyAI UniZero Baseline (no LLM)")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env_addr', type=str, default='http://127.0.0.1:8000')
    parser.add_argument('--use_low_level_actions', action='store_true', default=False)
    parser.add_argument('--max_env_step', type=int, default=int(5e5))
    args = parser.parse_args()

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main(
        seed=args.seed,
        env_addr=args.env_addr,
        use_high_level_actions=not args.use_low_level_actions,
        max_env_step=args.max_env_step,
    )
