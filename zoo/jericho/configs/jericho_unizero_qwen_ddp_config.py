import os
import argparse
from typing import Any, Dict

from easydict import EasyDict


def main(env_id: str = 'detective.z5', seed: int = 0, max_env_step: int = int(1e6)) -> None:
    """
    DDP entry for Jericho UniZero with Qwen2.5-0.5B as the latent world-model backbone.

    Most settings follow jericho_unizero_ddp_config.py in the current priorzero branch.
    The only intended algorithmic change is enabling the Qwen backbone
    inside UniZero's world model.
    """
    gpu_num = int(os.environ.get("WORLD_SIZE", "4"))
    collector_env_num: int = int(os.environ.get("COLLECTOR_ENV_NUM", "4"))
    n_episode = int(collector_env_num * gpu_num)

    # Keep the observation encoder from the current DDP config. Qwen is used as
    # the world-model backbone, not as the text observation encoder.
    encoder_option = 'legacy'
    model_name: str = '/mnt/afs/niuyazhe/workspace/xiongjyu/models/bge-base-en-v1.5'
    batch_size = int(os.environ.get("BATCH_SIZE", str(64 * gpu_num)))
    accumulation_steps = 1

    qwen_backbone_path: str = '/mnt/afs/niuyazhe/workspace/xiongjyu/models/Qwen2.5-0.5B'

    env_configurations = {
        'detective.z5': (12, 100),
        'omniquest.z5': (25, 100),
        'acorncourt.z5': (45, 50),
        'zork1.z5': (55, 500),
    }
    action_space_size, max_steps = env_configurations.get(env_id, (10, 50))
    max_steps = int(os.environ.get("MAX_STEPS", max_steps))

    evaluator_env_num: int = int(os.environ.get("EVALUATOR_ENV_NUM", "3"))
    num_simulations: int = int(os.environ.get("NUM_SIMULATIONS", "50"))
    num_unroll_steps: int = 10
    infer_context_length: int = 4

    # Qwen2.5-0.5B config: hidden_size=896, layers=24, attention_heads=14, kv_heads=2.
    # UniZero's KV cache stores key/value heads, so num_heads is the number of KV heads.
    num_layers: int = 24
    replay_ratio: float = 0.1
    embed_dim: int = 896
    num_heads: int = 2
    hidden_size: int = 64

    buffer_reanalyze_freq: float = 1 / 100000
    reanalyze_batch_size: int = 160
    reanalyze_partition: float = 0.75

    jericho_unizero_config: Dict[str, Any] = dict(
        env=dict(
            stop_value=int(1e6),
            observation_shape=512,
            max_steps=max_steps,
            max_action_num=action_space_size,
            tokenizer_path=model_name,
            max_seq_len=512,
            game_path=f"./zoo/jericho/envs/z-machine-games-master/jericho-game-suite/{env_id}",
            for_unizero=True,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False),
        ),
        policy=dict(
            multi_gpu=True,
            use_wandb=False,
            learn=dict(
                learner=dict(
                    hook=dict(
                        save_ckpt_after_iter=1000000,
                    ),
                ),
            ),
            accumulation_steps=accumulation_steps,
            model=dict(
                observation_shape=512,
                action_space_size=action_space_size,
                encoder_url=model_name,
                encoder_option=encoder_option,
                model_type="mlp",
                continuous_action_space=False,
                world_model_cfg=dict(
                    final_norm_option_in_obs_head='LayerNorm',
                    final_norm_option_in_encoder='LayerNorm',
                    predict_latent_loss_type='mse',
                    policy_entropy_weight=5e-2,
                    continuous_action_space=False,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    device="cuda",
                    action_space_size=action_space_size,
                    use_qwen_backbone=True,
                    pretrained_path=qwen_backbone_path,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    embed_dim=embed_dim,
                    hidden_size=hidden_size,
                    obs_type="text",
                    env_num=max(collector_env_num, evaluator_env_num),
                    task_embed_option=None,
                    use_task_embed=False,
                    use_normal_head=True,
                    use_softmoe_head=False,
                    use_moe_head=False,
                    num_experts_in_moe_head=4,
                    moe_in_transformer=False,
                    multiplication_moe_in_transformer=False,
                    n_shared_experts=1,
                    num_experts_per_tok=1,
                    num_experts_of_moe_in_transformer=8,
                    lora_r=0,
                    lora_alpha=1,
                    lora_dropout=0.0,
                    decode_loss_mode=None,
                    latent_recon_loss_weight=0.1,
                    game_segment_length=50,
                ),
            ),
            update_per_collect=int(collector_env_num * max_steps * replay_ratio * accumulation_steps),
            action_type="varied_action_space",
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            reanalyze_ratio=0,
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            learning_rate=0.0001,
            cos_lr_scheduler=False,
            fixed_temperature_value=0.25,
            manual_temperature_decay=False,
            num_simulations=num_simulations,
            n_episode=n_episode,
            train_start_after_envsteps=0,
            replay_buffer_size=int(5e5),
            eval_freq=int(3e4),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
        ),
    )
    jericho_unizero_config = EasyDict(jericho_unizero_config)

    jericho_unizero_create_config: Dict[str, Any] = dict(
        env=dict(
            type="jericho",
            import_names=["zoo.jericho.envs.jericho_env"],
        ),
        env_manager=dict(type="base"),
        policy=dict(
            type="unizero",
            import_names=["lzero.policy.unizero"],
        ),
    )
    jericho_unizero_create_config = EasyDict(jericho_unizero_create_config)

    main_config: EasyDict = jericho_unizero_config
    create_config: EasyDict = jericho_unizero_create_config

    from ding.utils import DDPContext
    from lzero.config.utils import lz_to_ddp_config
    with DDPContext():
        main_config = lz_to_ddp_config(main_config)
        main_config.exp_name = (
            f"data_lz/data_unizero_jericho/qwen2.5-0.5B/{env_id}/"
            f"uz_qwen_ddp-{gpu_num}gpu_cen{collector_env_num}_rr{replay_ratio}_"
            f"ftemp025_{env_id[:8]}_ms{max_steps}_ass-{action_space_size}_"
            f"nlayer{num_layers}_embed{embed_dim}_Htrain{num_unroll_steps}-"
            f"Hinfer{infer_context_length}_bs{batch_size}_seed{seed}"
        )
        from lzero.entry import train_unizero
        train_unizero(
            [main_config, create_config],
            seed=seed,
            model_path=main_config.policy.model_path,
            max_env_step=max_env_step,
        )


if __name__ == "__main__":
    """
    Example:
        torchrun --nproc_per_node=4 ./zoo/jericho/configs/jericho_unizero_qwen_ddp_config.py
    """
    parser = argparse.ArgumentParser(description='Process environment configuration and launch training.')
    parser.add_argument(
        '--env',
        type=str,
        help='Identifier of the environment, e.g., detective.z5 or zork1.z5',
        default='detective.z5'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility',
        default=0
    )
    parser.add_argument(
        '--max_env_step',
        type=int,
        help='Maximum number of environment steps',
        default=int(1e6)
    )
    args = parser.parse_args()

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main(args.env, args.seed, args.max_env_step)
