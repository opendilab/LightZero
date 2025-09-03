import os
import argparse
from typing import Any, Dict

from easydict import EasyDict


def main(env_id: str = 'messenger', seed: int = 0, max_env_step: int = int(1e6)) -> None:
    """
    Main entry point for setting up environment configurations and launching training.

    Args:
        env_id (str): Identifier of the environment, e.g., 'detective.z5'.
        seed (int): Random seed used for reproducibility.

    Returns:
        None
    """
    collector_env_num: int = 1       # Number of collector environments
    n_episode: int = collector_env_num
    batch_size: int = 64
    env_id: str = 'messenger'             
    action_space_size: int = 5
    max_steps: int = 100
    use_manual: bool = True
    task: str ='s1'
    max_seq_len: int = 256
 

    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    evaluator_env_num: int = 1       # Number of evaluator environments
    num_simulations: int = 50        # Number of simulations

    # Project training parameters
    num_unroll_steps: int = 10       # Number of unroll steps (for rollout sequence expansion)
    infer_context_length: int = 4    # Inference context length

    num_layers: int = 2              # Number of layers in the model
    replay_ratio: float = 0.1       # Replay ratio for experience replay
    embed_dim: int = 768             # Embedding dimension

    buffer_reanalyze_freq: float = 1 / 100000
    # reanalyze_batch_size: Number of sequences to reanalyze per reanalysis process
    reanalyze_batch_size: int = 160
    # reanalyze_partition: Partition ratio from the replay buffer to use during reanalysis
    reanalyze_partition: float = 0.75

    model_name: str = 'BAAI/bge-base-en-v1.5'

    messenger_unizero_config = dict(
        env=dict(
            stop_value=int(1e6),
            env_id=env_id,
            observation_shape=(17, 10, 10),
            max_steps=max_steps,
            max_action_num=5,
            n_entities=17,
            mode='train',
            task=task,
            max_seq_len=max_seq_len,
            model_path=model_name,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
        ),
        policy=dict(
            multi_gpu=False,
            use_wandb=False,
            accumulation_steps=1,
            model=dict(
                observation_shape=(17, 10, 10),
                action_space_size=action_space_size,
                downsample=False,
                continuous_action_space=False,
                image_channel=17,
                world_model_cfg=dict(
                    final_norm_option_in_obs_head='LayerNorm',
                    final_norm_option_in_encoder='LayerNorm',
                    predict_latent_loss_type='mse',
                    policy_entropy_weight=5e-2,
                    continuous_action_space=False,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,  # NOTE: each timestep has 2 tokens: obs and action
                    context_length=2 * infer_context_length,
                    device='cuda',
                    action_space_size=action_space_size,
                    num_layers=num_layers,
                    num_heads=8,
                    embed_dim=embed_dim,
                    obs_type='image',
                    env_num=max(collector_env_num, evaluator_env_num),
                    use_manual=use_manual,
                    manual_embed_dim=768,
                ),
            ),
            # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            update_per_collect=int(collector_env_num*max_steps*replay_ratio ),
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            optim_type='AdamW',
            num_simulations=num_simulations,
            n_episode=n_episode,
            replay_buffer_size=int(1e6),
            eval_freq=int(5e3),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
        ),
    )
    messenger_unizero_config = EasyDict(messenger_unizero_config)
    main_config = messenger_unizero_config

    messenger_unizero_create_config = dict(
        env=dict(
            type='messenger',
            import_names=['zoo.messenger.envs.messenger_env'],
        ),
        env_manager=dict(type='base'),
        policy=dict(
            type='unizero',
            import_names=['lzero.policy.unizero'],
        ),
    )
    messenger_unizero_create_config = EasyDict(messenger_unizero_create_config)
    create_config = messenger_unizero_create_config

    main_config.exp_name = (
        f"./data_lz/data_unizero_messenger/{env_id}_use_manual_{use_manual}/uz_cen{collector_env_num}_rr{replay_ratio}_ftemp025_{env_id[:8]}_ms{max_steps}_ass-{action_space_size}_"
        f"nlayer{num_layers}_embed{embed_dim}_Htrain{num_unroll_steps}-"
        f"Hinfer{infer_context_length}_bs{batch_size}_seed{seed}"
    )
    from lzero.entry import train_unizero
    # Launch the training process
    train_unizero(
        [main_config, create_config],
        seed=seed,
        model_path=main_config.policy.model_path,
        max_env_step=max_env_step,
    )

if __name__ == "__main__":
    """
    Overview:
        This script should be executed with <nproc_per_node> GPUs.
        Run the following command to launch the script:
        torchrun --nproc_per_node=4 ./zoo/jericho/configs/jericho_unizero_ddp_config.py
    """

    parser = argparse.ArgumentParser(description='Process environment configuration and launch training.')
    parser.add_argument(
        '--env',
        type=str,
        help='Identifier of the environment',
        default='messenger'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility',
        default=0
    )
    args = parser.parse_args()

    # Start the main process with the provided arguments
    main(args.env, args.seed)