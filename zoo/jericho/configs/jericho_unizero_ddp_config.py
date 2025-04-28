import os
import argparse
from typing import Any, Dict

from easydict import EasyDict


def main(env_id: str = 'detective.z5', seed: int = 0, max_env_step: int = int(1e6)) -> None:
    """
    Main entry point for setting up environment configurations and launching training.

    Args:
        env_id (str): Identifier of the environment, e.g., 'detective.z5'.
        seed (int): Random seed used for reproducibility.

    Returns:
        None
    """
    gpu_num = 4
    collector_env_num: int = 4       # Number of collector environments
    n_episode = int(collector_env_num*gpu_num)
    batch_size = int(8*gpu_num)

    # ------------------------------------------------------------------
    # Base environment parameters (Note: these values might be adjusted for different env_id)
    # ------------------------------------------------------------------
    # Define environment configurations

    # v1
    env_configurations = {
        'detective.z5': (12, 100),
        'omniquest.z5': (25, 150),
        'acorncourt.z5': (30, 50),
        'zork1.z5': (40, 400),
    }

    # v0
    # env_configurations = {
    #     'detective.z5': (10, 50),
    #     'omniquest.z5': (10, 100),
    #     'acorncourt.z5': (10, 50),
    #     'zork1.z5': (10, 400),
    # }

    # msteps=500 bug

    env_id = 'detective.z5'
    # env_id = 'omniquest.z5'
    # env_id = 'acorncourt.z5'
    # env_id = 'zork1.z5'

    # Set action_space_size and max_steps based on env_id
    action_space_size, max_steps = env_configurations.get(env_id, (10, 50))  # Default values if env_id not found

    # ------------------------------------------------------------------
    # User frequently modified configurations
    # ------------------------------------------------------------------
    evaluator_env_num: int = 3       # Number of evaluator environments
    num_simulations: int = 50        # Number of simulations

    # Project training parameters
    num_unroll_steps: int = 10       # Number of unroll steps (for rollout sequence expansion)
    infer_context_length: int = 4    # Inference context length

    num_layers: int = 2              # Number of layers in the model
    # replay_ratio: float = 0.25       # Replay ratio for experience replay
    replay_ratio: float = 0.1       # Replay ratio for experience replay
    embed_dim: int = 768             # Embedding dimension

    # Reanalysis (reanalyze) parameters:
    # buffer_reanalyze_freq: Frequency of reanalysis (e.g., 1 means reanalyze once per epoch)
    buffer_reanalyze_freq: float = 1 / 100000
    # reanalyze_batch_size: Number of sequences to reanalyze per reanalysis process
    reanalyze_batch_size: int = 160
    # reanalyze_partition: Partition ratio from the replay buffer to use during reanalysis
    reanalyze_partition: float = 0.75

    # Model name or path - configurable according to the predefined model paths or names
    model_name: str = 'BAAI/bge-base-en-v1.5'

    # ------------------------------------------------------------------
    # TODO: Debug configuration - override some parameters for debugging purposes
    # ------------------------------------------------------------------
    # max_env_step = int(2e5) 
    # batch_size = 10  
    # num_simulations = 2 
    # num_unroll_steps = 5
    # infer_context_length = 2
    # max_steps = 10
    # num_layers = 1
    # replay_ratio = 0.05             
    # ------------------------------------------------------------------
    # Configuration dictionary for the Jericho Unizero environment and policy
    # ------------------------------------------------------------------
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
            multi_gpu=True,  # Important for distributed data parallel (DDP)
            use_wandb=False,
            learn=dict(
                learner=dict(
                    hook=dict(
                        save_ckpt_after_iter=1000000,
                    ),
                ),
            ),
            accumulation_steps=1,  # TODO: Accumulated gradient steps (currently default)
            model=dict(
                observation_shape=512,
                action_space_size=action_space_size,
                encoder_url=model_name,
                model_type="mlp",
                continuous_action_space=False,
                world_model_cfg=dict(
                    final_norm_option_in_obs_head='LayerNorm',
                    final_norm_option_in_encoder='LayerNorm',
                    predict_latent_loss_type='mse', # TODO: for latent state layer_norm
                                        
                    # final_norm_option_in_obs_head='SimNorm',
                    # final_norm_option_in_encoder='SimNorm',
                    # predict_latent_loss_type='group_kl', # TODO: only for latent state sim_norm
                    policy_entropy_weight=5e-2,
                    # policy_entropy_weight=0.5,
                    continuous_action_space=False,
                    max_blocks=num_unroll_steps,
                    # Note: Each timestep contains 2 tokens: observation and action.
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    device="cuda",
                    action_space_size=action_space_size,
                    num_layers=num_layers,
                    num_heads=24,
                    embed_dim=embed_dim,
                    obs_type="text",  # TODO: Modify as needed.
                    env_num=max(collector_env_num, evaluator_env_num),
                    latent_recon_loss_weight=0.1
                ),
            ),
            update_per_collect=int(collector_env_num*max_steps*replay_ratio),  # Important for DDP
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
            # manual_temperature_decay=True,
            # threshold_training_steps_for_final_lr_temperature=3000,

            num_simulations=num_simulations,
            n_episode=n_episode,
            train_start_after_envsteps=0,  # TODO: Adjust training start trigger if needed.
            replay_buffer_size=int(5e5),
            eval_freq=int(1e4),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            # Reanalysis key parameters:
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
        ),
    )
    jericho_unizero_config = EasyDict(jericho_unizero_config)

    # ------------------------------------------------------------------
    # Create configuration for importing environment and policy modules
    # ------------------------------------------------------------------
    jericho_unizero_create_config: Dict[str, Any] = dict(
        env=dict(
            type="jericho",
            import_names=["zoo.jericho.envs.jericho_env"],
        ),
        # Use base env manager to avoid bugs present in subprocess env manager.
        env_manager=dict(type="base"),
        # If necessary, switch to subprocess env manager by uncommenting the following line:
        # env_manager=dict(type="subprocess"),
        policy=dict(
            type="unizero",
            import_names=["lzero.policy.unizero"],
        ),
    )
    jericho_unizero_create_config = EasyDict(jericho_unizero_create_config)

    # ------------------------------------------------------------------
    # Combine configuration dictionaries and construct an experiment name
    # ------------------------------------------------------------------
    main_config: EasyDict = jericho_unizero_config
    create_config: EasyDict = jericho_unizero_create_config

    from ding.utils import DDPContext
    from lzero.config.utils import lz_to_ddp_config
    with DDPContext():
        main_config = lz_to_ddp_config(main_config)
        # Construct experiment name containing key parameters
        main_config.exp_name = (
            f"data_lz/data_unizero_jericho_0428/bge-base-en-v1.5/uz_final-ln_ddp-{gpu_num}gpu_cen{collector_env_num}_rr{replay_ratio}_ftemp025_{env_id[:8]}_ms{max_steps}_ass-{action_space_size}_"
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
        help='Identifier of the environment, e.g., detective.z5 or zork1.z5',
        default='detective.z5'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility',
        default=0
    )
    args = parser.parse_args()

    # Disable tokenizer parallelism to prevent multi-process conflicts
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Start the main process with the provided arguments
    main(args.env, args.seed)
# 