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
    env_id = 'zork1.z5'

    collector_env_num: int = 4      # Number of collector environments
    n_episode = int(collector_env_num)
    
    encoder_option = 'legacy'        # ['qwen', 'legacy']. Legacy uses the bge encoder
    if encoder_option == 'qwen':
        model_name: str = 'Qwen/Qwen3-0.6B'
        batch_size = 4
        accumulation_steps=16
    elif encoder_option == 'legacy':
        model_name: str = 'BAAI/bge-base-en-v1.5'
        batch_size = 64
        accumulation_steps=1
    else:
        raise ValueError(f"Unsupported encoder option: {encoder_option}")    
    
    
    # key exploration related config
    eps_greedy_exploration_in_collect = True
    input_type = 'obs'  # options=['obs', 'latent_state', 'obs_latent_state']
    target_model_for_intrinsic_reward_update_type = 'assign'  # 'assign' or 'momentum'
    
    # ------------------------------------------------------------------
    # Base environment parameters (Note: these values might be adjusted for different env_id)
    # ------------------------------------------------------------------
    # Define environment configurations
    env_configurations = {
        'detective.z5': (12, 100),
        'omniquest.z5': (25, 100),
        'acorncourt.z5': (45, 50),
        'zork1.z5': (55, 500),
    }

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
    replay_ratio: float = 0.1       # Replay ratio for experience replay
    embed_dim: int = 768             # Embedding dimension

    # Reanalysis (reanalyze) parameters:
    # buffer_reanalyze_freq: Frequency of reanalysis (e.g., 1 means reanalyze once per epoch)
    buffer_reanalyze_freq: float = 1 / 100000
    # reanalyze_batch_size: Number of sequences to reanalyze per reanalysis process
    reanalyze_batch_size: int = 160
    # reanalyze_partition: Partition ratio from the replay buffer to use during reanalysis
    reanalyze_partition: float = 0.75

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
    jericho_unizero_rnd_config: Dict[str, Any] = dict(
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
        reward_model=dict(
            type='rnd_unizero',
            intrinsic_reward_type='add',
            input_type=input_type,  # options=['obs', 'latent_state', 'obs_latent_state']
            # intrinsic_reward_weight means the relative weight of RND intrinsic_reward.
            # Specifically for sparse reward env MiniGrid, in this env, if we reach goal, the agent gets reward ~1, otherwise 0.
            # We could set the intrinsic_reward_weight approximately equal to the inverse of max_episode_steps.Please refer to rnd_reward_model for details.
            intrinsic_reward_weight=0.003,  # 1/300
            obs_shape=512,
            latent_state_dim=256,
            hidden_size_list=[256, 256],
            learning_rate=3e-3,
            weight_decay=1e-4,
            batch_size=batch_size, 
            update_per_collect=200,
            rnd_buffer_size=int(1e6),
            input_norm=True,
            input_norm_clamp_max=5,
            input_norm_clamp_min=-5,
            extrinsic_reward_norm=True,
            extrinsic_reward_norm_max=10,
            # extrinsic_reward_norm_max=1,
        ),
        policy=dict(
            multi_gpu=False,  
            use_wandb=False,
            learn=dict(
                learner=dict(
                    hook=dict(
                        save_ckpt_after_iter=1000000, # To save memory, set a large value. If intermediate checkpoints are needed, reduce this value.
                    ),
                ),
            ),
            accumulation_steps=accumulation_steps,  # TODO: Accumulated gradient steps (currently default)
            model=dict(
                observation_shape=512,
                action_space_size=action_space_size,
                encoder_option=encoder_option,
                encoder_url=model_name,
                model_type="mlp",
                continuous_action_space=False,
                world_model_cfg=dict(
                    final_norm_option_in_obs_head='LayerNorm',
                    final_norm_option_in_encoder='LayerNorm',
                    predict_latent_loss_type='mse',
                    policy_entropy_weight=5e-2,
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
                    obs_type="text",
                    env_num=max(collector_env_num, evaluator_env_num),
                    decode_loss_mode=None, # Controls where to compute reconstruction loss: after_backbone, before_backbone, or None.
                    latent_recon_loss_weight=0.1
                ),
            ),
            use_rnd_model=True,
            # RND related config
            use_momentum_representation_network=True,
            target_model_for_intrinsic_reward_update_type=target_model_for_intrinsic_reward_update_type,
            target_update_freq_for_intrinsic_reward=1000,
            target_update_theta_for_intrinsic_reward=0.005,
            eps=dict(
                    eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
                    decay=int(2e5),
                ),
            update_per_collect=int(collector_env_num*max_steps*replay_ratio*accumulation_steps ),  # Important for DDP
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
    jericho_unizero_rnd_config = EasyDict(jericho_unizero_rnd_config)

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
        policy=dict(
            type="unizero",
            import_names=["lzero.policy.unizero"],
        ),
    )
    jericho_unizero_create_config = EasyDict(jericho_unizero_create_config)

    # ------------------------------------------------------------------
    # Combine configuration dictionaries and construct an experiment name
    # ------------------------------------------------------------------
    main_config: EasyDict = jericho_unizero_rnd_config
    create_config: EasyDict = jericho_unizero_create_config

    # Construct experiment name containing key parameters
    main_config.exp_name = (
        f"data_lz/data_unizero_jericho_rnd/{encoder_option}/{env_id}/uz_gpu_cen{collector_env_num}_rr{replay_ratio}_ftemp025_{env_id[:8]}_ms{max_steps}_ass-{action_space_size}_"
        f"nlayer{num_layers}_embed{embed_dim}_Htrain{num_unroll_steps}-"
        f"Hinfer{infer_context_length}_bs{batch_size}_{input_type}_{target_model_for_intrinsic_reward_update_type}_seed{seed}"
    )
    from lzero.entry import train_unizero_with_reward_model
    # Launch the training process
    train_unizero_with_reward_model(
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