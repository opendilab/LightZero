# -*- coding: utf-8 -*-
"""
Overview:
    This script defines the configuration for a multi-task reinforcement learning experiment
    using the UniZero model on DeepMind Control Suite (DMC) environments.
    It is designed to be launched with PyTorch's Distributed Data Parallel (DDP) for multi-GPU training.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from easydict import EasyDict

# ==============================================================
# Global setup: Logging
# ==============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("output.log", encoding="utf-8"),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)


def get_base_config(env_id_list: list[str], collector_env_num: int, evaluator_env_num: int,
                    num_unroll_steps: int, infer_context_length: int, curriculum_stage_num: int) -> EasyDict:
    """
    Overview:
        Creates the base configuration EasyDict with default settings for the experiment.
        These settings are shared across all tasks but can be overridden.

    Arguments:
        - env_id_list (:obj:`list[str]`): A list of environment IDs for all tasks.
        - collector_env_num (:obj:`int`): The number of environments for data collection.
        - evaluator_env_num (:obj:`int`): The number of environments for evaluation.
        - num_unroll_steps (:obj:`int`): The number of game steps to unroll in the model.
        - infer_context_length (:obj:`int`): The context length for inference.
        - curriculum_stage_num (:obj:`int`): The number of stages in the curriculum learning.

    Returns:
        - (:obj:`EasyDict`): A dictionary containing the base configuration.
    """
    return EasyDict(dict(
        # Environment-specific settings
        env=dict(
            stop_value=int(5e5),
            from_pixels=False,
            continuous=True,  # Assuming all DMC tasks use continuous action spaces
            manager=dict(shared_memory=False),
            game_segment_length=100,
            # TODO(user): For debugging only. Uncomment to use smaller segments and episodes.
            # game_segment_length=10,
            # collect_max_episode_steps=int(40),
            # eval_max_episode_steps=int(40),
        ),
        # Policy-specific settings
        policy=dict(
            multi_gpu=True,  # TODO(user): Enable multi-GPU for DDP.
            # TODO(user): Configure MoCo settings.
            only_use_moco_stats=False,
            use_moco=False,
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000000))),
            grad_correct_params=dict(
                MoCo_beta=0.5, MoCo_beta_sigma=0.5, MoCo_gamma=0.1, MoCo_gamma_sigma=0.5, MoCo_rho=0,
                calpha=0.5, rescale=1,
            ),
            total_task_num=len(env_id_list),
            task_num=len(env_id_list),
            # Model configuration
            model=dict(
                continuous_action_space=True,
                num_of_sampled_actions=20,
                model_type='mlp',
                world_model_cfg=dict(
                    final_norm_option_in_obs_head='LayerNorm',
                    final_norm_option_in_encoder='LayerNorm',
                    predict_latent_loss_type='mse',  # TODO(user): Loss type for latent state with LayerNorm.
                    
                    share_head=False,  # TODO(user): Whether to share the prediction head across tasks.
                    use_shared_projection=False,

                    # TODO(user): analysis_dormant_ratio needs to be corrected for the DMC encoder.
                    analysis_dormant_ratio_weight_rank=False,
                    analysis_dormant_ratio_interval=5000,
                    # analysis_dormant_ratio_interval=20, # For debugging
                    
                    # TODO(user): Configure task embedding options.
                    task_embed_option=None,
                    use_task_embed=False,
                    # task_embed_option='concat_task_embed',
                    # use_task_embed=True,
                    # task_embed_dim=128,

                    policy_loss_type='kl',
                    obs_type='vector',
                    policy_entropy_weight=5e-2,
                    continuous_action_space=True,
                    num_of_sampled_actions=20,
                    sigma_type='conditioned',
                    fixed_sigma_value=0.5,
                    bound_type=None,
                    model_type='mlp',
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,  # Each timestep has 2 tokens: obs and action
                    context_length=2 * infer_context_length,
                    device='cuda',
                    
                    # TODO(user): For debugging only. Use a smaller model.
                    # num_layers=1,
                    num_layers=4,
                    # num_layers=8,

                    num_heads=24,
                    embed_dim=768,
                    env_num=max(collector_env_num, evaluator_env_num),
                    task_num=len(env_id_list),
                    
                    # Mixture of Experts (MoE) head configuration
                    use_normal_head=True,
                    use_softmoe_head=False,
                    use_moe_head=False,
                    num_experts_in_moe_head=4,

                    # MoE in Transformer configuration
                    moe_in_transformer=False,
                    multiplication_moe_in_transformer=True,
                    n_shared_experts=1,
                    num_experts_per_tok=1,
                    num_experts_of_moe_in_transformer=8,
                    
                    # LoRA (Low-Rank Adaptation) parameters
                    # TODO(user): Enable or disable LoRA for MoE layers.
                    moe_use_lora=True,
                    lora_target_modules=["attn", "feed_forward"],
                    lora_r=64,
                    lora_alpha=1,
                    lora_dropout=0.0,
                    lora_scale_init=1,

                    # Curriculum learning stage iteration counts
                    curriculum_stage_num=curriculum_stage_num,
                    min_stage0_iters=10000,  # Corresponds to 400k envsteps, 40k iters
                    max_stage_iters=5000,

                    # TODO(user): For debugging only. Use very short stage iterations.
                    # min_stage0_iters=2,
                    # max_stage_iters=5,
                ),
            ),
            # TODO(user): Enable or disable task exploitation weight.
            use_task_exploitation_weight=False,
            balance_pipeline=True,
            # TODO(user): Enable or disable task complexity weight.
            task_complexity_weight=True,
            allocated_batch_sizes=False,
            # TODO(user): Set the number of environment steps to collect before training starts.
            train_start_after_envsteps=int(0),
            use_priority=False,
            print_task_priority_logs=False,
            cuda=True,
            model_path=None,
            
            # TODO(user): For debugging only. Set a smaller update_per_collect.
            # update_per_collect=3,
            update_per_collect=200,  # e.g., 8 envs * 100 steps/env * 0.25 replay_ratio = 200
            replay_buffer_size=int(1e6),
            eval_freq=int(4e3),
            grad_clip_value=5,
            learning_rate=1e-4,
            discount_factor=0.99,
            td_steps=5,
            piecewise_decay_lr_scheduler=False,
            manual_temperature_decay=True,
            threshold_training_steps_for_final_temperature=int(2.5e4),
            cos_lr_scheduler=True,
        ),
    ))


def create_task_config(
        base_config: EasyDict,
        env_id: str,
        observation_shape_list: list[int],
        action_space_size_list: list[int],
        target_return_dict: dict[str, int],
        collector_env_num: int,
        evaluator_env_num: int,
        n_episode: int,
        num_simulations: int,
        reanalyze_ratio: float,
        batch_size: int,
        num_unroll_steps: int,
        norm_type: str,
        buffer_reanalyze_freq: float,
        reanalyze_batch_size: int,
        reanalyze_partition: float,
        num_segments: int,
        total_batch_size: int
) -> EasyDict:
    """
    Overview:
        Creates a specialized configuration for a single task by updating the base config.

    Arguments:
        - base_config (:obj:`EasyDict`): The base configuration dictionary.
        - env_id (:obj:`str`): The ID of the environment for this specific task.
        - observation_shape_list (:obj:`list[int]`): List of observation shapes for all tasks.
        - action_space_size_list (:obj:`list[int]`): List of action space sizes for all tasks.
        - target_return_dict (:obj:`dict[str, int]`): A dictionary mapping env_id to its target return.
        - collector_env_num (:obj:`int`): The number of collector environments.
        - evaluator_env_num (:obj:`int`): The number of evaluator environments.
        - n_episode (:obj:`int`): The number of episodes to run for collection.
        - num_simulations (:obj:`int`): The number of simulations in MCTS.
        - reanalyze_ratio (:obj:`float`): The ratio of reanalyzed data in a batch.
        - batch_size (:obj:`int`): The batch size for training this task.
        - num_unroll_steps (:obj:`int`): The number of steps to unroll the model.
        - norm_type (:obj:`str`): The type of normalization to use (e.g., 'LN').
        - buffer_reanalyze_freq (:obj:`float`): Frequency of buffer reanalysis.
        - reanalyze_batch_size (:obj:`int`): Batch size for reanalysis.
        - reanalyze_partition (:obj:`float`): Partition ratio for reanalysis.
        - num_segments (:obj:`int`): The number of segments in the replay buffer.
        - total_batch_size (:obj:`int`): The total batch size across all tasks.

    Returns:
        - (:obj:`EasyDict`): The final configuration for the specified task.
    """
    domain_name, task_name = env_id.split('-', 1)
    frame_skip = 8 if domain_name == "pendulum" else 4

    config = base_config
    
    # Update environment settings
    config.env.update(dict(
        env_id=env_id,
        domain_name=domain_name,
        task_name=task_name,
        observation_shape_list=observation_shape_list,
        action_space_size_list=action_space_size_list,
        frame_skip=frame_skip,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
    ))

    # Update model settings
    config.policy.model.update(dict(
        observation_shape_list=observation_shape_list,
        action_space_size_list=action_space_size_list,
    ))
    config.policy.model.world_model_cfg.update(dict(
        observation_shape_list=observation_shape_list,
        action_space_size_list=action_space_size_list,
        num_unroll_steps=num_unroll_steps,
        norm_type=norm_type,
    ))
    
    # Update policy settings
    config.policy.update(dict(
        target_return=target_return_dict.get(env_id),
        total_batch_size=total_batch_size,
        num_unroll_steps=num_unroll_steps,
        replay_ratio=reanalyze_ratio,
        batch_size=batch_size,
        num_segments=num_segments,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        buffer_reanalyze_freq=buffer_reanalyze_freq,
        reanalyze_batch_size=reanalyze_batch_size,
        reanalyze_partition=reanalyze_partition,
    ))
    
    return config


def create_env_manager_config() -> EasyDict:
    """
    Overview:
        Creates the configuration for the environment manager and policy type.

    Returns:
        - (:obj:`EasyDict`): A dictionary with environment manager and policy import settings.
    """
    return EasyDict(dict(
        env=dict(
            type='dmc2gym_lightzero',
            import_names=['zoo.dmc2gym.envs.dmc2gym_lightzero_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(
            type='sampled_unizero_multitask',
            import_names=['lzero.policy.sampled_unizero_multitask'],
        ),
    ))


def generate_experiment_name(num_tasks: int, curriculum_stage_num: int, buffer_reanalyze_freq: float, seed: int) -> str:
    """
    Overview:
        Generates a descriptive name for the experiment.

    Arguments:
        - num_tasks (:obj:`int`): Number of tasks in the experiment.
        - curriculum_stage_num (:obj:`int`): Number of curriculum stages.
        - buffer_reanalyze_freq (:obj:`float`): Frequency of buffer reanalysis.
        - seed (:obj:`int`): The random seed for the experiment.

    Returns:
        - (:obj:`str`): The generated experiment name prefix.
    """
    # NOTE: This is a template for the experiment name.
    # Users should customize it to reflect their specific experiment settings.
    return (
        f'data_suz_dmc_mt_balance_20250625/dmc_{num_tasks}tasks_frameskip4-pen-fs8_balance-stage-total-{curriculum_stage_num}'
        f'_stage0-10k-5k_fix-lora-update-stablescale_moe8-uselora_nlayer4_not-share-head'
        f'_brf{buffer_reanalyze_freq}_seed{seed}/'
    )


def generate_all_task_configs(
        env_id_list: list[str],
        target_return_dict: dict[str, int],
        action_space_size_list: list[int],
        observation_shape_list: list[int],
        curriculum_stage_num: int,
        collector_env_num: int,
        n_episode: int,
        evaluator_env_num: int,
        num_simulations: int,
        reanalyze_ratio: float,
        batch_size: list[int],
        num_unroll_steps: int,
        infer_context_length: int,
        norm_type: str,
        seed: int,
        buffer_reanalyze_freq: float,
        reanalyze_batch_size: int,
        reanalyze_partition: float,
        num_segments: int,
        total_batch_size: int
) -> list[tuple[int, list[EasyDict, EasyDict]]]:
    """
    Overview:
        Generates a list of configurations, one for each task in the experiment.

    Arguments:
        - env_id_list (:obj:`list[str]`): A list of all environment IDs.
        - target_return_dict (:obj:`dict[str, int]`): Mapping from env_id to target return.
        - action_space_size_list (:obj:`list[int]`): List of action space sizes for all tasks.
        - observation_shape_list (:obj:`list[int]`): List of observation shapes for all tasks.
        - curriculum_stage_num (:obj:`int`): The number of curriculum stages.
        - (other args): Hyperparameters for the experiment. See `create_task_config` for details.

    Returns:
        - (:obj:`list`): A list where each element is `[task_id, [task_config, env_manager_config]]`.
    """
    configs = []
    exp_name_prefix = generate_experiment_name(
        num_tasks=len(env_id_list),
        curriculum_stage_num=curriculum_stage_num,
        buffer_reanalyze_freq=buffer_reanalyze_freq,
        seed=seed
    )

    base_config = get_base_config(
        env_id_list=env_id_list,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        num_unroll_steps=num_unroll_steps,
        infer_context_length=infer_context_length,
        curriculum_stage_num=curriculum_stage_num
    )

    for task_id, env_id in enumerate(env_id_list):
        task_specific_config = create_task_config(
            base_config=base_config.clone(),  # Use a clone to avoid modifying the base config
            env_id=env_id,
            action_space_size_list=action_space_size_list,
            observation_shape_list=observation_shape_list,
            target_return_dict=target_return_dict,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_episode=n_episode,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            batch_size=batch_size[task_id],
            num_unroll_steps=num_unroll_steps,
            infer_context_length=infer_context_length,
            norm_type=norm_type,
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
            num_segments=num_segments,
            total_batch_size=total_batch_size,
        )
        task_specific_config.policy.task_id = task_id
        task_specific_config.exp_name = exp_name_prefix + f"{env_id}_seed{seed}"
        
        env_manager_cfg = create_env_manager_config()
        configs.append([task_id, [task_specific_config, env_manager_cfg]])
        
    return configs


def main():
    """
    Overview:
        Main function to set up and launch the multi-task UniZero training experiment.
        This script should be executed with <nproc_per_node> GPUs.

        Example launch commands:
        1. Using `torch.distributed.launch`:
           cd <PATH_TO_YOUR_PROJECT>/LightZero/
           python -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 \\
               ./zoo/dmc2gym/config/dmc2gym_state_suz_multitask_ddp_balance_config.py 2>&1 | tee \\
               ./logs/uz_mt_dmc18_balance_moe8_seed0.log

        2. Using `torchrun`:
           cd <PATH_TO_YOUR_PROJECT>/LightZero/
           torchrun --nproc_per_node=8 ./zoo/dmc2gym/config/dmc2gym_state_suz_multitask_ddp_balance_config.py
    """
    from lzero.entry import train_unizero_multitask_balance_segment_ddp
    from ding.utils import DDPContext
    import torch.distributed as dist
    from zoo.dmc2gym.config.dmc_state_env_space_map import dmc_state_env_action_space_map, dmc_state_env_obs_space_map

    # ==============================================================
    # Experiment-level settings
    # ==============================================================
    # NOTE: You can switch between different sets of environments by uncommenting them.
    # DMC 8-task benchmark
    # env_id_list = [
    #     'acrobot-swingup', 'cartpole-balance', 'cartpole-balance_sparse',
    #     'cartpole-swingup', 'cartpole-swingup_sparse', 'cheetah-run',
    #     "ball_in_cup-catch", "finger-spin",
    # ]
    # target_return_dict = {
    #     'acrobot-swingup': 500, 'cartpole-balance': 950, 'cartpole-balance_sparse': 950,
    #     'cartpole-swingup': 800, 'cartpole-swingup_sparse': 750, 'cheetah-run': 650,
    #     "ball_in_cup-catch": 950, "finger-spin": 800,
    # }

    # DMC 18-task benchmark
    env_id_list = [
        'acrobot-swingup', 'cartpole-balance', 'cartpole-balance_sparse', 'cartpole-swingup',
        'cartpole-swingup_sparse', 'cheetah-run', "ball_in_cup-catch", "finger-spin",
        "finger-turn_easy", "finger-turn_hard", 'hopper-hop', 'hopper-stand',
        'pendulum-swingup', 'reacher-easy', 'reacher-hard', 'walker-run',
        'walker-stand', 'walker-walk',
    ]
    target_return_dict = {
        'acrobot-swingup': 500, 'cartpole-balance': 900, 'cartpole-balance_sparse': 950,
        'cartpole-swingup': 750, 'cartpole-swingup_sparse': 750, 'cheetah-run': 550,
        "ball_in_cup-catch": 950, "finger-spin": 800, "finger-turn_easy": 950,
        "finger-turn_hard": 950, 'hopper-hop': 150, 'hopper-stand': 600,
        'pendulum-swingup': 800, 'reacher-easy': 900, 'reacher-hard': 900,
        'walker-run': 500, 'walker-stand': 900, 'walker-walk': 900,
    }

    # ==============================================================
    # Hyperparameters
    # ==============================================================
    # NOTE: For debugging, you can use smaller values.
    # collector_env_num, num_segments, n_episode = 2, 2, 2
    # evaluator_env_num, num_simulations, total_batch_size = 2, 1, 8
    # batch_size = [3] * len(env_id_list)
    # max_env_step = int(1e3)

    # Production settings
    curriculum_stage_num = 5
    collector_env_num = 8
    num_segments = 8
    n_episode = 8
    evaluator_env_num = 3
    num_simulations = 50
    max_env_step = int(4e5)
    reanalyze_ratio = 0.0
    total_batch_size = 512
    batch_size = [int(min(64, total_batch_size / len(env_id_list)))] * len(env_id_list)
    num_unroll_steps = 5
    infer_context_length = 2
    norm_type = 'LN'
    buffer_reanalyze_freq = 1 / 100000
    reanalyze_batch_size = 160
    reanalyze_partition = 0.75
    seed = 0  # You can iterate over multiple seeds if needed

    # Fetch observation and action space info from predefined maps
    action_space_size_list = [dmc_state_env_action_space_map[env_id] for env_id in env_id_list]
    observation_shape_list = [dmc_state_env_obs_space_map[env_id] for env_id in env_id_list]
    
    # ==============================================================
    # Generate configurations and start training
    # ==============================================================
    configs = generate_all_task_configs(
        env_id_list=env_id_list,
        target_return_dict=target_return_dict,
        action_space_size_list=action_space_size_list,
        observation_shape_list=observation_shape_list,
        curriculum_stage_num=curriculum_stage_num,
        collector_env_num=collector_env_num,
        n_episode=n_episode,
        evaluator_env_num=evaluator_env_num,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        batch_size=batch_size,
        num_unroll_steps=num_unroll_steps,
        infer_context_length=infer_context_length,
        norm_type=norm_type,
        seed=seed,
        buffer_reanalyze_freq=buffer_reanalyze_freq,
        reanalyze_batch_size=reanalyze_batch_size,
        reanalyze_partition=reanalyze_partition,
        num_segments=num_segments,
        total_batch_size=total_batch_size,
    )

    with DDPContext():
        # To train only a subset of tasks for debugging, you can slice the configs list.
        # e.g., train_unizero_multitask_balance_segment_ddp(configs[:1], ...)
        train_unizero_multitask_balance_segment_ddp(configs, seed=seed, max_env_step=max_env_step, benchmark_name="dmc")
        dist.destroy_process_group()


if __name__ == "__main__":
    main()