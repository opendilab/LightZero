from easydict import EasyDict
from typing import List, Any, Dict, Tuple

import logging

# Set up logging configuration
# Configure logging to output to both a file and the console.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("output.log", encoding="utf-8"),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)


def create_config(
        env_id: str,
        env_id_list: List[str],
        target_return_dict: Dict[str, int],
        observation_shape_list: List[Tuple[int, ...]],
        action_space_size_list: List[int],
        collector_env_num: int,
        evaluator_env_num: int,
        n_episode: int,
        num_simulations: int,
        reanalyze_ratio: float,
        batch_size: List[int],
        num_unroll_steps: int,
        infer_context_length: int,
        norm_type: str,
        buffer_reanalyze_freq: float,
        reanalyze_batch_size: int,
        reanalyze_partition: float,
        num_segments: int,
        total_batch_size: int,
) -> EasyDict:
    """
    Overview:
        Create a configuration EasyDict for a single reinforcement learning task.

    Arguments:
        - env_id (:obj:`str`): The ID of the environment, e.g., 'cartpole-swingup'.
        - env_id_list (:obj:`List[str]`): A list of all environment IDs for the multi-task setup.
        - target_return_dict (:obj:`Dict[str, int]`): A dictionary mapping environment IDs to their target return values.
        - observation_shape_list (:obj:`List[Tuple[int, ...]]`): List of observation shapes for all tasks.
        - action_space_size_list (:obj:`List[int]`): List of action space sizes for all tasks.
        - collector_env_num (:obj:`int`): Number of environments for data collection.
        - evaluator_env_num (:obj:`int`): Number of environments for evaluation.
        - n_episode (:obj:`int`): Number of episodes to run for collection.
        - num_simulations (:obj:`int`): Number of simulations in the MCTS search.
        - reanalyze_ratio (:obj:`float`): The ratio of reanalyzed data in a batch.
        - batch_size (:obj:`List[int]`): Batch size for training per task.
        - num_unroll_steps (:obj:`int`): Number of steps to unroll the model during training.
        - infer_context_length (:obj:`int`): The context length for inference.
        - norm_type (:obj:`str`): The type of normalization to use (e.g., 'LN').
        - buffer_reanalyze_freq (:obj:`float`): Frequency of reanalyzing the buffer.
        - reanalyze_batch_size (:obj:`int`): Batch size for reanalyzing.
        - reanalyze_partition (:obj:`float`): Partition ratio for reanalyzing.
        - num_segments (:obj:`int`): Number of segments for the replay buffer.
        - total_batch_size (:obj:`int`): The total batch size across all tasks.

    Returns:
        - (:obj:`EasyDict`): A configuration object for the specified task.
    """
    domain_name, task_name = env_id.split('-')

    # Specific frame_skip settings for certain domains.
    if domain_name == "pendulum":
        frame_skip = 8
    else:
        frame_skip = 4

    # --- Environment Configuration ---
    env_cfg = dict(
        stop_value=int(5e5),
        env_id=env_id,
        domain_name=domain_name,
        task_name=task_name,
        observation_shape_list=observation_shape_list,
        action_space_size_list=action_space_size_list,
        from_pixels=False,
        frame_skip=frame_skip,
        continuous=True,  # Assuming all DMC tasks use continuous action spaces
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False),
        game_segment_length=100,
        # TODO: Settings for debugging purposes.
        # game_segment_length=10,
        # collect_max_episode_steps=int(40),
        # eval_max_episode_steps=int(40),
    )

    # --- World Model Configuration ---
    world_model_cfg = dict(
        # --- Normalization and Loss ---
        final_norm_option_in_obs_head='LayerNorm',
        final_norm_option_in_encoder='LayerNorm',
        predict_latent_loss_type='mse',  # TODO: for latent state layer_norm
        # final_norm_option_in_obs_head='SimNorm',
        # final_norm_option_in_encoder='SimNorm',
        # predict_latent_loss_type='group_kl', # TODO: only for latent state sim_norm

        # --- Architecture ---
        share_head=False,  # TODO
        use_shared_projection=False,
        obs_type='vector',
        model_type='mlp',
        continuous_action_space=True,
        num_of_sampled_actions=20,
        sigma_type='conditioned',
        fixed_sigma_value=0.5,
        bound_type=None,
        norm_type=norm_type,
        device='cuda',

        # --- Transformer/MOE Settings ---
        num_layers=8,  # TODO: 8 for standard, 1 for debug
        num_heads=24,
        embed_dim=768,
        moe_in_transformer=False,
        multiplication_moe_in_transformer=True,
        num_experts_of_moe_in_transformer=8,
        n_shared_experts=1,
        num_experts_per_tok=1,
        use_normal_head=True,
        use_softmoe_head=False,
        use_moe_head=False,
        num_experts_in_moe_head=4,

        # --- LoRA Parameters ---
        moe_use_lora=False,  # TODO
        curriculum_stage_num=3,
        lora_target_modules=["attn", "feed_forward"],
        lora_r=0,
        lora_alpha=1,
        lora_dropout=0.0,

        # --- Multi-task Settings ---
        task_embed_option=None,  # TODO: 'concat_task_embed' or None
        use_task_embed=False,  # TODO
        # task_embed_dim=128,
        task_num=len(env_id_list),

        # --- Analysis ---
        analysis_dormant_ratio_weight_rank=False,  # TODO
        analysis_dormant_ratio_interval=5000,

        # --- Dynamic Properties ---
        observation_shape_list=observation_shape_list,
        action_space_size_list=action_space_size_list,
        num_unroll_steps=num_unroll_steps,
        max_blocks=num_unroll_steps,
        max_tokens=2 * num_unroll_steps,  # Each timestep has 2 tokens: obs and action
        context_length=2 * infer_context_length,
        env_num=max(collector_env_num, evaluator_env_num),

        # --- Loss Weights ---
        policy_loss_type='kl',
        policy_entropy_weight=5e-2,
    )

    # --- Policy Configuration ---
    policy_cfg = dict(
        # --- Hardware & Distribution ---
        multi_gpu=True,  # TODO: enable multi-GPU for DDP
        cuda=True,

        # --- Model ---
        model=dict(
            observation_shape_list=observation_shape_list,
            action_space_size_list=action_space_size_list,
            continuous_action_space=True,
            num_of_sampled_actions=20,
            model_type='mlp',
            world_model_cfg=world_model_cfg,
        ),

        # --- Learning ---
        learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000000))),
        optim_type='AdamW',
        learning_rate=1e-4,
        grad_clip_value=5,
        cos_lr_scheduler=True,
        piecewise_decay_lr_scheduler=False,

        # --- Training Loop ---
        train_start_after_envsteps=int(0),  # TODO: 2e3 for standard, 0 for quick debug
        update_per_collect=200,
        replay_ratio=reanalyze_ratio,

        # --- Batch Sizes ---
        batch_size=batch_size,
        total_batch_size=total_batch_size,
        allocated_batch_sizes=False,

        # --- Replay Buffer ---
        replay_buffer_size=int(1e6),
        num_segments=num_segments,
        use_priority=False,

        # --- Reanalyze ---
        reanalyze_ratio=reanalyze_ratio,
        buffer_reanalyze_freq=buffer_reanalyze_freq,
        reanalyze_batch_size=reanalyze_batch_size,
        reanalyze_partition=reanalyze_partition,

        # --- Algorithm Hyperparameters ---
        num_simulations=num_simulations,
        num_unroll_steps=num_unroll_steps,
        td_steps=5,
        discount_factor=0.99,
        manual_temperature_decay=True,
        threshold_training_steps_for_final_temperature=int(2.5e4),

        # --- MoCo (Momentum Contrast) ---
        use_moco=False,  # TODO
        only_use_moco_stats=False,
        grad_correct_params=dict(
            MoCo_beta=0.5, MoCo_beta_sigma=0.5, MoCo_gamma=0.1, MoCo_gamma_sigma=0.5, MoCo_rho=0,
            calpha=0.5, rescale=1,
        ),

        # --- Multi-task Specific ---
        total_task_num=len(env_id_list),
        task_num=len(env_id_list),
        task_id=0,  # To be set per task
        target_return=target_return_dict.get(env_id),
        use_task_exploitation_weight=False,  # TODO
        task_complexity_weight=True,  # TODO
        balance_pipeline=True,
        print_task_priority_logs=False,

        # --- Environment Interaction ---
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_episode=n_episode,
        eval_freq=int(4e3),

        # --- Checkpointing ---
        model_path=None,
    )

    # --- Combine configurations into the final EasyDict object ---
    main_config = EasyDict(dict(
        env=env_cfg,
        policy=policy_cfg,
    ))

    return main_config


def generate_configs(
        env_id_list: List[str],
        target_return_dict: Dict[str, int],
        collector_env_num: int,
        n_episode: int,
        evaluator_env_num: int,
        num_simulations: int,
        reanalyze_ratio: float,
        batch_size: List[int],
        num_unroll_steps: int,
        infer_context_length: int,
        norm_type: str,
        seed: int,
        buffer_reanalyze_freq: float,
        reanalyze_batch_size: int,
        reanalyze_partition: float,
        num_segments: int,
        total_batch_size: int,
        dmc_state_env_action_space_map: Dict[str, int],
        dmc_state_env_obs_space_map: Dict[str, Tuple[int, ...]],
) -> List[Tuple[int, List[Any]]]:
    """
    Overview:
        Generate a list of configurations for all specified multi-task environments.

    Arguments:
        - env_id_list (:obj:`List[str]`): A list of all environment IDs for the multi-task setup.
        - target_return_dict (:obj:`Dict[str, int]`): A dictionary mapping environment IDs to their target return values.
        - collector_env_num (:obj:`int`): Number of environments for data collection.
        - n_episode (:obj:`int`): Number of episodes to run for collection.
        - evaluator_env_num (:obj:`int`): Number of environments for evaluation.
        - num_simulations (:obj:`int`): Number of simulations in the MCTS search.
        - reanalyze_ratio (:obj:`float`): The ratio of reanalyzed data in a batch.
        - batch_size (:obj:`List[int]`): Batch size for training per task.
        - num_unroll_steps (:obj:`int`): Number of steps to unroll the model during training.
        - infer_context_length (:obj:`int`): The context length for inference.
        - norm_type (:obj:`str`): The type of normalization to use (e.g., 'LN').
        - seed (:obj:`int`): The random seed.
        - buffer_reanalyze_freq (:obj:`float`): Frequency of reanalyzing the buffer.
        - reanalyze_batch_size (:obj:`int`): Batch size for reanalyzing.
        - reanalyze_partition (:obj:`float`): Partition ratio for reanalyzing.
        - num_segments (:obj:`int`): Number of segments for the replay buffer.
        - total_batch_size (:obj:`int`): The total batch size across all tasks.
        - dmc_state_env_action_space_map (:obj:`Dict[str, int]`): Map from env_id to action space size.
        - dmc_state_env_obs_space_map (:obj:`Dict[str, Tuple[int, ...]]`): Map from env_id to observation shape.

    Returns:
        - (:obj:`List[Tuple[int, List[Any]]]`): A list where each element contains the task ID and its corresponding
          configuration objects.
    """
    configs = []

    # Define the experiment name prefix. This helps in organizing experiment logs and results.
    exp_name_prefix = (
        f'data_suz_dmc_mt_20250601/dmc_{len(env_id_list)}tasks_frameskip4-pendulum-skip8_ln-mse'
        f'_nlayer8_trans-moe8_brf{buffer_reanalyze_freq}_seed{seed}/'
    )

    # Get action_space_size and observation_shape for each environment.
    action_space_size_list = [dmc_state_env_action_space_map[env_id] for env_id in env_id_list]
    observation_shape_list = [dmc_state_env_obs_space_map[env_id] for env_id in env_id_list]

    for task_id, env_id in enumerate(env_id_list):
        config = create_config(
            env_id=env_id,
            env_id_list=env_id_list,
            target_return_dict=target_return_dict,
            action_space_size_list=action_space_size_list,
            observation_shape_list=observation_shape_list,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_episode=n_episode,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            batch_size=batch_size,
            num_unroll_steps=num_unroll_steps,
            infer_context_length=infer_context_length,
            norm_type=norm_type,
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
            num_segments=num_segments,
            total_batch_size=total_batch_size,
        )
        config.policy.task_id = task_id
        config.exp_name = exp_name_prefix + f"{env_id}_seed{seed}"
        configs.append([task_id, [config, create_env_manager()]])
    return configs


def create_env_manager() -> EasyDict:
    """
    Overview:
        Create the environment and policy manager configuration. This specifies the types
        of environment, policy, and their import paths.

    Returns:
        - (:obj:`EasyDict`): A configuration object for the environment and policy managers.
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


if __name__ == "__main__":
    """
    Overview:
        Main script to configure and launch a multi-task training session for DeepMind Control Suite (DMC)
        environments using Distributed Data Parallel (DDP).

    Usage:
        This script should be executed with <nproc_per_node> GPUs.
        Navigate to the project root directory and run the launch command.

        Example command:
        cd <PATH_TO_YOUR_PROJECT_ROOT>
        # Using torch.distributed.launch (deprecated)
        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 \\
            <PATH_TO_THIS_SCRIPT>/dmc2gym_state_suz_multitask_ddp_config.py 2>&1 | tee \\
            <PATH_TO_LOG_DIR>/uz_mt_dmc18_train.log

        # Using torchrun (recommended)
        torchrun --nproc_per_node=8 <PATH_TO_THIS_SCRIPT>/dmc2gym_state_suz_multitask_ddp_config.py
    """
    # --- Import necessary components for training ---
    # It's good practice to place imports inside the main guard
    # if they are only used for script execution.
    from lzero.entry import train_unizero_multitask_segment_ddp
    from ding.utils import DDPContext
    import torch.distributed as dist
    from zoo.dmc2gym.config.dmc_state_env_space_map import dmc_state_env_action_space_map, dmc_state_env_obs_space_map

    # --- Experiment constants ---
    BENCHMARK_NAME = 'dmc'

    # --- Environment and Task Definitions ---
    # Target return values for each DMC task, used for evaluation and potential curriculum.
    target_return_dict = {
        'acrobot-swingup': 500,
        'cartpole-balance': 950,
        'cartpole-balance_sparse': 950,
        'cartpole-swingup': 800,
        'cartpole-swingup_sparse': 750,
        'cheetah-run': 650,
        "ball_in_cup-catch": 950,
        "finger-spin": 800,
        "finger-turn_easy": 950,
        "finger-turn_hard": 950,
        'hopper-hop': 150,
        'hopper-stand': 600,
        'pendulum-swingup': 800,
        'reacher-easy': 950,
        'reacher-hard': 950,
        'walker-run': 600,
        'walker-stand': 950,
        'walker-walk': 950,
    }

    # List of DMC environments to be used in the multi-task setup.
    env_id_list = list(target_return_dict.keys())

    # --- Hyperparameters for the training session ---
    # Environment and Collector settings
    collector_env_num = 8
    evaluator_env_num = 3
    n_episode = 8
    max_env_step = int(4e5)

    # Replay Buffer and Reanalyze settings
    num_segments = 8
    reanalyze_ratio = 0.0
    buffer_reanalyze_freq = 1 / 100000
    reanalyze_batch_size = 160
    reanalyze_partition = 0.75

    # Model and Training settings
    total_batch_size = 512
    # Allocate batch size per task, ensuring a minimum of 64 or distributing the total size.
    batch_size = [int(min(64, total_batch_size / len(env_id_list))) for _ in range(len(env_id_list))]
    num_unroll_steps = 5
    infer_context_length = 2
    norm_type = 'LN'
    num_simulations = 50

    # --- Main training loop ---
    # Iterate over different random seeds for multiple runs.
    for seed in [1, 2]:
        # Generate the specific configurations for each task for the current run.
        configs = generate_configs(
            env_id_list=env_id_list,
            target_return_dict=target_return_dict,
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
            dmc_state_env_action_space_map=dmc_state_env_action_space_map,
            dmc_state_env_obs_space_map=dmc_state_env_obs_space_map,
        )

        with DDPContext():
            train_unizero_multitask_segment_ddp(configs, seed=seed, max_env_step=max_env_step,
                                                benchmark_name=BENCHMARK_NAME)
            # If you only want to train a subset of tasks, you can slice the configs list.
            # For example, to train only the first four tasks:
            # train_unizero_multitask_segment_ddp(configs[:4], seed=seed, max_env_step=max_env_step, benchmark_name=BENCHMARK_NAME)
            dist.destroy_process_group()