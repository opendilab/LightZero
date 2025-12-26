# -*- coding: utf-8 -*-
"""
Overview:
    This script contains the configuration generation logic for a multi-task UniZero agent
    designed for Atari environments. It sets up experiment parameters, computes batch sizes
    for distributed training, and generates the final configuration objects required to
    launch the training process.

Execution Command Example:
    To run this script using distributed training with <N> GPUs, use the following command.
    Replace <N> with the number of GPUs per node (e.g., 8) and adjust paths and log files as needed.
    
    cd /path/to/your/project/LightZero
    python -m torch.distributed.launch --nproc_per_node=<N> --master_port=<PORT> \
    /path/to/this/script.py 2>&1 | tee /path/to/your/logs/training.log
"""
import math
from typing import List, Tuple, Dict, Any

from easydict import EasyDict
from ding.utils import DDPContext
# It is recommended to place entry point imports within the main execution block
# to avoid circular dependencies or premature initializations.
# from lzero.entry import train_unizero_multitask_balance_segment_ddp


# ==============================================================
#           Configuration Computation and Generation
# ==============================================================

def compute_batch_config(
        env_id_list: List[str],
        effective_batch_size: int,
        gpus_per_node: int = 8,
        max_micro_batch_per_gpu: int = 400
) -> Tuple[List[int], int]:
    """
    Overview:
        Computes the micro-batch size for each environment and the number of gradient accumulation steps.
        This is designed to balance the load across multiple environments and GPUs while respecting
        memory constraints (max_micro_batch_per_gpu).

    Arguments:
        - env_id_list (:obj:`List[str]`): A list of environment IDs.
        - effective_batch_size (:obj:`int`): The target total batch size after gradient accumulation.
        - gpus_per_node (:obj:`int`): The number of GPUs available for training. Defaults to 8.
        - max_micro_batch_per_gpu (:obj:`int`): The maximum micro-batch size that can fit on a single GPU. Defaults to 400.

    Returns:
        - (:obj:`Tuple[List[int], int]`): A tuple containing:
            - A list of micro-batch sizes, one for each environment.
            - The number of gradient accumulation steps required.
    """
    num_envs = len(env_id_list)
    if num_envs == 0:
        return [], 1

    # To avoid division by zero, assume at least one environment is processed per GPU group.
    envs_per_gpu_group = max(1, num_envs // gpus_per_node)

    # Calculate the maximum micro-batch size per environment based on GPU memory limits.
    max_micro_batch_per_env = int(max_micro_batch_per_gpu / envs_per_gpu_group)

    # Calculate the theoretical batch size per environment if distributed evenly.
    theoretical_env_batch = effective_batch_size / num_envs

    if theoretical_env_batch > max_micro_batch_per_env:
        # If the theoretical batch size exceeds the per-environment limit,
        # cap the micro-batch size at the maximum allowed value.
        micro_batch_size = max_micro_batch_per_env
        # Calculate gradient accumulation steps needed to reach the effective batch size.
        grad_accumulate_steps = math.ceil(theoretical_env_batch / max_micro_batch_per_env)
    else:
        # If the theoretical batch size is within limits, use it directly.
        micro_batch_size = int(theoretical_env_batch)
        grad_accumulate_steps = 1

    # Assign the same computed micro-batch size to all environments.
    batch_sizes = [micro_batch_size] * num_envs

    # Logging for debugging purposes.
    print(f"Number of environments: {num_envs}")
    print(f"Effective total batch size: {effective_batch_size}")
    print(f"Theoretical batch size per environment: {theoretical_env_batch:.2f}")
    print(f"Micro-batch size per environment: {micro_batch_size}")
    print(f"Gradient accumulation steps: {grad_accumulate_steps}")

    return batch_sizes, grad_accumulate_steps


def create_config(
        env_id: str,
        action_space_size: int,
        collector_env_num: int,
        evaluator_env_num: int,
        n_episode: int,
        num_simulations: int,
        reanalyze_ratio: float,
        batch_size: int,
        num_unroll_steps: int,
        infer_context_length: int,
        norm_type: str,
        buffer_reanalyze_freq: float,
        reanalyze_batch_size: int,
        reanalyze_partition: float,
        num_segments: int,
        total_batch_size: int,
        target_return: int,
        curriculum_stage_num: int,
        num_envs: int,
) -> EasyDict:
    """
    Overview:
        Creates the main configuration dictionary for a single UniZero task.

    Arguments:
        - env_id (:obj:`str`): The ID of the environment (e.g., 'PongNoFrameskip-v4').
        - action_space_size (:obj:`int`): The size of the action space.
        - collector_env_num (:obj:`int`): Number of environments for data collection.
        - evaluator_env_num (:obj:`int`): Number of environments for evaluation.
        - n_episode (:obj:`int`): Number of episodes to run for collection.
        - num_simulations (:obj:`int`): Number of simulations for MCTS.
        - reanalyze_ratio (:obj:`float`): The ratio of reanalyzed data in a batch.
        - batch_size (:obj:`int`): The micro-batch size for training.
        - num_unroll_steps (:obj:`int`): The number of steps to unroll the model dynamics.
        - infer_context_length (:obj:`int`): The context length for inference.
        - norm_type (:obj:`str`): The type of normalization layer to use (e.g., 'LN').
        - buffer_reanalyze_freq (:obj:`float`): Frequency of reanalyzing the replay buffer.
        - reanalyze_batch_size (:obj:`int`): Batch size for reanalysis.
        - reanalyze_partition (:obj:`float`): Partition ratio for reanalysis.
        - num_segments (:obj:`int`): Number of segments for game episodes.
        - total_batch_size (:obj:`int`): The effective total batch size.
        - target_return (:obj:`int`): The target return for the environment.
        - curriculum_stage_num (:obj:`int`): The number of stages in curriculum learning.
        - num_envs (:obj:`int`): The total number of environments in the multi-task setup.

    Returns:
        - (:obj:`EasyDict`): A configuration object for the agent.
    """
    return EasyDict(dict(
        env=dict(
            stop_value=int(1e6),
            env_id=env_id,
            observation_shape=(3, 64, 64),
            gray_scale=False,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False),
            full_action_space=True,
            collect_max_episode_steps=int(5e3),
            eval_max_episode_steps=int(5e3),
        ),
        policy=dict(
            multi_gpu=True,  # Crucial for DDP
            model=dict(
                observation_shape=(3, 64, 64),
                action_space_size=action_space_size,
                norm_type=norm_type,
                num_res_blocks=2,
                num_channels=256,
                continuous_action_space=False,
                world_model_cfg=dict(
                    use_global_pooling=False,
                    final_norm_option_in_obs_head='LayerNorm',
                    final_norm_option_in_encoder='LayerNorm',
                    predict_latent_loss_type='mse',
                    share_head=False,
                    analysis_dormant_ratio_weight_rank=False,
                    dormant_threshold=0.025,
                    continuous_action_space=False,
                    task_embed_option=None,
                    use_task_embed=False,
                    use_shared_projection=False,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    device='cuda',
                    action_space_size=action_space_size,
                    num_layers=4,
                    num_heads=24,
                    embed_dim=768,
                    obs_type='image',
                    env_num=num_envs,
                    task_num=num_envs,
                    encoder_type='vit',
                    use_normal_head=True,
                    use_softmoe_head=False,
                    use_moe_head=False,
                    num_experts_in_moe_head=4,
                    moe_in_transformer=False,
                    multiplication_moe_in_transformer=True,
                    n_shared_experts=1,
                    num_experts_per_tok=1,
                    num_experts_of_moe_in_transformer=8,
                    moe_use_lora=True,
                    curriculum_stage_num=curriculum_stage_num,
                    lora_target_modules=["attn", "feed_forward"],
                    lora_r=64,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    lora_scale_init=1,
                    min_stage0_iters=50000,
                    max_stage_iters=20000,
                    apply_curriculum_to_encoder=False,
                ),
            ),
            # --- Task and Learning Settings ---
            total_task_num=num_envs,
            task_num=num_envs,
            task_id=0,  # This will be overridden for each task.
            target_return=target_return,
            use_task_exploitation_weight=False,
            task_complexity_weight=True,
            balance_pipeline=True,
            # --- Training Settings ---
            cuda=True,
            total_batch_size=total_batch_size,
            allocated_batch_sizes=False,
            batch_size=batch_size,
            num_unroll_steps=num_unroll_steps,
            update_per_collect=80,
            replay_ratio=0.25,
            optim_type='AdamW',
            cos_lr_scheduler=False,
            train_start_after_envsteps=int(0),
            # --- Replay Buffer and Reanalysis ---
            replay_buffer_size=int(5e5),
            num_segments=num_segments,
            use_priority=False,
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
            reanalyze_ratio=reanalyze_ratio,
            # --- MCTS Settings ---
            num_simulations=num_simulations,
            collect_num_simulations=num_simulations,
            eval_num_simulations=50,
            # --- Collector and Evaluator Settings ---
            n_episode=n_episode,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            eval_freq=int(1e4),
            # --- Miscellaneous ---
            print_task_priority_logs=False,
            model_path=None,
            game_segment_length=20,
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=200000))),
        ),
    ))


def _generate_experiment_name(
        base_path_prefix: str,
        num_envs: int,
        curriculum_stage_num: int,
        buffer_reanalyze_freq: float,
        seed: int,
        env_id: str
) -> str:
    """
    Overview:
        Helper function to generate a standardized experiment name.

    Arguments:
        - base_path_prefix (:obj:`str`): The prefix for the experiment path, e.g., 'data_unizero_atari_mt_balance_YYYYMMDD'.
        - num_envs (:obj:`int`): The total number of environments.
        - curriculum_stage_num (:obj:`int`): The number of curriculum stages.
        - buffer_reanalyze_freq (:obj:`float`): The buffer reanalyze frequency.
        - seed (:obj:`int`): The random seed for the experiment.
        - env_id (:obj:`str`): The environment ID for this specific task.

    Returns:
        - (:obj:`str`): The generated experiment name.
    """
    # Template for the experiment's parent directory.
    brf_str = str(buffer_reanalyze_freq).replace('.', '')
    parent_dir = (
        f"{base_path_prefix}/atari_{num_envs}games_balance-total-stage{curriculum_stage_num}_"
        f"stage-50k-20k_vit-small-ln_trans-nlayer4-moe8_backbone-attn-mlp-lora_no-lora-scale_"
        f"brf{brf_str}_not-share-head_seed{seed}/"
    )

    # Clean the environment ID for the final part of the name.
    env_name_part = env_id.split('NoFrameskip')[0]

    return f"{parent_dir}{env_name_part}_seed{seed}"


def generate_configs(
        env_id_list: List[str],
        action_space_size: int,
        collector_env_num: int,
        n_episode: int,
        evaluator_env_num: int,
        num_simulations: int,
        reanalyze_ratio: float,
        batch_sizes: List[int],
        num_unroll_steps: int,
        infer_context_length: int,
        norm_type: str,
        seed: int,
        buffer_reanalyze_freq: float,
        reanalyze_batch_size: int,
        reanalyze_partition: float,
        num_segments: int,
        total_batch_size: int,
        target_return_dict: Dict[str, int],
        curriculum_stage_num: int,
) -> List[Tuple[int, List[Any]]]:
    """
    Overview:
        Generates a list of configuration tuples, one for each task/environment.

    Returns:
        - (:obj:`List[Tuple[int, List[Any]]]`): A list where each element is a tuple containing
          the task_id and a list with the main config and the environment manager config.
    """
    configs = []
    exp_name_base_prefix = 'data_unizero_mt_balance_atari'  # YYYYMMDD format

    for task_id, env_id in enumerate(env_id_list):
        config = create_config(
            env_id=env_id,
            action_space_size=action_space_size,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_episode=n_episode,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            batch_size=batch_sizes[task_id],
            num_unroll_steps=num_unroll_steps,
            infer_context_length=infer_context_length,
            norm_type=norm_type,
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
            num_segments=num_segments,
            total_batch_size=total_batch_size,
            target_return=target_return_dict[env_id],
            curriculum_stage_num=curriculum_stage_num,
            num_envs=len(env_id_list),
        )
        config.policy.task_id = task_id
        config.exp_name = _generate_experiment_name(
            base_path_prefix=exp_name_base_prefix,
            num_envs=len(env_id_list),
            curriculum_stage_num=curriculum_stage_num,
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            seed=seed,
            env_id=env_id
        )
        configs.append([task_id, [config, create_env_manager()]])
    return configs


def create_env_manager() -> EasyDict:
    """
    Overview:
        Creates the environment manager configuration, specifying the types of environment,
        policy, and manager to be used.

    Returns:
        - (:obj:`EasyDict`): A configuration object for the environment manager.
    """
    return EasyDict(dict(
        env=dict(
            type='atari_lightzero',
            import_names=['zoo.atari.envs.atari_lightzero_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(
            type='unizero_multitask',
            import_names=['lzero.policy.unizero_multitask'],
        ),
    ))


def get_atari_target_return_dict(ratio: float = 1.0) -> Dict[str, int]:
    """
    Overview:
        Calculates the target return for each Atari game based on a predefined score
        and a scaling ratio.

    Arguments:
        - ratio (:obj:`float`): A scaling factor for the target returns. Defaults to 1.0.

    Returns:
        - (:obj:`Dict[str, int]`): A dictionary mapping environment IDs to their calculated target returns.
    """
    # Pre-defined target scores for various Atari games.
    target_scores = {
        'PongNoFrameskip-v4': 20,
        'MsPacmanNoFrameskip-v4': 6951.6,
        'SeaquestNoFrameskip-v4': 42054.7,
        'BoxingNoFrameskip-v4': 12.1,
        'AlienNoFrameskip-v4': 7127.7,
        'ChopperCommandNoFrameskip-v4': 7387.8,
        'HeroNoFrameskip-v4': 30826.4,
        'RoadRunnerNoFrameskip-v4': 7845.0,
        'AmidarNoFrameskip-v4': 100.5,
        'AssaultNoFrameskip-v4': 742.0,
        'AsterixNoFrameskip-v4': 1503.3,
        'BankHeistNoFrameskip-v4': 753.1,
        'BattleZoneNoFrameskip-v4': 12187.5,
        'CrazyClimberNoFrameskip-v4': 15829.4,
        'DemonAttackNoFrameskip-v4': 1971.0,
        'FreewayNoFrameskip-v4': 29.6,
        'FrostbiteNoFrameskip-v4': 334.7,
        'GopherNoFrameskip-v4': 2412.5,
        'JamesbondNoFrameskip-v4': 302.8,
        'KangarooNoFrameskip-v4': 3035.0,
        'KrullNoFrameskip-v4': 2665.5,
        'KungFuMasterNoFrameskip-v4': 12736.3,
        'PrivateEyeNoFrameskip-v4': 1001.3,
        'UpNDownNoFrameskip-v4': 11693.2,
        'QbertNoFrameskip-v4': 13455.0,
        'BreakoutNoFrameskip-v4': 30.5,
    }
    return {env: int(round(score * ratio)) for env, score in target_scores.items()}


def get_env_id_list(num_games: int) -> List[str]:
    """
    Overview:
        Returns a list of Atari environment IDs based on the specified number of games.

    Arguments:
        - num_games (:obj:`int`): The number of games to include (e.g., 8 or 26).

    Returns:
        - (:obj:`List[str]`): A list of environment ID strings.
    """
    games_8 = [
        'PongNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SeaquestNoFrameskip-v4', 'BoxingNoFrameskip-v4',
        'AlienNoFrameskip-v4', 'ChopperCommandNoFrameskip-v4', 'HeroNoFrameskip-v4', 'RoadRunnerNoFrameskip-v4',
    ]
    games_26 = games_8 + [
        'AmidarNoFrameskip-v4', 'AssaultNoFrameskip-v4', 'AsterixNoFrameskip-v4', 'BankHeistNoFrameskip-v4',
        'BattleZoneNoFrameskip-v4', 'CrazyClimberNoFrameskip-v4', 'DemonAttackNoFrameskip-v4',
        'FreewayNoFrameskip-v4',
        'FrostbiteNoFrameskip-v4', 'GopherNoFrameskip-v4', 'JamesbondNoFrameskip-v4', 'KangarooNoFrameskip-v4',
        'KrullNoFrameskip-v4', 'KungFuMasterNoFrameskip-v4', 'PrivateEyeNoFrameskip-v4', 'UpNDownNoFrameskip-v4',
        'QbertNoFrameskip-v4', 'BreakoutNoFrameskip-v4',
    ]
    if num_games == 3:
        return ['PongNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SeaquestNoFrameskip-v4']
    elif num_games == 8:
        return games_8
    elif num_games == 26:
        return games_26
    else:
        raise ValueError(f"Unsupported number of games: {num_games}. Supported values are 3, 8, 26.")


def main():
    """
    Overview:
        Main function to configure and launch the multi-task training process.
    """
    # ==============================================================
    #           Primary Hyperparameters
    # ==============================================================
    # --- Experiment ---
    num_games = 8  # Options: 3, 8, 26
    seeds = [0]
    max_env_step = int(4e5)
    benchmark_name = "atari"

    # --- Curriculum ---
    curriculum_stage_num = 5

    # --- Environment and Agent ---
    action_space_size = 18
    num_simulations = 50
    num_unroll_steps = 10
    infer_context_length = 4
    norm_type = 'LN'

    # --- Collector and Evaluator ---
    collector_env_num = 8
    evaluator_env_num = 3
    n_episode = 8
    num_segments = 8

    # --- Reanalysis ---
    reanalyze_ratio = 0.0
    buffer_reanalyze_freq = 1 / 50
    reanalyze_batch_size = 160
    reanalyze_partition = 0.75

    # ==============================================================
    #           Derived Configurations
    # ==============================================================
    env_id_list = get_env_id_list(num_games)
    target_return_dict = get_atari_target_return_dict(ratio=1.0)

    # --- Batch Size Calculation ---
    if num_games == 8:
        effective_batch_size = 512
    elif num_games == 26:
        effective_batch_size = 512  # For ViT-Base encoder
    else:
        # Default or other cases
        effective_batch_size = 512

    batch_sizes, grad_acc_steps = compute_batch_config(env_id_list, effective_batch_size)
    # Note: `total_batch_size` is passed to the config but `effective_batch_size` is used for calculation.
    # This maintains consistency with the original script's logic.
    total_batch_size = effective_batch_size

    # ==============================================================
    #           Launch Training
    # ==============================================================
    from lzero.entry import train_unizero_multitask_balance_segment_ddp

    for seed in seeds:
        configs = generate_configs(
            env_id_list=env_id_list,
            action_space_size=action_space_size,
            collector_env_num=collector_env_num,
            n_episode=n_episode,
            evaluator_env_num=evaluator_env_num,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            batch_sizes=batch_sizes,
            num_unroll_steps=num_unroll_steps,
            infer_context_length=infer_context_length,
            norm_type=norm_type,
            seed=seed,
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
            num_segments=num_segments,
            total_batch_size=total_batch_size,
            target_return_dict=target_return_dict,
            curriculum_stage_num=curriculum_stage_num
        )

        with DDPContext():
            train_unizero_multitask_balance_segment_ddp(
                configs,
                seed=seed,
                max_env_step=max_env_step,
                benchmark_name=benchmark_name
            )


if __name__ == "__main__":
    main()