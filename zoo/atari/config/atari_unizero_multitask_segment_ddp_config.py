from easydict import EasyDict
import math
from typing import List, Tuple, Any, Dict, Union

# -------------------------------------------------
# 1. Refactored compute_batch_config
# -------------------------------------------------
def compute_batch_config(
    env_id_list: List[str],
    effective_batch_size: int,
    gpu_num: int = 8,
    max_micro_batch_one_gpu: int = 400,
) -> Tuple[List[int], int]:
    """
    Overview:
        Calculate the micro-batch size for each environment and the number of gradient accumulation steps
        to approach a target effective batch size across multiple GPUs and environments.

    Arguments:
        - env_id_list (:obj:`List[str]`): A list of environment IDs for all tasks.
        - effective_batch_size (:obj:`int`): The target global batch size for one backward pass.
        - gpu_num (:obj:`int`): The number of GPUs actually used. Defaults to 8.
        - max_micro_batch_one_gpu (:obj:`int`): The maximum micro-batch size a single GPU can handle. Defaults to 400.

    Returns:
        - batch_sizes (:obj:`List[int]`): A list of micro-batch sizes for each environment.
        - grad_acc_steps (:obj:`int`): The number of gradient accumulation steps.
    """
    n_env = len(env_id_list)
    # Number of environments that each GPU needs to handle simultaneously.
    envs_per_gpu = max(1, math.ceil(n_env / gpu_num))
    # Reduce the micro-batch limit if multiple environments share one GPU.
    max_micro_batch = max(1, max_micro_batch_one_gpu // envs_per_gpu)

    # First, calculate a candidate micro-batch by distributing the effective batch size evenly.
    candidate = max(1, effective_batch_size // n_env)
    micro_batch = min(candidate, max_micro_batch)

    # Gradient accumulation steps = ceil(global_batch / (micro_batch * n_env)).
    grad_acc_steps = max(1, math.ceil(effective_batch_size / (micro_batch * n_env)))

    # Fine-tune the micro-batch downwards to ensure:
    # micro_batch * n_env * grad_acc_steps <= effective_batch_size
    # This aims to get as close as possible to the target without exceeding it.
    while micro_batch * n_env * grad_acc_steps > effective_batch_size:
        micro_batch -= 1
        if micro_batch == 0:  # Defensive check, should not happen in theory.
            micro_batch = 1
            break

    batch_sizes = [micro_batch] * n_env

    # --- Debug Information --- #
    real_total_batch_size = micro_batch * n_env * grad_acc_steps
    print(
        f"[BatchConfig] Envs={n_env}, TargetTotalBS={effective_batch_size}, "
        f"MicroBS={micro_batch}, GradAccSteps={grad_acc_steps}, RealTotalBS={real_total_batch_size}"
    )

    return batch_sizes, grad_acc_steps

def create_config(
        env_id: str, action_space_size: int, collector_env_num: int, evaluator_env_num: int, n_episode: int,
        num_simulations: int, reanalyze_ratio: float, batch_size: int, num_unroll_steps: int,
        infer_context_length: int, norm_type: str, buffer_reanalyze_freq: float, reanalyze_batch_size: int,
        reanalyze_partition: float, num_segments: int, total_batch_size: int, num_layers: int
) -> EasyDict:
    """
    Overview:
        Creates the main configuration structure for a single training task.

    Arguments:
        - env_id (:obj:`str`): The environment ID.
        - action_space_size (:obj:`int`): The size of the action space.
        - collector_env_num (:obj:`int`): Number of environments for data collection.
        - evaluator_env_num (:obj:`int`): Number of environments for evaluation.
        - n_episode (:obj:`int`): Number of episodes to run for evaluation.
        - num_simulations (:obj:`int`): Number of simulations in MCTS.
        - reanalyze_ratio (:obj:`float`): The ratio of reanalyzed samples in a batch.
        - batch_size (:obj:`int`): The batch size for training.
        - num_unroll_steps (:obj:`int`): The number of steps to unroll the model dynamics.
        - infer_context_length (:obj:`int`): The context length for inference.
        - norm_type (:obj:`str`): The type of normalization layer to use (e.g., 'LN').
        - buffer_reanalyze_freq (:obj:`float`): Frequency of reanalyzing the replay buffer.
        - reanalyze_batch_size (:obj:`int`): Batch size for reanalysis.
        - reanalyze_partition (:obj:`float`): Partition ratio for reanalysis.
        - num_segments (:obj:`int`): Number of segments for data collection.
        - total_batch_size (:obj:`int`): The total effective batch size.
        - num_layers (:obj:`int`): Number of layers in the transformer model.

    Returns:
        - (:obj:`EasyDict`): A configuration object.
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
        ),
        policy=dict(
            multi_gpu=True,
            only_use_moco_stats=False,
            use_moco=False,
            moco_version="v1",
            total_task_num=len(env_id_list),
            task_num=len(env_id_list),
            task_id=0,  # This will be overridden for each task
            model=dict(
                observation_shape=(3, 64, 64),
                action_space_size=action_space_size,
                norm_type=norm_type,
                num_res_blocks=2,
                num_channels=256,
                num_layers=num_layers,
                world_model_cfg=dict(
                    norm_type=norm_type,
                    action_space_size=action_space_size,
                    num_layers=num_layers,
                    num_heads=8,
                    embed_dim=768,
                    env_num=len(env_id_list),
                    task_num=len(env_id_list),
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    final_norm_option_in_obs_head='LayerNorm',
                    final_norm_option_in_encoder='LayerNorm',
                    predict_latent_loss_type='mse',
                    encoder_type='vit',
                    device='cuda',
                    game_segment_length=20,
                ),
            ),
            device='cuda',
            game_segment_length=20,
            update_per_collect=80,  # Corresponds to replay_ratio=0.5 for 8 games (20*8*0.5=80)
            learning_rate=0.0001,
            weight_decay=1e-2,
            batch_size=batch_size,
            num_unroll_steps=num_unroll_steps,
            num_segments=num_segments,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            n_episode=n_episode,
            total_batch_size=total_batch_size,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
            replay_buffer_size=int(5e5),
            eval_freq=int(1e4),
        ),
    ))

def generate_configs(
        env_id_list: List[str], action_space_size: int, collector_env_num: int, n_episode: int,
        evaluator_env_num: int, num_simulations: int, reanalyze_ratio: float, batch_size: List[int],
        num_unroll_steps: int, infer_context_length: int, norm_type: str, seed: int,
        buffer_reanalyze_freq: float, reanalyze_batch_size: int, reanalyze_partition: float,
        num_segments: int, total_batch_size: int, num_layers: int
) -> List[List[Union[int, List[EasyDict]]]]:
    """
    Overview:
        Generates a list of configurations for all specified tasks.

    Arguments:
        (See arguments for `create_config` function)
        - seed (:obj:`int`): The random seed for the experiment.

    Returns:
        - (:obj:`List[List[Union[int, List[EasyDict]]]]`): A list where each element contains a task_id
          and its corresponding configuration objects.
    """
    configs = []

    # --- Experiment Name Template ---
    benchmark_tag = "data_unizero_mt"
    model_tag = f"vit_nlayer{num_layers}_tbs{total_batch_size}"
    exp_name_prefix = f'{benchmark_tag}/atari_{len(env_id_list)}games_{model_tag}_seed{seed}/'

    for task_id, env_id in enumerate(env_id_list):
        config = create_config(
            env_id, action_space_size, collector_env_num, evaluator_env_num, n_episode, num_simulations,
            reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type,
            buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition, num_segments, total_batch_size, num_layers
        )
        config.policy.task_id = task_id
        # Correctly extract the game name from 'ALE/GameName-v5' format.
        game_name = env_id.split('/')[1].split('-')[0]
        config.exp_name = exp_name_prefix + f"{game_name}_seed{seed}"
        configs.append([task_id, [config, create_env_manager()]])
    return configs

def create_env_manager() -> EasyDict:
    """
    Overview:
        Creates the environment manager configuration, specifying the types of environment,
        policy, and their import paths.

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

if __name__ == "__main__":
    """
    Overview:
        This script should be executed with <nproc_per_node> GPUs for distributed training.

        Example launch commands:

        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        cd /path/to/your/project/

        torchrun --nproc_per_node=4 /mnt/shared-storage-user/puyuan/code/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py
    """
    from lzero.entry import train_unizero_multitask_segment_ddp
    from ding.utils import DDPContext
    import torch.distributed as dist
    import os

    # ==================== Main Experiment Settings ====================
    num_games = 8  # Options: 3, 8, 26
    num_layers = 2
    action_space_size = 18
    collector_env_num = 8
    num_segments = 8
    n_episode = 8
    evaluator_env_num = 3
    num_simulations = 50
    max_env_step = int(5e6)
    reanalyze_ratio = 0.0

    # ==================== Environment Configuration ====================
    if num_games == 3:
        env_id_list = ['ALE/Pong-v5', 'ALE/MsPacman-v5', 'ALE/Seaquest-v5']
    elif num_games == 8:
        env_id_list = [
            'ALE/Pong-v5', 'ALE/MsPacman-v5', 'ALE/Seaquest-v5', 'ALE/Boxing-v5',
            'ALE/Alien-v5', 'ALE/ChopperCommand-v5', 'ALE/Hero-v5', 'ALE/RoadRunner-v5',
        ]
    elif num_games == 26:
        env_id_list = [
            'ALE/Pong-v5', 'ALE/MsPacman-v5', 'ALE/Seaquest-v5', 'ALE/Boxing-v5',
            'ALE/Alien-v5', 'ALE/ChopperCommand-v5', 'ALE/Hero-v5', 'ALE/RoadRunner-v5',
            'ALE/Amidar-v5', 'ALE/Assault-v5', 'ALE/Asterix-v5', 'ALE/BankHeist-v5',
            'ALE/BattleZone-v5', 'ALE/CrazyClimber-v5', 'ALE/DemonAttack-v5', 'ALE/Freeway-v5',
            'ALE/Frostbite-v5', 'ALE/Gopher-v5', 'ALE/Jamesbond-v5', 'ALE/Kangaroo-v5',
            'ALE/Krull-v5', 'ALE/KungFuMaster-v5', 'ALE/PrivateEye-v5', 'ALE/UpNDown-v5',
            'ALE/Qbert-v5', 'ALE/Breakout-v5',
        ]
    else:
        raise ValueError(f"Unsupported number of environments: {num_games}")

    # ==================== Batch Size Calculation ====================
    if len(env_id_list) == 8:
        if num_layers in [2, 4]:
            effective_batch_size = 1024
        elif num_layers == 8:
            effective_batch_size = 512
    elif len(env_id_list) == 26:
        effective_batch_size = 512
    elif len(env_id_list) == 3:
        effective_batch_size = 10  # For debugging
    else:
        raise ValueError(f"Batch size not configured for {len(env_id_list)} environments.")

    batch_sizes, grad_acc_steps = compute_batch_config(env_id_list, effective_batch_size, gpu_num=4)
    total_batch_size = effective_batch_size

    # ==================== Model and Training Settings ====================
    num_unroll_steps = 10
    infer_context_length = 4
    norm_type = 'LN'
    buffer_reanalyze_freq = 1 / 100000000  # Effectively disable buffer reanalyze
    reanalyze_batch_size = 160
    reanalyze_partition = 0.75

    # ==================== Training Loop ====================
    # Set NCCL timeout to prevent watchdog hang due to unbalanced data collection speeds
    os.environ.setdefault('NCCL_TIMEOUT', '3600')  # 60 minutes in seconds
    os.environ.setdefault('NCCL_BLOCKING_WAIT', '1')

    for seed in [0]:
        configs = generate_configs(
            env_id_list, action_space_size, collector_env_num, n_episode, evaluator_env_num,
            num_simulations, reanalyze_ratio, batch_sizes, num_unroll_steps, infer_context_length,
            norm_type, seed, buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition,
            num_segments, total_batch_size, num_layers
        )

        with DDPContext():
            train_unizero_multitask_segment_ddp(configs, seed=seed, max_env_step=max_env_step, benchmark_name="atari")
            print(f"Seed: {seed} training finished!")
            if dist.is_initialized():
                dist.destroy_process_group()
