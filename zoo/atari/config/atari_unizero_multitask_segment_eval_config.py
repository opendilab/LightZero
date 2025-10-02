from easydict import EasyDict
from typing import List, Any, Dict

# ==============================================================
# Environment and Policy Manager Configuration
# ==============================================================

def create_env_manager() -> EasyDict:
    """
    Overview:
        Creates the configuration for the environment and policy managers.
        This config specifies the types and import paths for core components
        like the environment wrapper and the policy definition.
    Returns:
        - manager_config (:obj:`EasyDict`): A dictionary containing the types and import names
                                            for the environment and policy managers.
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

# ==============================================================
# Main Configuration Generation
# ==============================================================

def create_config(
        env_id: str,
        action_space_size: int,
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
        env_id_list: List[str],
) -> EasyDict:
    """
    Overview:
        Creates the main configuration dictionary for a single task in a multi-task setup.
    Arguments:
        - env_id (:obj:`str`): The ID of the environment for this specific task.
        - action_space_size (:obj:`int`): The size of the action space for the model.
        - collector_env_num (:obj:`int`): The number of environments for the data collector.
        - evaluator_env_num (:obj:`int`): The number of environments for the evaluator.
        - n_episode (:obj:`int`): The number of episodes to run for collection.
        - num_simulations (:obj:`int`): The number of simulations for the MCTS algorithm.
        - reanalyze_ratio (:obj:`float`): The ratio of reanalyzed data in the replay buffer.
        - batch_size (:obj:`List[int]`): The batch size for training, specified per task.
        - num_unroll_steps (:obj:`int`): The number of steps to unroll the model during training.
        - infer_context_length (:obj:`int`): The context length for inference.
        - norm_type (:obj:`str`): The type of normalization to use (e.g., 'LN' for LayerNorm).
        - buffer_reanalyze_freq (:obj:`float`): The frequency at which to reanalyze the buffer.
        - reanalyze_batch_size (:obj:`int`): The batch size for reanalyzing data.
        - reanalyze_partition (:obj:`float`): The partition ratio for reanalyzing data.
        - num_segments (:obj:`int`): The number of segments for game data.
        - total_batch_size (:obj:`int`): The total batch size across all tasks.
        - env_id_list (:obj:`List[str]`): The list of all environment IDs in the multi-task setup.
    Returns:
        - config (:obj:`EasyDict`): The complete configuration for a single training task.
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
            multi_gpu=True,  # Enable multi-GPU for DDP
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=50000))),
            grad_correct_params=dict(
                MoCo_beta=0.5, MoCo_beta_sigma=0.5, MoCo_gamma=0.1, MoCo_gamma_sigma=0.5,
                MoCo_rho=0, calpha=0.5, rescale=1,
            ),
            task_num=len(env_id_list),
            task_id=0,  # Placeholder, will be set in generate_configs
            model=dict(
                observation_shape=(3, 64, 64),
                action_space_size=action_space_size,
                norm_type=norm_type,
                num_res_blocks=2,
                num_channels=256,
                world_model_cfg=dict(
                    env_id_list=env_id_list,
                    # TODO: Implement and verify the t-SNE analysis functionality.
                    analysis_tsne=True,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    device='cuda',
                    action_space_size=action_space_size,
                    num_layers=8,  # Transformer layers
                    num_heads=8,
                    embed_dim=768,
                    obs_type='image',
                    env_num=len(env_id_list),
                    task_num=len(env_id_list),
                    use_normal_head=True,
                    use_softmoe_head=False,
                    use_moe_head=False,
                    moe_in_transformer=False,
                    num_experts_of_moe_in_transformer=4,
                ),
            ),
            total_batch_size=total_batch_size,
            allocated_batch_sizes=False,
            train_start_after_envsteps=int(0),
            use_priority=False,
            print_task_priority_logs=False,
            cuda=True,
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            game_segment_length=20,
            update_per_collect=80,
            replay_ratio=0.25,
            batch_size=batch_size,
            optim_type='AdamW',
            num_segments=num_segments,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            n_episode=n_episode,
            replay_buffer_size=int(5e5),
            eval_freq=int(2e4),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
        ),
    ))


def _generate_exp_name_prefix(
        exp_base_path: str,
        num_games: int,
        buffer_reanalyze_freq: float,
        norm_type: str,
        seed: int
) -> str:
    """
    Overview:
        Generates a standardized prefix for the experiment name based on key hyperparameters.
    Arguments:
        - exp_base_path (:obj:`str`): The base directory for the experiment logs.
        - num_games (:obj:`int`): The number of games in the multi-task setup.
        - buffer_reanalyze_freq (:obj:`float`): The frequency of buffer reanalysis.
        - norm_type (:obj:`str`): The normalization type used in the model.
        - seed (:obj:`int`): The random seed for the experiment.
    Returns:
        - (:obj:`str`): The generated experiment name prefix.
    """
    # NOTE: This name is constructed based on a specific convention to encode hyperparameters.
    # It includes details about the model architecture, training parameters, and environment setup.
    return (
        f'{exp_base_path}/{num_games}games_brf{buffer_reanalyze_freq}_'
        f'1-encoder-{norm_type}-res2-channel256_gsl20_{num_games}-pred-head_'
        f'nlayer8-nh24-lsd768_seed{seed}/'
    )


def generate_configs(
        env_id_list: List[str],
        action_space_size: int,
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
        exp_base_path: str,
) -> List[List[Any]]:
    """
    Overview:
        Generates a list of configurations for each task in a multi-task training setup.
        Each configuration is paired with an environment manager config.
    Arguments:
        - (All arguments from create_config, plus):
        - seed (:obj:`int`): The random seed for the experiment, used for naming.
        - exp_base_path (:obj:`str`): The base path for saving experiment results.
    Returns:
        - configs (:obj:`List[List[Any]]`): A list where each item contains
          [task_id, [task_specific_config, env_manager_config]].
    """
    configs = []
    exp_name_prefix = _generate_exp_name_prefix(
        exp_base_path, len(env_id_list), buffer_reanalyze_freq, norm_type, seed
    )

    for task_id, env_id in enumerate(env_id_list):
        config = create_config(
            env_id, action_space_size, collector_env_num, evaluator_env_num,
            n_episode, num_simulations, reanalyze_ratio, batch_size,
            num_unroll_steps, infer_context_length, norm_type,
            buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition,
            num_segments, total_batch_size, env_id_list
        )
        # Assign the specific task ID for this configuration
        config.policy.task_id = task_id
        # Set the full experiment name for logging and checkpointing
        env_name = env_id.split('NoFrameskip')[0]
        config.exp_name = exp_name_prefix + f"{env_name}_unizero-mt_seed{seed}"
        
        configs.append([task_id, [config, create_env_manager()]])
        
    return configs

# ==============================================================
# Main execution block
# ==============================================================

if __name__ == "__main__":
    """
    Overview:
        This program is designed to obtain the t-SNE of the latent states in multi-task learning
        across a set of Atari games (e.g., 8 games).

        This script should be executed with <nproc_per_node> GPUs for Distributed Data Parallel (DDP) training.
        Run one of the following commands to launch the script:
        
        Using `torch.distributed.launch` (deprecated):
        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 ./path/to/this/script.py
        
        Using `torchrun` (recommended):
        torchrun --nproc_per_node=8 ./path/to/this/script.py
    """
    from lzero.entry import train_unizero_multitask_segment_eval
    from ding.utils import DDPContext

    # --- Basic Environment and Model Setup ---
    env_id_list = [
        'PongNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SeaquestNoFrameskip-v4',
        'BoxingNoFrameskip-v4', 'AlienNoFrameskip-v4', 'ChopperCommandNoFrameskip-v4',
        'HeroNoFrameskip-v4', 'RoadRunnerNoFrameskip-v4',
    ]
    action_space_size = 18  # Standard action space size for Atari games

    # --- Hyperparameter Configuration ---
    # Grouping hyperparameters for better readability and management.
    main_hyperparams = {
        'seed': 0,
        'collector_env_num': 2,
        'evaluator_env_num': 2,
        'n_episode': 2,
        'num_simulations': 50,
        'max_env_step': int(4e5),
        'reanalyze_ratio': 0.0,
        'num_segments': 2,
        'num_unroll_steps': 10,
        'infer_context_length': 4,
        'norm_type': 'LN',
        'buffer_reanalyze_freq': 1/50,
        'reanalyze_batch_size': 160,
        'reanalyze_partition': 0.75,
        'total_batch_size': int(4 * len(env_id_list)),
        'batch_size_per_task': 4,
        # --- Path for experiment logs and pretrained model ---
        # NOTE: Please update these paths to your local directory structure.
        'exp_base_path': 'data/unizero_mt_ddp-8gpu_eval-latent_state_tsne',
        # Example for an 8-game pretrained model
        'pretrained_model_path': '/path/to/your/pretrained_model.pth.tar',
        # Example for a 26-game pretrained model
        # 'pretrained_model_path': '/path/to/your/26_game_model.pth.tar',
    }

    # --- Generate Configurations for each seed ---
    # This loop allows running experiments with multiple seeds easily.
    for seed in [main_hyperparams['seed']]:
        # The batch size is a list, with one entry per task.
        batch_size_list = [main_hyperparams['batch_size_per_task']] * len(env_id_list)

        # Generate the list of configurations for the trainer
        configs = generate_configs(
            env_id_list=env_id_list,
            action_space_size=action_space_size,
            collector_env_num=main_hyperparams['collector_env_num'],
            n_episode=main_hyperparams['n_episode'],
            evaluator_env_num=main_hyperparams['evaluator_env_num'],
            num_simulations=main_hyperparams['num_simulations'],
            reanalyze_ratio=main_hyperparams['reanalyze_ratio'],
            batch_size=batch_size_list,
            num_unroll_steps=main_hyperparams['num_unroll_steps'],
            infer_context_length=main_hyperparams['infer_context_length'],
            norm_type=main_hyperparams['norm_type'],
            seed=seed,
            buffer_reanalyze_freq=main_hyperparams['buffer_reanalyze_freq'],
            reanalyze_batch_size=main_hyperparams['reanalyze_batch_size'],
            reanalyze_partition=main_hyperparams['reanalyze_partition'],
            num_segments=main_hyperparams['num_segments'],
            total_batch_size=main_hyperparams['total_batch_size'],
            exp_base_path=main_hyperparams['exp_base_path'],
        )

        # --- Launch Training ---
        # Use DDPContext to manage the distributed training environment.
        with DDPContext():
            train_unizero_multitask_segment_eval(
                configs,
                seed=seed,
                model_path=main_hyperparams['pretrained_model_path'],
                max_env_step=main_hyperparams['max_env_step']
            )