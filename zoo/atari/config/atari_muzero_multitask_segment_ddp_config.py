"""
Overview:
    Configuration generation script for multi-task MuZero training on Atari environments.
    This script defines and generates the necessary configuration files for a distributed training setup.
"""
from easydict import EasyDict
from copy import deepcopy
from typing import List, Union, Dict, Any

# The 'atari_env_action_space_map' was not used in the original code, so it has been removed.

class AtariMuZeroMultitaskConfig:
    """
    Overview:
        A class to generate and manage configurations for multi-task MuZero experiments on Atari.
        It encapsulates the entire configuration logic, providing a clean and extensible interface.
    """

    def __init__(
        self,
        env_id_list: List[str],
        seed: int,
        num_unroll_steps: int,
        num_simulations: int,
        collector_env_num: int,
        evaluator_env_num: int,
        max_env_step: int,
        batch_size: Union[List[int], int],
        norm_type: str,
        buffer_reanalyze_freq: float,
        reanalyze_batch_size: int,
        reanalyze_partition: float,
        num_segments: int,
        exp_path_prefix: str = 'YOUR_EXPERIMENT_PATH_PREFIX/data_muzero_mt_atari',
    ) -> None:
        """
        Overview:
            Initializes the multi-task configuration generator.
        Arguments:
            - env_id_list (:obj:`List[str]`): A list of Atari environment IDs to be trained on.
            - seed (:obj:`int`): The random seed for the experiment.
            - num_unroll_steps (:obj:`int`): The number of steps to unroll the model during training.
            - num_simulations (:obj:`int`): The number of simulations to run in the MCTS search.
            - collector_env_num (:obj:`int`): The number of environments for data collection.
            - evaluator_env_num (:obj:`int`): The number of environments for evaluation.
            - max_env_step (:obj:`int`): The total number of environment steps to train for.
            - batch_size (:obj:`Union[List[int], int]`): The batch size for training. Can be a list for per-task sizes or a single int.
            - norm_type (:obj:`str`): The type of normalization to use in the model (e.g., 'BN', 'LN').
            - buffer_reanalyze_freq (:obj:`float`): The frequency at which to reanalyze the replay buffer.
            - reanalyze_batch_size (:obj:`int`): The batch size for reanalysis.
            - reanalyze_partition (:obj:`float`): The partition ratio for reanalysis.
            - num_segments (:obj:`int`): The number of segments for the replay buffer.
            - exp_path_prefix (:obj:`str`): A template for the experiment's output path.
        """
        self.env_id_list = env_id_list
        self.seed = seed
        self.num_unroll_steps = num_unroll_steps
        self.num_simulations = num_simulations
        self.collector_env_num = collector_env_num
        self.evaluator_env_num = evaluator_env_num
        self.max_env_step = max_env_step
        self.batch_size = batch_size
        self.norm_type = norm_type
        self.buffer_reanalyze_freq = buffer_reanalyze_freq
        self.reanalyze_batch_size = reanalyze_batch_size
        self.reanalyze_partition = reanalyze_partition
        self.num_segments = num_segments
        self.exp_path_prefix = exp_path_prefix

        # --- Derived attributes ---
        self.num_tasks = len(self.env_id_list)
        self.action_space_size = 18  # Default full action space for Atari

    def _create_base_config(self) -> EasyDict:
        """
        Overview:
            Creates the base configuration dictionary with shared settings for all tasks.
        Returns:
            - (:obj:`EasyDict`): A dictionary containing the base configuration.
        """
        return EasyDict(dict(
            env=dict(
                stop_value=int(self.max_env_step),
                observation_shape=(4, 96, 96),
                frame_stack_num=4,
                gray_scale=True,
                collector_env_num=self.collector_env_num,
                evaluator_env_num=self.evaluator_env_num,
                n_evaluator_episode=self.evaluator_env_num,
                manager=dict(shared_memory=False),
                full_action_space=True,
                collect_max_episode_steps=int(5e3),
                eval_max_episode_steps=int(5e3),
            ),
            policy=dict(
                multi_gpu=True,  # Very important for DDP
                learn=dict(
                    learner=dict(
                        hook=dict(save_ckpt_after_iter=200000),
                    ),
                ),
                grad_correct_params=dict(),
                task_num=self.num_tasks,
                model=dict(
                    device='cuda',
                    num_res_blocks=2,
                    num_channels=256,
                    reward_head_channels=16,
                    value_head_channels=16,
                    policy_head_channels=16,
                    fc_reward_layers=[32],
                    fc_value_layers=[32],
                    fc_policy_layers=[32],
                    observation_shape=(4, 96, 96),
                    frame_stack_num=4,
                    gray_scale=True,
                    action_space_size=self.action_space_size,
                    norm_type=self.norm_type,
                    model_type='conv',
                    image_channel=1,
                    downsample=True,
                    self_supervised_learning_loss=True,
                    discrete_action_encoding_type='one_hot',
                    use_sim_norm=True,
                    use_sim_norm_kl_loss=False,
                    task_num=self.num_tasks,
                ),
                allocated_batch_sizes=False,
                cuda=True,
                env_type='not_board_games',
                train_start_after_envsteps=2000,
                # train_start_after_envsteps=0, # TODO: debug
                game_segment_length=20,
                random_collect_episode_num=0,
                use_augmentation=True,
                use_priority=False,
                replay_ratio=0.25,
                num_unroll_steps=self.num_unroll_steps,
                update_per_collect=80,
                optim_type='SGD',
                td_steps=5,
                lr_piecewise_constant_decay=True,
                manual_temperature_decay=False,
                learning_rate=0.2,
                target_update_freq=100,
                num_segments=self.num_segments,
                num_simulations=self.num_simulations,
                policy_entropy_weight=5e-3, # TODO: Fine-tune this weight.
                ssl_loss_weight=2,
                eval_freq=int(5e3),
                replay_buffer_size=int(5e5),
                collector_env_num=self.collector_env_num,
                evaluator_env_num=self.evaluator_env_num,
                # ============= Reanalyze Parameters =============
                buffer_reanalyze_freq=self.buffer_reanalyze_freq,
                reanalyze_batch_size=self.reanalyze_batch_size,
                reanalyze_partition=self.reanalyze_partition,
            ),
        ))

    def _get_exp_name(self, env_id: str) -> str:
        """
        Overview:
            Generates a formatted experiment name for a given task.
        Arguments:
            - env_id (:obj:`str`): The environment ID for the specific task.
        Returns:
            - (:obj:`str`): The formatted experiment name.
        """
        # TODO: debug name
        prefix = (
            f'{self.exp_path_prefix}/{self.num_tasks}games_brf{self.buffer_reanalyze_freq}/'
            f'{self.num_tasks}games_brf{self.buffer_reanalyze_freq}_1-encoder-{self.norm_type}-res2-channel256_gsl20_'
            f'{self.num_tasks}-pred-head_mbs-512_upc80_H{self.num_unroll_steps}_seed{self.seed}/'
        )
        env_name = env_id.split('NoFrameskip')[0]
        return f"{prefix}{env_name}_muzero-mt_seed{self.seed}"

    def generate_configs(self) -> List[List[Union[int, List[Any]]]]:
        """
        Overview:
            Generates the final list of configurations for all specified tasks,
            ready to be used by the training entry point.
        Returns:
            - (:obj:`List[List[Union[int, List[Any]]]]`): A list where each element corresponds to a task,
              containing the task_id and a list with the task's config and env_manager config.
        """
        base_config = self._create_base_config()
        env_manager_config = self._create_env_manager_config()
        
        configs = []
        for task_id, env_id in enumerate(self.env_id_list):
            task_config = deepcopy(base_config)
            
            # --- Apply task-specific settings ---
            task_config.env.env_id = env_id
            task_config.policy.task_id = task_id
            
            # Handle per-task batch size if provided as a list
            if isinstance(self.batch_size, list):
                task_config.policy.batch_size = self.batch_size[task_id]
            else:
                task_config.policy.batch_size = self.batch_size
            
            task_config.exp_name = self._get_exp_name(env_id)

            configs.append([task_id, [task_config, env_manager_config]])
            
        return configs

    @staticmethod
    def _create_env_manager_config() -> EasyDict:
        """
        Overview:
            Creates a static configuration for the environment and policy managers.
        Returns:
            - (:obj:`EasyDict`): A dictionary containing manager configurations.
        """
        return EasyDict(dict(
            env=dict(
                type='atari_lightzero',
                import_names=['zoo.atari.envs.atari_lightzero_env'],
            ),
            env_manager=dict(type='subprocess'),
            policy=dict(
                type='muzero_multitask',
                import_names=['lzero.policy.muzero_multitask'],
            ),
        ))


if __name__ == "__main__":
    # ==============================================================
    #           Hyperparameters for Multi-Task Training
    # ==============================================================
    
    # --- List of Atari environments for multi-task learning ---
    env_id_list = [
        'PongNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SeaquestNoFrameskip-v4',
        'BoxingNoFrameskip-v4', 'AlienNoFrameskip-v4', 'ChopperCommandNoFrameskip-v4',
        'HeroNoFrameskip-v4', 'RoadRunnerNoFrameskip-v4', 'AmidarNoFrameskip-v4',
        'AssaultNoFrameskip-v4', 'AsterixNoFrameskip-v4', 'BankHeistNoFrameskip-v4',
        'BattleZoneNoFrameskip-v4', 'CrazyClimberNoFrameskip-v4', 'DemonAttackNoFrameskip-v4',
        'FreewayNoFrameskip-v4', 'FrostbiteNoFrameskip-v4', 'GopherNoFrameskip-v4',
        'JamesbondNoFrameskip-v4', 'KangarooNoFrameskip-v4', 'KrullNoFrameskip-v4',
        'KungFuMasterNoFrameskip-v4', 'PrivateEyeNoFrameskip-v4', 'UpNDownNoFrameskip-v4',
        'QbertNoFrameskip-v4', 'BreakoutNoFrameskip-v4',
    ]

    # --- Core Experiment Settings ---
    seed = 0
    max_env_step = int(5e5)
    
    # --- Training & Model Parameters ---
    num_unroll_steps = 5
    num_simulations = 50
    norm_type = 'BN'  # 'BN' (Batch Normalization) or 'LN' (Layer Normalization)

    # --- Environment & Collector Settings ---
    collector_env_num = 8
    evaluator_env_num = 3
    num_segments = 8

    # --- Batch Size Configuration ---
    # The batch size is dynamically calculated per task to not exceed a maximum total batch size.
    max_batch_size = 512
    per_task_batch_size = int(min(64, max_batch_size / len(env_id_list)))
    batch_size = [per_task_batch_size] * len(env_id_list)

    # --- Reanalyze Buffer Settings ---
    buffer_reanalyze_freq = 1 / 50
    reanalyze_batch_size = 160
    reanalyze_partition = 0.75

    # --- (Optional) Debug Settings ---
    # To use debug settings, uncomment the following lines.
    # collector_env_num = 2
    # evaluator_env_num = 2
    # num_segments = 2
    # num_simulations = 3
    # debug_batch_size = int(min(2, max_batch_size / len(env_id_list)))
    # batch_size = [debug_batch_size] * len(env_id_list)
    # print("--- RUNNING IN DEBUG MODE ---")
    
    print(f'=========== Batch size per task: {batch_size[0]} ===========')

    # ==============================================================
    #           Configuration Generation and Training Launch
    # ==============================================================
    
    # --- Instantiate and generate configurations ---
    experiment_config = AtariMuZeroMultitaskConfig(
        env_id_list=env_id_list,
        seed=seed,
        max_env_step=max_env_step,
        num_unroll_steps=num_unroll_steps,
        num_simulations=num_simulations,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        batch_size=batch_size,
        norm_type=norm_type,
        buffer_reanalyze_freq=buffer_reanalyze_freq,
        reanalyze_batch_size=reanalyze_batch_size,
        reanalyze_partition=reanalyze_partition,
        num_segments=num_segments,
        # Note: Update this path to your desired location.
        exp_path_prefix='YOUR_EXPERIMENT_PATH_PREFIX/data_muzero_mt_atari_20250228'
    )
    
    configs_to_run = experiment_config.generate_configs()

    # --- Launch Distributed Training ---
    """
    Overview:
        This script should be executed with <nproc_per_node> GPUs.
        Set the NCCL timeout and launch the script using one of the following commands.
    
    Command using torch.distributed.launch:
        export NCCL_TIMEOUT=3600000
        python -m torch.distributed.launch --nproc_per_node=4 --master_port=29501 ./path/to/this/script.py
        
    Command using torchrun:
        export NCCL_TIMEOUT=3600000
        torchrun --nproc_per_node=4 --master_port=29501 ./path/to/this/script.py
    """
    from lzero.entry import train_muzero_multitask_segment_ddp
    from ding.utils import DDPContext

    with DDPContext():
        train_muzero_multitask_segment_ddp(configs_to_run, seed=seed, max_env_step=max_env_step)