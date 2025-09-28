from easydict import EasyDict
from typing import List, Tuple, Union, Any, Dict

class UniZeroAtariConfig:
    """
    Overview:
        Default configuration class for UniZero Atari experiments.
        This class centralizes all default parameters, making it easier to manage and extend.
    """
    def __init__(self) -> None:
        self.exp_name: str = ''
        self.env: EasyDict = self._get_default_env_config()
        self.policy: EasyDict = self._get_default_policy_config()

    @staticmethod
    def _get_default_env_config() -> EasyDict:
        """
        Overview:
            Returns the default environment configuration.
        """
        return EasyDict(dict(
            stop_value=int(1e6),
            env_id='PongNoFrameskip-v4',
            observation_shape=(3, 64, 64),
            gray_scale=False,
            collector_env_num=8,
            evaluator_env_num=3,
            n_evaluator_episode=3,
            manager=dict(shared_memory=False),
            full_action_space=True,
            collect_max_episode_steps=int(5e3),
            eval_max_episode_steps=int(5e3),
        ))

    @staticmethod
    def _get_default_policy_config() -> EasyDict:
        """
        Overview:
            Returns the default policy configuration.
        """
        return EasyDict(dict(
            multi_gpu=True,
            # ==============TODO==============
            use_moco=False,
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=50000))),
            grad_correct_params=dict(
                MoCo_beta=0.5,
                MoCo_beta_sigma=0.5,
                MoCo_gamma=0.1,
                MoCo_gamma_sigma=0.5,
                MoCo_rho=0,
                calpha=0.5,
                rescale=1,
            ),
            task_num=1,
            task_id=0,
            model=dict(
                observation_shape=(3, 64, 64),
                action_space_size=18,
                norm_type='LN',
                num_res_blocks=2,
                num_channels=256,
                world_model_cfg=dict(
                    # TODO: for latent state layer_norm
                    final_norm_option_in_obs_head='LayerNorm',
                    final_norm_option_in_encoder='LayerNorm',
                    predict_latent_loss_type='mse',
                    # TODO: only for latent state sim_norm
                    # final_norm_option_in_obs_head='SimNorm',
                    # final_norm_option_in_encoder='SimNorm',
                    # predict_latent_loss_type='group_kl',
                    share_head=False,  # TODO
                    analysis_dormant_ratio_weight_rank=False,  # TODO
                    dormant_threshold=0.025,
                    continuous_action_space=False,
                    # ==============TODO: none ==============
                    task_embed_option=None,
                    use_task_embed=False,
                    # ==============TODO==============
                    # task_embed_option='concat_task_embed',
                    # use_task_embed=True,
                    # task_embed_dim=96,
                    # task_embed_dim=128,
                    use_shared_projection=False,
                    max_blocks=10, # num_unroll_steps
                    max_tokens=20, # 2 * num_unroll_steps
                    context_length=8, # 2 * infer_context_length
                    device='cuda',
                    action_space_size=18,
                    num_layers=8,
                    num_heads=24,
                    embed_dim=768,
                    obs_type='image',
                    env_num=8,
                    task_num=1,
                    use_normal_head=True,
                    use_softmoe_head=False,
                    use_moe_head=False,
                    num_experts_in_moe_head=4,
                    moe_in_transformer=False,
                    multiplication_moe_in_transformer=False,
                    num_experts_of_moe_in_transformer=4,
                    # LoRA parameters (enable LoRA by setting lora_r > 0)
                    lora_r=0,
                    # lora_r=8,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    # Default target modules: attn and feed_forward
                    lora_target_modules=["attn", "feed_forward"],
                ),
            ),
            # TODO
            use_task_exploitation_weight=False,
            task_complexity_weight=False,
            total_batch_size=512,
            allocated_batch_sizes=False,
            train_start_after_envsteps=int(0),
            use_priority=False,
            print_task_priority_logs=False,
            cuda=True,
            model_path=None,
            num_unroll_steps=10,
            game_segment_length=20,
            update_per_collect=80,
            replay_ratio=0.25,
            batch_size=64,
            optim_type='AdamW',
            cos_lr_scheduler=True,
            num_segments=8,
            num_simulations=50,
            reanalyze_ratio=0.0,
            n_episode=8,
            replay_buffer_size=int(5e5),
            eval_freq=int(2e4),
            collector_env_num=8,
            evaluator_env_num=3,
            buffer_reanalyze_freq=1 / 10000000,
            reanalyze_batch_size=160,
            reanalyze_partition=0.75,
        ))

def create_config(
        env_id: str,
        action_space_size: int,
        collector_env_num: int,
        evaluator_env_num: int,
        n_episode: int,
        num_simulations: int,
        reanalyze_ratio: float,
        batch_size: Union[int, List[int]],
        num_unroll_steps: int,
        infer_context_length: int,
        norm_type: str,
        buffer_reanalyze_freq: float,
        reanalyze_batch_size: int,
        reanalyze_partition: float,
        num_segments: int,
        total_batch_size: int,
        task_num: int
) -> EasyDict:
    """
    Overview:
        Creates and customizes a configuration for a specific Atari environment task.

    Arguments:
        - env_id (:obj:`str`): The ID of the Atari environment.
        - action_space_size (:obj:`int`): The size of the action space.
        - collector_env_num (:obj:`int`): Number of environments for collecting data.
        - evaluator_env_num (:obj:`int`): Number of environments for evaluation.
        - n_episode (:obj:`int`): Number of episodes to run for each collection.
        - num_simulations (:obj:`int`): Number of simulations in the MCTS.
        - reanalyze_ratio (:obj:`float`): The ratio of reanalyzed samples in the replay buffer.
        - batch_size (:obj:`Union[int, List[int]]`): The batch size for training.
        - num_unroll_steps (:obj:`int`): The number of steps to unroll the model.
        - infer_context_length (:obj:`int`): The context length for inference.
        - norm_type (:obj:`str`): The type of normalization to use.
        - buffer_reanalyze_freq (:obj:`float`): Frequency of reanalyzing the buffer.
        - reanalyze_batch_size (:obj:`int`): Batch size for reanalyzing.
        - reanalyze_partition (:obj:`float`): Partition ratio for reanalyzing.
        - num_segments (:obj:`int`): Number of segments for each game.
        - total_batch_size (:obj:`int`): The total batch size across all tasks.
        - task_num (:obj:`int`): The total number of tasks.

    Returns:
        - (:obj:`EasyDict`): A fully configured EasyDict object for the experiment.
    """
    cfg = UniZeroAtariConfig()

    # == Update Environment Config ==
    cfg.env.env_id = env_id
    cfg.env.collector_env_num = collector_env_num
    cfg.env.evaluator_env_num = evaluator_env_num
    cfg.env.n_evaluator_episode = evaluator_env_num

    # == Update Policy Config ==
    policy = cfg.policy
    policy.task_num = task_num
    policy.action_space_size = action_space_size
    policy.n_episode = n_episode
    policy.num_simulations = num_simulations
    policy.reanalyze_ratio = reanalyze_ratio
    policy.batch_size = batch_size
    policy.total_batch_size = total_batch_size
    policy.num_unroll_steps = num_unroll_steps
    policy.collector_env_num = collector_env_num
    policy.evaluator_env_num = evaluator_env_num
    policy.buffer_reanalyze_freq = buffer_reanalyze_freq
    policy.reanalyze_batch_size = reanalyze_batch_size
    policy.reanalyze_partition = reanalyze_partition
    policy.num_segments = num_segments

    # == Update Model Config ==
    model = policy.model
    model.action_space_size = action_space_size
    model.norm_type = norm_type
    
    # == Update World Model Config ==
    world_model = model.world_model_cfg
    world_model.max_blocks = num_unroll_steps
    world_model.max_tokens = 2 * num_unroll_steps
    world_model.context_length = 2 * infer_context_length
    world_model.action_space_size = action_space_size
    world_model.task_num = task_num

    return EasyDict(cfg)


def generate_experiment_configs(
        env_id_list: List[str],
        action_space_size: int,
        collector_env_num: int,
        n_episode: int,
        evaluator_env_num: int,
        num_simulations: int,
        reanalyze_ratio: float,
        batch_size: Union[int, List[int]],
        num_unroll_steps: int,
        infer_context_length: int,
        norm_type: str,
        seed: int,
        buffer_reanalyze_freq: float,
        reanalyze_batch_size: int,
        reanalyze_partition: float,
        num_segments: int,
        total_batch_size: int
) -> List[Tuple[int, List[Union[EasyDict, Any]]]]:
    """
    Overview:
        Generates a list of configurations for multi-task experiments.

    Arguments:
        - env_id_list (:obj:`List[str]`): List of environment IDs for the tasks.
        - ... (same as create_config): Other experiment parameters.
        - seed (:obj:`int`): The random seed for the experiment.

    Returns:
        - (:obj:`List[Tuple[int, List[Union[EasyDict, Any]]]]`): A list where each element contains a task_id and its
          corresponding configuration and environment manager setup.
    """
    configs = []
    task_num = len(env_id_list)
    
    # --- Experiment Name Prefix ---
    # This prefix defines the storage path for experiment data and logs.
    # Please replace `<YOUR_EXPERIMENT_DATA_PATH>` with your actual data storage path.
    exp_name_prefix_template = (
        "<YOUR_EXPERIMENT_DATA_PATH>/data_unizero_atari_mt_finetune_{timestamp}/"
        "experiment_name/{task_num}games_brf{brf}_1-encoder-{norm}-res2-channel256_"
        "gsl20_lsd768-nlayer8-nh8_upc80_seed{seed}/"
    )
    exp_name_prefix = exp_name_prefix_template.format(
        timestamp="20250308",
        task_num=task_num,
        brf=buffer_reanalyze_freq,
        norm=norm_type,
        seed=seed
    )

    for task_id, env_id in enumerate(env_id_list):
        config = create_config(
            env_id, action_space_size, collector_env_num, evaluator_env_num,
            n_episode, num_simulations, reanalyze_ratio, batch_size,
            num_unroll_steps, infer_context_length, norm_type,
            buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition,
            num_segments, total_batch_size, task_num
        )
        config.policy.task_id = task_id
        config.exp_name = exp_name_prefix + f"{env_id.split('NoFrameskip')[0]}_unizero-mt_seed{seed}"
        configs.append([task_id, [config, create_env_manager()]])
    return configs


def create_env_manager() -> EasyDict:
    """
    Overview:
        Creates the environment and policy manager configuration.
        This specifies the types and import paths for the environment and policy used in the experiment.

    Returns:
        - (:obj:`EasyDict`): An EasyDict object containing manager configurations.
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
        This script should be executed with <nproc_per_node> GPUs.
        Run one of the following commands to launch the script:
        - Using torch.distributed.launch:
          python -m torch.distributed.launch --nproc_per_node=8 --master_port=29507 ./path/to/this/script.py
        - Using torchrun:
          torchrun --nproc_per_node=8 ./path/to/this/script.py
    """
    from lzero.entry import train_unizero_multitask_segment_ddp
    from ding.utils import DDPContext
    import os

    # --- Main Experiment Settings ---
    # Use DEBUG mode for fast iteration and debugging.
    DEBUG = False

    # --- Environment and Task Settings ---
    env_id_list = ['AmidarNoFrameskip-v4']
    action_space_size = 18

    # --- Distributed Training Settings ---
    os.environ["NCCL_TIMEOUT"] = "3600000000"

    # --- Loop over seeds for multiple runs ---
    for seed in [0]:
        # --- Core Algorithm Parameters ---
        if DEBUG:
            # Settings for quick debugging
            collector_env_num = 2
            num_segments = 2
            n_episode = 2
            evaluator_env_num = 2
            num_simulations = 2
            total_batch_size = 32
            batch_size = [int(total_batch_size / len(env_id_list))] * len(env_id_list)
            reanalyze_batch_size = 4
            max_env_step = int(1e3)
        else:
            # Standard experiment settings
            collector_env_num = 8
            num_segments = 8
            n_episode = 8
            evaluator_env_num = 3
            num_simulations = 50
            total_batch_size = 512
            batch_size = [int(min(64, total_batch_size / len(env_id_list)))] * len(env_id_list)
            reanalyze_batch_size = 160
            max_env_step = int(4e5)

        # --- Shared Parameters ---
        reanalyze_ratio = 0.0
        num_unroll_steps = 10
        infer_context_length = 4
        norm_type = 'LN'
        buffer_reanalyze_freq = 1 / 10000000  # Effectively disabled
        reanalyze_partition = 0.75

        # --- Generate Configurations ---
        configs = generate_experiment_configs(
            env_id_list=env_id_list,
            action_space_size=action_space_size,
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
            total_batch_size=total_batch_size
        )

        # --- Pretrained Model Path ---
        # Please replace `<YOUR_PRETRAINED_MODEL_PATH>` with the actual path to your model.
        pretrained_model_path = (
            "<YOUR_PRETRAINED_MODEL_PATH>/data_unizero_atari_mt_20250307/"
            "atari_8games_brf0.02_not-share-head_final-ln_seed0/Pong_seed0/ckpt/ckpt_best.pth.tar"
        )
        
        # --- Start Training ---
        with DDPContext():
            train_unizero_multitask_segment_ddp(
                configs,
                seed=seed,
                model_path=pretrained_model_path,
                max_env_step=max_env_step
            )