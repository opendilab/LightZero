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
            collect_max_episode_steps=int(5e3),
            eval_max_episode_steps=int(5e3),
        ),
        policy=dict(
            multi_gpu=True,  # Essential for DDP (Distributed Data Parallel)
            only_use_moco_stats=False,
            use_moco=False,
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=200000))),
            grad_correct_params=dict(
                MoCo_beta=0.5, MoCo_beta_sigma=0.5, MoCo_gamma=0.1, MoCo_gamma_sigma=0.5, MoCo_rho=0,
                calpha=0.5, rescale=1,
            ),
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
                continuous_action_space=False,
                world_model_cfg=dict(
                    num_res_blocks=2,
                    num_channels=256,
                    norm_type=norm_type,
                    use_global_pooling=False,
                    final_norm_option_in_obs_head='LayerNorm',
                    final_norm_option_in_encoder='LayerNorm',
                    predict_latent_loss_type='mse',
                    share_head=False,
                    analysis_dormant_ratio_weight_rank=False,
                    # analysis_dormant_ratio_weight_rank=True,
                    # analysis_dormant_ratio_interval=5000,
                    continuous_action_space=False,
                    task_embed_option=None,
                    use_task_embed=False,
                    use_shared_projection=False,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    device='cuda',
                    action_space_size=action_space_size,
                    num_layers=num_layers,
                    # num_heads=24,
                    num_heads=8,
                    embed_dim=768,
                    obs_type='image',
                    env_num=len(env_id_list),
                    task_num=len(env_id_list),
                    # game_segment_length=game_segment_length,
                    game_segment_length=20, # TODO
                    use_priority=True,
                    # use_priority=False, # TODO=====
                    priority_prob_alpha=1,
                    priority_prob_beta=1,
                    # encoder_type='vit',
                    encoder_type='resnet',
                    use_normal_head=True,
                    use_softmoe_head=False,
                    use_moe_head=False,
                    num_experts_in_moe_head=4,
                    moe_in_transformer=False,

                    multiplication_moe_in_transformer=True,
                    # multiplication_moe_in_transformer=False, # TODO=====

                    n_shared_experts=1,
                    num_experts_per_tok=1,
                    num_experts_of_moe_in_transformer=8,
                    # LoRA parameters
                    moe_use_lora=False,
                    lora_r=0,
                    lora_alpha=1,
                    lora_dropout=0.0,


                    optim_type='AdamW_mix_lr_wdecay', # only for tsne plot
                ),
            ),
            optim_type='AdamW_mix_lr_wdecay',
            weight_decay=1e-2, # TODO: encoder 5*wd, transformer wd, head 0
            learning_rate=0.0001,

            # (bool) 是否启用自适应策略熵权重 (alpha)
            use_adaptive_entropy_weight=True,
            # use_adaptive_entropy_weight=False,

            # (float) 自适应alpha优化器的学习率
            adaptive_entropy_alpha_lr=1e-4,
            target_entropy_start_ratio =0.98,
            # target_entropy_end_ratio =0.9, # TODO=====
            target_entropy_end_ratio =0.7,
            target_entropy_decay_steps = 100000, # 例如，在100k次迭代后达到最终值


            # ==================== START: Encoder-Clip Annealing Config ====================
            # (bool) 是否启用 encoder-clip 值的退火。
            use_encoder_clip_annealing=True,
            # (str) 退火类型。可选 'linear' 或 'cosine'。
            encoder_clip_anneal_type='cosine',
            # (float) 退火的起始 clip 值 (训练初期，较宽松)。
            encoder_clip_start_value=30.0,
            # (float) 退火的结束 clip 值 (训练后期，较严格)。
            encoder_clip_end_value=10.0,
            # (int) 完成从起始值到结束值的退火所需的训练迭代步数。
            encoder_clip_anneal_steps=100000,  # 例如，在100k次迭代后达到最终值

            # ==================== START: label smooth ====================
            policy_ls_eps_start=0.05, #TODO============= good start in Pong and MsPacman
            policy_ls_eps_end=0.01,
            policy_ls_eps_decay_steps=50000, # 50k
            label_smoothing_eps=0.1,  #TODO============= for value

            # ==================== [新增] 范数监控频率 ====================
            # 每隔多少个训练迭代步数，监控一次模型参数的范数。设置为0则禁用。
            monitor_norm_freq=10000,
            # monitor_norm_freq=2,  # only for debug

            use_task_exploitation_weight=False,
            task_complexity_weight=False,
            total_batch_size=total_batch_size,
            allocated_batch_sizes=False,
            train_start_after_envsteps=int(0),
            # use_priority=False, # TODO=====
            use_priority=True,
            priority_prob_alpha=1,
            priority_prob_beta=1,
            print_task_priority_logs=False,
            cuda=True,
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            game_segment_length=20,
            update_per_collect=80,  # Corresponds to replay_ratio=0.5 for 8 games (20*8*0.5=80)
            replay_ratio=0.25,
            batch_size=batch_size,
            # optim_type='AdamW',
            cos_lr_scheduler=False,
            num_segments=num_segments,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            n_episode=n_episode,
            replay_buffer_size=int(5e5),
            # eval_freq=int(2e4),  # Evaluation frequency for 26 games
            eval_freq=int(1e4),  # Evaluation frequency for 8 games
            # eval_freq=int(1e4),  # Evaluation frequency for 8 games
            # eval_freq=int(2),  # ======== TODO: only for debug========
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
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
    # Replace placeholders like [BENCHMARK_TAG] and [MODEL_TAG] to define the experiment name.
    benchmark_tag = "data_unizero_mt_refactor1010"  # e.g., unizero_atari_mt_20250612
    # model_tag = f"vit-small_moe8_tbs512_tran-nlayer{num_layers}_brf{buffer_reanalyze_freq}_not-share-head"
    # model_tag = f"resnet_noprior_noalpha_nomoe_head-inner-ln_adamw-wd1e-2_tbs512_tran-nlayer{num_layers}_brf{buffer_reanalyze_freq}"
    
    # model_tag = f"vit_prior_alpha-100k-098-07_encoder-100k-30-10_moe8_head-inner-ln_adamw-wd1e-2_tbs512_tran-nlayer{num_layers}_brf{buffer_reanalyze_freq}"

    model_tag = f"resnet_encoder-100k-30-10-true_label-smooth_prior_alpha-100k-098-07_moe8_head-inner-ln_adamw-wd1e-2_tbs512_tran-nlayer{num_layers}_brf{buffer_reanalyze_freq}"

    exp_name_prefix = f'{benchmark_tag}/atari_{len(env_id_list)}games_{model_tag}_seed{seed}/'

    for task_id, env_id in enumerate(env_id_list):
        config = create_config(
            env_id, action_space_size, collector_env_num, evaluator_env_num, n_episode, num_simulations,
            reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type,
            buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition, num_segments, total_batch_size, num_layers
        )
        config.policy.task_id = task_id
        config.exp_name = exp_name_prefix + f"{env_id.split('NoFrameskip')[0]}_seed{seed}"
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
        Run the following command to launch the script:

        Example launch command:
        export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
        cd /path/to/your/project/
        python -m torch.distributed.launch --nproc_per_node=6 --master_port=29502 /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/config/atari_unizero_multitask_segment_ddp_config.py
            /path/to/this/script.py 2>&1 | tee /path/to/your/log/file.log
    """
    from lzero.entry import train_unizero_multitask_segment_ddp
    from ding.utils import DDPContext
    import torch.distributed as dist
    import os

    # --- Main Experiment Settings ---
    num_games = 8  # Options: 3, 8, 26
    # num_layers = 4
    num_layers = 2 # debug
    action_space_size = 18
    collector_env_num = 8
    num_segments = 8
    n_episode = 8
    evaluator_env_num = 3
    num_simulations = 50
    # max_env_step = int(4e5)
    max_env_step = int(5e6) # TODO
    reanalyze_ratio = 0.0

    if num_games == 3:
        env_id_list = ['PongNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SeaquestNoFrameskip-v4']
    elif num_games == 8:
        env_id_list = [
            'PongNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SeaquestNoFrameskip-v4', 'BoxingNoFrameskip-v4',
            'AlienNoFrameskip-v4', 'ChopperCommandNoFrameskip-v4', 'HeroNoFrameskip-v4', 'RoadRunnerNoFrameskip-v4',
        ]
    elif num_games == 26:
        env_id_list = [
            'PongNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SeaquestNoFrameskip-v4', 'BoxingNoFrameskip-v4',
            'AlienNoFrameskip-v4', 'ChopperCommandNoFrameskip-v4', 'HeroNoFrameskip-v4', 'RoadRunnerNoFrameskip-v4',
            'AmidarNoFrameskip-v4', 'AssaultNoFrameskip-v4', 'AsterixNoFrameskip-v4', 'BankHeistNoFrameskip-v4',
            'BattleZoneNoFrameskip-v4', 'CrazyClimberNoFrameskip-v4', 'DemonAttackNoFrameskip-v4', 'FreewayNoFrameskip-v4',
            'FrostbiteNoFrameskip-v4', 'GopherNoFrameskip-v4', 'JamesbondNoFrameskip-v4', 'KangarooNoFrameskip-v4',
            'KrullNoFrameskip-v4', 'KungFuMasterNoFrameskip-v4', 'PrivateEyeNoFrameskip-v4', 'UpNDownNoFrameskip-v4',
            'QbertNoFrameskip-v4', 'BreakoutNoFrameskip-v4',
        ]
    else:
        raise ValueError(f"Unsupported number of environments: {num_games}")

    # --- Batch Size Calculation ---
    # The effective batch size is adjusted based on the number of games and model size (layers)
    # to fit within GPU memory constraints.
    if len(env_id_list) == 8:
        if num_layers in [2, 4]:
            effective_batch_size = 512
        elif num_layers == 8:
            effective_batch_size = 512
    elif len(env_id_list) == 26:
        effective_batch_size = 512
    elif len(env_id_list) == 18:
        effective_batch_size = 1536
    elif len(env_id_list) == 3:
        effective_batch_size = 10  # For debugging
    else:
        raise ValueError(f"Batch size not configured for {len(env_id_list)} environments.")

    batch_sizes, grad_acc_steps = compute_batch_config(env_id_list, effective_batch_size)
    total_batch_size = effective_batch_size  # Currently for logging purposes

    # --- Model and Training Settings ---
    num_unroll_steps = 10
    infer_context_length = 4
    norm_type = 'LN'
    buffer_reanalyze_freq = 1 / 100000000  # Effectively disable buffer reanalyze
    reanalyze_batch_size = 160
    reanalyze_partition = 0.75

    # ====== only for debug =====
    # num_games = 4  # Options: 3, 8, 26
    # num_layers = 2 # debug
    # collector_env_num = 2
    # num_segments = 2
    # evaluator_env_num = 2
    # num_simulations = 5
    # batch_sizes = [num_games] * len(env_id_list)
    # buffer_reanalyze_freq = 1/100000000
    # total_batch_size = num_games * len(env_id_list)


    # --- Training Loop ---
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