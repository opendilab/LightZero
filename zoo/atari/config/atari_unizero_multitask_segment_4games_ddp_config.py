import torch
from easydict import EasyDict
from copy import deepcopy

def create_config(env_id, action_space_size, collector_env_num, evaluator_env_num, n_episode, num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type, buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition, num_segments):
    return EasyDict(dict(
        env=dict(
            stop_value=int(1e6),
            env_id=env_id,
            observation_shape=(3, 64, 64),
            gray_scale=False,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
            full_action_space=True,
        ),
        policy=dict(
            multi_gpu=True, # ======== Very important for ddp =============
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=200000,),),),  # 默认值是 10000
            grad_correct_params=dict(
                # MoCo 参数
                MoCo_beta=0.5,
                MoCo_beta_sigma=0.5,
                MoCo_gamma=0.1,
                MoCo_gamma_sigma=0.5,
                MoCo_rho=0,
                # CAGrad 参数
                calpha=0.5,
                rescale=1,
            ),
            task_num=len(env_id_list),
            task_id=0,
            model=dict(
                observation_shape=(3, 64, 64),
                action_space_size=action_space_size,
                norm_type=norm_type,
                num_res_blocks=2,  # 适用于 4 个游戏的编码器
                num_channels=128,
                world_model_cfg=dict(
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    action_space_size=action_space_size,
                    num_layers=4,  # Transformer 层数
                    num_heads=8,
                    embed_dim=768,
                    obs_type='image',
                    env_num=8,  # TODO: 所有任务的最大值
                    task_num=len(env_id_list),
                    use_normal_head=True,
                    use_softmoe_head=False,
                    use_moe_head=False,
                    num_experts_in_moe_head=4,  # 注意
                    moe_in_transformer=False,  # 注意
                    multiplication_moe_in_transformer=False,  # 注意
                    num_experts_of_moe_in_transformer=4,
                ),
            ),
            use_priority=False,
            print_task_priority_logs=False,
            cuda=True,
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            game_segment_length=20,
            update_per_collect=160,  # 调整
            replay_ratio=0.25,
            batch_size=batch_size,
            optim_type='AdamW',
            num_segments=num_segments,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            n_episode=n_episode,
            replay_buffer_size=int(5e5),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            # 重新分析的关键参数
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
        ),
    ))

def generate_configs(env_id_list, action_space_size, collector_env_num, n_episode, evaluator_env_num, num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type, seed, buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition, num_segments):
    configs = []
    exp_name_prefix = f'data_unizero_mt_segcollect_1111_ddp4/{len(env_id_list)}games_brf{buffer_reanalyze_freq}/{len(env_id_list)}games_brf{buffer_reanalyze_freq}_1-encoder-{norm_type}-res2-channel128_gsl20_{len(env_id_list)}-pred-head_lsd768-nlayer4-nh8_mbs-320_upc160_seed{seed}/'

    for task_id, env_id in enumerate(env_id_list):
        config = create_config(
            env_id,
            action_space_size,
            collector_env_num,
            evaluator_env_num,
            n_episode,
            num_simulations,
            reanalyze_ratio,
            batch_size,
            num_unroll_steps,
            infer_context_length,
            norm_type,
            buffer_reanalyze_freq,
            reanalyze_batch_size,
            reanalyze_partition,
            num_segments
        )
        config.policy.task_id = task_id
        config.exp_name = exp_name_prefix + f"{env_id.split('NoFrameskip')[0]}_unizero-mt_seed{seed}"

        configs.append([task_id, [config, create_env_manager()]])
    return configs

def create_env_manager():
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
    from lzero.entry import train_unizero_multitask_segment
    env_id_list = [
        'PongNoFrameskip-v4',
        'MsPacmanNoFrameskip-v4',
        'SeaquestNoFrameskip-v4',
        'BoxingNoFrameskip-v4'
    ]

    action_space_size = 18  # 完整的动作空间
    seed = 0

    gpu_num = 4
    collector_env_num = 8
    num_segments = 8
    n_episode = 8

    evaluator_env_num = 3
    num_simulations = 50
    max_env_step = int(5e5)  # 调整

    reanalyze_ratio = 0.

    max_batch_size = 320
    batch_size = [int(min(64, max_batch_size / len(env_id_list))) for _ in range(len(env_id_list))]
    print(f'=========== batch_size: {batch_size} ===========')

    num_unroll_steps = 10
    infer_context_length = 4
    norm_type = 'LN'

    # 定义重新分析的频率。例如，1 表示每个 epoch 重新分析一次，1/10 表示每十个 epoch 重新分析一次
    # buffer_reanalyze_freq = 1 / 50
    buffer_reanalyze_freq = 1 / 100000
    reanalyze_batch_size = 160
    reanalyze_partition = 0.75

    configs = generate_configs(env_id_list, action_space_size, collector_env_num, n_episode, evaluator_env_num, num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type, seed, buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition, num_segments)

    """
    Overview:
        This script should be executed with <nproc_per_node> GPUs.
        Run the following command to launch the script:
        export CUDA_VISIBLE_DEVICES=1,2,3,4
        python -m torch.distributed.launch --nproc_per_node=4 ./zoo/atari/config/atari_unizero_multitask_segment_4games_ddp_config.py
        torchrun --nproc_per_node=4 ./zoo/atari/config/atari_unizero_multitask_segment_4games_ddp_config.py
    """
    from ding.utils import DDPContext
    import numpy as np
    from easydict import EasyDict
    with DDPContext():
        train_unizero_multitask_segment(configs, seed=seed, max_env_step=max_env_step)