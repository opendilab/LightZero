from easydict import EasyDict

def create_config(env_id, action_space_size, collector_env_num, evaluator_env_num, n_episode, num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type, buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition, num_segments, total_batch_size):
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
            multi_gpu=True,
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=50000))),
            grad_correct_params=dict(  # Gradient correction parameters
                MoCo_beta=0.5,
                MoCo_beta_sigma=0.5,
                MoCo_gamma=0.1,
                MoCo_gamma_sigma=0.5,
                MoCo_rho=0,
                calpha=0.5,
                rescale=1,
            ),
            task_num=len(env_id_list),
            task_id=0,
            model=dict(
                observation_shape=(3, 64, 64),
                action_space_size=action_space_size,
                norm_type=norm_type,
                num_res_blocks=2,
                num_channels=256,
                world_model_cfg=dict(
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    device='cuda',
                    action_space_size=action_space_size,
                    num_layers=8,
                    num_heads=24,
                    embed_dim=768,
                    obs_type='image',
                    env_num=8,
                    task_num=len(env_id_list),
                    use_normal_head=True,
                    use_softmoe_head=False,
                    moe_in_transformer=False,
                    multiplication_moe_in_transformer=False,
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

def generate_configs(env_id_list, action_space_size, collector_env_num, n_episode, evaluator_env_num, num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type, seed, buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition, num_segments, total_batch_size):
    configs = []
    exp_name_prefix = f'data_unizero_mt_ddp-2gpu_1201/finetune_pong/{len(env_id_list)}games_brf{buffer_reanalyze_freq}_1-encoder-{norm_type}-res2-channel256_gsl20_{len(env_id_list)}-pred-head_lsd768-nlayer8-nh24_mbs-512-bs64_upc80_seed{seed}/'

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
            num_segments,
            total_batch_size
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
    """
    Overview:
        This script should be executed with <nproc_per_node> GPUs.
        Run the following command to launch the script:
        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 ./zoo/atari/config/atari_unizero_multitask_segment_finetune_config.py
        torchrun --nproc_per_node=8 ./zoo/atari/config/atari_unizero_multitask_segment_finetune_config.py
    """

    from lzero.entry import train_unizero_multitask_segment
    from ding.utils import DDPContext
    from easydict import EasyDict

    env_id_list = ['PongNoFrameskip-v4']  # Debug setup
    action_space_size = 18

    # NCCL environment setup
    import os
    os.environ["NCCL_TIMEOUT"] = "3600000000"

    for seed in [0, 1, 2]:
        collector_env_num = 8
        num_segments = 8
        n_episode = 8
        evaluator_env_num = 3
        num_simulations = 50
        max_env_step = int(4e5)

        reanalyze_ratio = 0.0
        total_batch_size = 512
        batch_size = [int(min(64, total_batch_size / len(env_id_list))) for _ in range(len(env_id_list))]

        num_unroll_steps = 10
        infer_context_length = 4
        norm_type = 'LN'
        buffer_reanalyze_freq = 1 / 50
        reanalyze_batch_size = 160
        reanalyze_partition = 0.75

        configs = generate_configs(env_id_list, action_space_size, collector_env_num, n_episode, evaluator_env_num, num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type, seed, buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition, num_segments, total_batch_size)

        pretrained_model_path = '/mnt/afs/niuyazhe/code/LightZero/data_unizero_mt_ddp-8gpu_1127/8games_brf0.02_nlayer8-nhead24_seed1/8games_brf0.02_1-encoder-LN-res2-channel256_gsl20_8-pred-head_lsd768-nlayer8-nh24_mbs-512-bs64_upc80_seed1/Pong_unizero-mt_seed1/ckpt/iteration_200000.pth.tar'

        with DDPContext():
            train_unizero_multitask_segment(configs, seed=seed, model_path=pretrained_model_path, max_env_step=max_env_step)