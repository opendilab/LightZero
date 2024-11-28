# zoo/atari/config/atari_muzero_multitask_segment_8games_ddp_config.py

from easydict import EasyDict
import os

def create_config(env_id, action_space_size, collector_env_num, evaluator_env_num, n_episode,
                 num_simulations, reanalyze_ratio, batch_size, num_unroll_steps,
                 infer_context_length, norm_type, buffer_reanalyze_freq,
                 reanalyze_batch_size, reanalyze_partition, num_segments, total_batch_size):
    return EasyDict(dict(
        env=dict(
            stop_value=int(1e6),
            env_id=env_id,
            observation_shape=(4, 96, 96),  # MuZero typically uses frame stacking
            frame_stack_num=4,
            gray_scale=True,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False,),
            full_action_space=True,
            collect_max_episode_steps=int(5e3),
            eval_max_episode_steps=int(5e3),

            # ===== only for debug =====
            # collect_max_episode_steps=int(30),
            # eval_max_episode_steps=int(30),
        ),
        policy=dict(
            type='muzero_multitask',
            import_names=['lzero.policy.muzero_multitask'],
            multi_gpu=True,  # Essential for DDP
            learn=dict(
                learner=dict(
                    hook=dict(save_ckpt_after_iter=50000,),
                ),
            ),
            grad_correct_params=dict(
                # Parameters for gradient correction techniques, if any
                # Modify or extend based on MuZero's requirements
                alpha=0.5,
                beta=0.1,
            ),
            task_num=len(env_id_list),  # Total number of tasks
            task_id=0,
            model=dict(
                observation_shape=(4, 96, 96),
                frame_stack_num=4,
                gray_scale=True,
                action_space_size=action_space_size,
                norm_type=norm_type,
                model_type='conv',  # MuZero typically uses convolutional networks
                use_sim_norm=True,  # Simulation normalization
                use_sim_norm_kl_loss=False,
                downsample=True,
                self_supervised_learning_loss=True,
                discrete_action_encoding_type='one_hot',

                # Architecture specifics
                num_layers=12,
                num_heads=12,
                embed_dim=768,

                # Transformer or other layers can be added here if MuZero uses them
                world_model_cfg=dict(
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    device='cuda',
                    action_space_size=action_space_size,
                    num_layers=12,
                    num_heads=12,
                    embed_dim=768,

                    obs_type='image',
                    env_num=8,  # Should match the maximum number of environments across tasks
                    task_num=len(env_id_list),
                    use_normal_head=True,
                    use_softmoe_head=False,
                    use_moe_head=False,
                    num_experts_in_moe_head=4,
                    moe_in_transformer=False,
                    multiplication_moe_in_transformer=False,
                    num_experts_of_moe_in_transformer=4,
                ),
            ),
            total_batch_size=total_batch_size,
            allocated_batch_sizes=False,
            train_start_after_envsteps=int(2000),
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

            # Reanalyze settings
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
        ),
    ))

def generate_configs(env_id_list, action_space_size, collector_env_num, n_episode,
                    evaluator_env_num, num_simulations, reanalyze_ratio,
                    batch_size, num_unroll_steps, infer_context_length,
                    norm_type, seed, buffer_reanalyze_freq,
                    reanalyze_batch_size, reanalyze_partition,
                    num_segments, total_batch_size):
    configs = []
    exp_name_prefix = f'data_muzero_mt_ddp_8gpu/{len(env_id_list)}games_brf{buffer_reanalyze_freq}_nlayer12-nhead12_seed{seed}/' \
                      f'{len(env_id_list)}games_brf{buffer_reanalyze_freq}_1-encoder-{norm_type}-conv_lsd768_nlayer12_nh12_bs{total_batch_size}_upc80_seed{seed}/'

    for task_id, env_id in enumerate(env_id_list):
        config = create_config(
            env_id=env_id,
            action_space_size=action_space_size,
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
            total_batch_size=total_batch_size
        )
        config.policy.task_id = task_id
        config.exp_name = exp_name_prefix + f"{env_id.split('NoFrameskip')[0]}_muzero-mt_seed{seed}"

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
            type='muzero_multitask',
            import_names=['lzero.policy.muzero_multitask'],
        ),
    ))

if __name__ == "__main__":
    from lzero.entry import train_muzero_multitask_segment_ddp
    from ding.utils import DDPContext

    # Define your environment list here
    env_id_list = [
        'PongNoFrameskip-v4',
        'MsPacmanNoFrameskip-v4',
        'SeaquestNoFrameskip-v4',
        'BoxingNoFrameskip-v4',
        'AlienNoFrameskip-v4',
        'ChopperCommandNoFrameskip-v4',
        'HeroNoFrameskip-v4',
        'RoadRunnerNoFrameskip-v4',
    ]

    action_space_size = 18  # Full action space

    # Set NCCL environment variables for DDP
    os.environ["NCCL_TIMEOUT"] = "3600000000"

    for seed in [0, 1, 2, 3]:
        collector_env_num = 8
        num_segments = 8
        n_episode = 8
        evaluator_env_num = 3
        num_simulations = 50
        max_env_step = int(5e5)  # Maximum environment steps

        reanalyze_ratio = 0.0

        # Determine batch sizes based on total_batch_size and number of environments
        total_batch_size = 512
        batch_size = [int(min(64, total_batch_size / len(env_id_list))) for _ in range(len(env_id_list))]
        print(f'=========== batch_size: {batch_size} ===========')

        num_unroll_steps = 10
        infer_context_length = 4
        norm_type = 'LN'

        # Reanalyze configurations
        buffer_reanalyze_freq = 1/50
        reanalyze_batch_size = 160
        reanalyze_partition = 0.75

        configs = generate_configs(
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

        """
        Overview:
            This script should be executed with <nproc_per_node> GPUs.
            Run the following command to launch the script:
            export NCCL_TIMEOUT=3600000
            python -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 ./zoo/atari/config/atari_muzero_multitask_segment_8games_ddp_config.py
            或者使用 torchrun:
            torchrun --nproc_per_node=8 ./zoo/atari/config/atari_muzero_multitask_segment_8games_ddp_config.py
        """
        with DDPContext():
            train_muzero_multitask_segment_ddp(configs, seed=seed, max_env_step=max_env_step)