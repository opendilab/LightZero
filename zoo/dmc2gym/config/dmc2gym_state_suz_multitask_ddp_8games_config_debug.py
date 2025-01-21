from easydict import EasyDict
from typing import List

def create_config(env_id, observation_shape_list, action_space_size_list, collector_env_num, evaluator_env_num, n_episode,
                 num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length,
                 norm_type, buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition, num_segments,
                 total_batch_size):
    domain_name = env_id.split('-')[0]
    task_name = env_id.split('-')[1]
    return EasyDict(dict(
        env=dict(
            stop_value=int(5e5),
            env_id=env_id,
            domain_name=domain_name,
            task_name=task_name,
            observation_shape_list=observation_shape_list,
            action_space_size_list=action_space_size_list,
            from_pixels=False,
            # ===== TODO: only for debug =====
            # frame_skip=100, # 10
            frame_skip=2,
            continuous=True,  # Assuming all DMC tasks use continuous action spaces
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False),
            game_segment_length=100,  # As per single-task config
            # ===== only for debug =====
            # collect_max_episode_steps=int(20),
            # eval_max_episode_steps=int(20),
        ),
        policy=dict(
            multi_gpu=True,  # TODO: enable multi-GPU for DDP
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000000))),
            grad_correct_params=dict(
                # Example gradient correction parameters, adjust as needed
                MoCo_beta=0.5, MoCo_beta_sigma=0.5, MoCo_gamma=0.1, MoCo_gamma_sigma=0.5, MoCo_rho=0,
                calpha=0.5, rescale=1,
            ),
            # use_moco=True,  # ==============TODO==============
            use_moco=False,  # ==============TODO==============
            task_num=len(env_id_list),
            task_id=0,  # To be set per task
            model=dict(
                observation_shape_list=observation_shape_list,
                action_space_size_list=action_space_size_list,
                continuous_action_space=True,
                num_of_sampled_actions=20,
                model_type='mlp',
                world_model_cfg=dict(
                    observation_shape_list=observation_shape_list,
                    action_space_size_list=action_space_size_list,
                    policy_loss_type='kl',
                    obs_type='vector',
                    # use_shared_projection=True, # TODO
                    use_shared_projection=False,
                    # task_embed_option='concat_task_embed',   # ==============TODO: none ==============
                    task_embed_option='register_task_embed',   # ==============TODO: none ==============
                    # task_embed_option=None,   # ==============TODO: none ==============
                    register_token_num=4,
                    use_task_embed=True, # TODO
                    # use_task_embed=False, # ==============TODO==============
                    num_unroll_steps=num_unroll_steps,
                    policy_entropy_weight=5e-2,
                    continuous_action_space=True,
                    num_of_sampled_actions=20,
                    sigma_type='conditioned',
                    fixed_sigma_value=0.5,
                    bound_type=None,
                    model_type='mlp',
                    norm_type=norm_type,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,  # Each timestep has 2 tokens: obs and action
                    context_length=2 * infer_context_length,
                    device='cuda',
                    # device='cpu', # TODO
                    # num_layers=2,
                    # num_layers=4, # TODO

                    num_layers=8, # TODO
                    num_heads=8,

                    # num_layers=12, # TODO
                    # num_heads=12,

                    embed_dim=768,
                    env_num=max(collector_env_num, evaluator_env_num),
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
            # use_task_exploitation_weight=True, # TODO
            use_task_exploitation_weight=False, # TODO
            # task_complexity_weight=True, # TODO
            task_complexity_weight=False, # TODO
            total_batch_size=total_batch_size,
            allocated_batch_sizes=False,
            # train_start_after_envsteps=int(2e3),
            train_start_after_envsteps=int(0),
            use_priority=False,
            print_task_priority_logs=False,
            cuda=True,
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            # update_per_collect=2,  # TODO: 80
            update_per_collect=200,  # TODO: 8*100*0.25=200 
            # update_per_collect=80,  # TODO: 8*100*0.1=80
            replay_ratio=reanalyze_ratio,
            batch_size=batch_size,
            optim_type='AdamW',
            num_segments=num_segments,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            n_episode=n_episode,
            replay_buffer_size=int(5e5),
            # eval_freq=int(5e3),
            eval_freq=int(4e3),
            # eval_freq=int(2e3),
            # eval_freq=int(1e3), # TODO: task_weight===========
            grad_clip_value=5,
            learning_rate=1e-4,
            discount_factor=0.99,
            td_steps=5,
            piecewise_decay_lr_scheduler=False,
            manual_temperature_decay=True,
            threshold_training_steps_for_final_temperature=int(2.5e4),
            cos_lr_scheduler=True,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
        ),
    ))


def generate_configs(env_id_list: List[str],
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
                    total_batch_size: int):
    configs = []
    # TODO: debug
    exp_name_prefix = f'data_suz_mt_20250113_debug/ddp_4gpu_nlayer8_upc200_no-taskweight-obsloss-temp1_concat-task-embed_{len(env_id_list)}tasks_brf{buffer_reanalyze_freq}_tbs{total_batch_size}_seed{seed}/'

    # exp_name_prefix = f'data_suz_mt_20250113/ddp_8gpu_nlayer8_upc200_taskweight-eval1e3-10k-temp10-1_task-embed_{len(env_id_list)}tasks_brf{buffer_reanalyze_freq}_tbs{total_batch_size}_seed{seed}/'
    # exp_name_prefix = f'data_suz_mt_20250113_debug/ddp_8gpu_nlayer8_upc200_taskweight-eval1e3-10k-temp10-1_no-task-embed_{len(env_id_list)}tasks_brf{buffer_reanalyze_freq}_tbs{total_batch_size}_seed{seed}/'

    # exp_name_prefix = f'data_suz_mt_20250113/ddp_3gpu_3games_nlayer8_upc200_notusp_notaskweight-symlog-01-05-eval1e3_{len(env_id_list)}tasks_brf{buffer_reanalyze_freq}_tbs{total_batch_size}_seed{seed}/'

    action_space_size_list = [dmc_state_env_action_space_map[env_id] for env_id in env_id_list]
    observation_shape_list = [dmc_state_env_obs_space_map[env_id] for env_id in env_id_list]

    for task_id, (env_id, obs_shape, act_space) in enumerate(zip(env_id_list, observation_shape_list, action_space_size_list)):
        config = create_config(
            env_id=env_id,
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


def create_env_manager():
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
        This script should be executed with <nproc_per_node> GPUs.
        Run the following command to launch the script:
        python -m torch.distributed.launch --nproc_per_node=7 --master_port=29503 ./zoo/dmc2gym/config/dmc2gym_state_suz_multitask_ddp_8games_config.py

        python -m torch.distributed.launch --nproc_per_node=1 --master_port=29503 ./zoo/dmc2gym/config/dmc2gym_state_suz_multitask_ddp_8games_config_debug.py
        torchrun --nproc_per_node=8 ./zoo/dmc2gym/config/dmc2gym_state_suz_multitask_ddp_config.py
    """

    from lzero.entry import train_unizero_multitask_segment_ddp
    from ding.utils import DDPContext
    import os
    from zoo.dmc2gym.config.dmc_state_env_space_map import dmc_state_env_action_space_map, dmc_state_env_obs_space_map

    # os.environ["NCCL_TIMEOUT"] = "3600000000"

    # 定义环境列表
    env_id_list = [
        # 'acrobot-swingup', # 6 1
        'cartpole-swingup', # 5 1
    ]



    # env_id_list = [
    #     # 'acrobot-swingup',
    #     # 'cartpole-balance',
    #     # 'cartpole-balance_sparse',
    #     # 'cartpole-swingup',
    #     'cartpole-swingup_sparse',
    #     'cheetah-run',
    #     # "ball_in_cup-catch",
    #     "finger-spin",
    # ]

    # env_id_list = [
    #     # 'acrobot-swingup',
    #     # 'cartpole-balance',
    #     # 'cartpole-balance_sparse',
    #     # 'cartpole-swingup',
    #     # 'cartpole-swingup_sparse',
    #     # 'cheetah-run',
    #     # "ball_in_cup-catch",
    #     "finger-spin",
    # ]

    # DMC 8games
    # env_id_list = [
    #     'acrobot-swingup',
    #     'cartpole-balance',
    #     'cartpole-balance_sparse',
    #     'cartpole-swingup',
    #     'cartpole-swingup_sparse',
    #     'cheetah-run',
    #     "ball_in_cup-catch",
    #     "finger-spin",
    # ]

    # DMC 18games
    # env_id_list = [
    #     'acrobot-swingup',
    #     'cartpole-balance',
    #     'cartpole-balance_sparse',
    #     'cartpole-swingup',
    #     'cartpole-swingup_sparse',
    #     'cheetah-run',
    #     "ball_in_cup-catch",
    #     "finger-spin",
    #     "finger-turn_easy",
    #     "finger-turn_hard",
    #     'hopper-hop',
    #     'hopper-stand',
    #     'pendulum-swingup',
    #     'reacher-easy',
    #     'reacher-hard',
    #     'walker-run',
    #     'walker-stand',
    #     'walker-walk',
    # ]

    # 获取各环境的 action_space_size 和 observation_shape
    action_space_size_list = [dmc_state_env_action_space_map[env_id] for env_id in env_id_list]
    observation_shape_list = [dmc_state_env_obs_space_map[env_id] for env_id in env_id_list]

    collector_env_num = 8
    num_segments = 8
    n_episode = 8
    evaluator_env_num = 3
    num_simulations = 50
    # max_env_step = int(5e5)
    max_env_step = int(1e6)
    reanalyze_ratio = 0.0

    # nlayer=4
    total_batch_size = 512
    batch_size = [int(min(64, total_batch_size / len(env_id_list))) for _ in range(len(env_id_list))]

    # nlayer=8/12
    total_batch_size = 512
    batch_size = [int(min(64, total_batch_size / len(env_id_list))) for _ in range(len(env_id_list))]

    num_unroll_steps = 5
    # infer_context_length = 3
    infer_context_length = 4 # 尾部有4个register token, 相当于infer_context_length还是2
    norm_type = 'LN'
    buffer_reanalyze_freq = 1 / 100000
    reanalyze_batch_size = 160
    reanalyze_partition = 0.75

    # ======== TODO: only for debug ========
    collector_env_num = 2
    num_segments = 2
    n_episode = 2
    evaluator_env_num = 2
    num_simulations = 50
    batch_size = [2 for _ in range(len(env_id_list))]
    # =======================================

    seed = 0  # You can iterate over multiple seeds if needed

    configs = generate_configs(
        env_id_list=env_id_list,
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
    )

    with DDPContext():
        train_unizero_multitask_segment_ddp(configs, seed=seed, max_env_step=max_env_step)
        # 如果只想训练部分任务，可以修改 configs，例如:
        # train_unizero_multitask_segment_ddp(configs[:4], seed=seed, max_env_step=max_env_step)