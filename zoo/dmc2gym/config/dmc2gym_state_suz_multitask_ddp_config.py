from easydict import EasyDict
from typing import List

def create_config(env_id, observation_shape, action_space_size, collector_env_num, evaluator_env_num, n_episode,
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
            multi_gpu=True,  # TODO: nable multi-GPU for DDP
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000000))),
            grad_correct_params=dict(
                # Example gradient correction parameters, adjust as needed
                MoCo_beta=0.5, MoCo_beta_sigma=0.5, MoCo_gamma=0.1, MoCo_gamma_sigma=0.5, MoCo_rho=0,
                calpha=0.5, rescale=1,
            ),
            task_num=len(env_id),  # Number of tasks
            task_id=0,  # To be set per task
            model=dict(
                observation_shape_list=observation_shape_list,
                action_space_size_list=action_space_size_list,
                norm_type=norm_type,
                num_layers=2,
                num_unroll_steps=num_unroll_steps,
                infer_context_length=infer_context_length,
                world_model_cfg=dict(
                    observation_shape_list=observation_shape_list,
                    action_space_size_list=action_space_size_list,
                    policy_loss_type='kl',
                    obs_type='vector',
                    num_unroll_steps=num_unroll_steps,
                    policy_entropy_weight=5e-2,
                    continuous_action_space=True,
                    num_of_sampled_actions=batch_size,  # Adjust as per batch_size
                    sigma_type='conditioned',
                    fixed_sigma_value=0.5,
                    bound_type=None,
                    model_type='mlp',
                    norm_type=norm_type,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,  # Each timestep has 2 tokens: obs and action
                    context_length=2 * infer_context_length,
                    # device='cuda',
                    device='cpu', # TODO
                    action_space_size=action_space_size,
                    num_layers=num_unroll_steps,  # Adjust if different
                    num_heads=8,
                    embed_dim=768,
                    env_num=max(collector_env_num, evaluator_env_num),
                ),
            ),
            total_batch_size=total_batch_size,
            allocated_batch_sizes=False,
            train_start_after_envsteps=int(2e3),
            use_priority=False,
            print_task_priority_logs=False,
            cuda=True,
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            update_per_collect=80,  # TODO
            replay_ratio=reanalyze_ratio,
            batch_size=batch_size,
            optim_type='AdamW',
            num_segments=num_segments,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            n_episode=n_episode,
            replay_buffer_size=int(1e6),
            eval_freq=int(5e3),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
        ),
    ))


def generate_configs(env_id_list: List[str],
                    observation_shape_list: List[int],
                    action_space_size_list: List[int],
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
    exp_name_prefix = f'data_suz_mt_ddp/{len(env_id_list)}tasks_brf{buffer_reanalyze_freq}_seed{seed}/'

    for task_id, (env_id, obs_shape, act_space) in enumerate(zip(env_id_list, observation_shape_list, action_space_size_list)):
        config = create_config(
            env_id=env_id,
            observation_shape=obs_shape,
            action_space_size=act_space,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_episode=n_episode,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            batch_size=batch_size[task_id],
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
            type='unizero_multitask',
            import_names=['lzero.policy.unizero_multitask'],
        ),
    ))


if __name__ == "__main__":
    """
    Overview:
        This script should be executed with <nproc_per_node> GPUs.
        Run the following command to launch the script:
        python -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 ./zoo/dmc2gym/config/dmc2gym_multitask_segment_ddp_config.py
        torchrun --nproc_per_node=8 ./zoo/dmc2gym/config/dmc2gym_multitask_segment_ddp_config.py
    """

    from lzero.entry import train_unizero_multitask_segment_ddp
    from ding.utils import DDPContext
    import os
    from zoo.dmc2gym.config.dmc_state_env_space_map import dmc_state_env_action_space_map, dmc_state_env_obs_space_map

    os.environ["NCCL_TIMEOUT"] = "3600000000"

    # 定义环境列表
    env_id_list = [
        'cartpole-swingup',
        'cartpole-balance',
        # 'cheetah-run',
        # 'walker-walk',
        # 'hopper-hop',
        # 'humanoid-walk',
        # 'quadruped-run',
        # 'finger-spin',
    ]

    # 获取各环境的 action_space_size 和 observation_shape
    action_space_size_list = [dmc_state_env_action_space_map[env_id] for env_id in env_id_list]
    observation_shape_list = [dmc_state_env_obs_space_map[env_id] for env_id in env_id_list]


    collector_env_num = 8
    num_segments = 8
    n_episode = 8
    evaluator_env_num = 3
    num_simulations = 50
    max_env_step = int(5e5)
    reanalyze_ratio = 0.0
    total_batch_size = 512
    batch_size = [int(min(64, total_batch_size / len(env_id_list))) for _ in range(len(env_id_list))]
    num_unroll_steps = 5
    infer_context_length = 2
    norm_type = 'LN'
    buffer_reanalyze_freq = 1 / 100000
    reanalyze_batch_size = 160
    reanalyze_partition = 0.75

    # ======== TODO: only for debug ========
    # collector_env_num = 2
    # num_segments = 2
    # n_episode = 2
    # evaluator_env_num = 2
    # num_simulations = 2
    # batch_size = [4, 4, 4, 4, 4, 4, 4, 4]
    # =======================================

    seed = 0  # You can iterate over multiple seeds if needed

    configs = generate_configs(
        env_id_list=env_id_list,
        observation_shape_list=observation_shape_list,
        action_space_size_list=action_space_size_list,
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