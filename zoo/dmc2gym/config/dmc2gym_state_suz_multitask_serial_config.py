from dizoo.classic_control.pendulum.config.pendulum_ibc_config import multi_gpu
from easydict import EasyDict
from copy import deepcopy

from zoo.dmc2gym.config.dmc_state_env_space_map import dmc_state_env_action_space_map, dmc_state_env_obs_space_map


def create_config(env_id, action_space_size_list, observation_shape_list, collector_env_num, evaluator_env_num,
                  n_episode, num_simulations, batch_size, num_unroll_steps, infer_context_length,
                  norm_type, buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition, num_segments, seed, update_per_collect):
    """
    Create a multi-task configuration for DMC environments.
    """
    domain_name = env_id.split('-')[0]
    task_name = env_id.split('-')[1]
    return EasyDict(dict(
        env=dict(
            stop_value=int(5e5),
            domain_name=domain_name,
            task_name=task_name,
            observation_shape_list=observation_shape_list,
            action_space_size_list=action_space_size_list,
            from_pixels=False,
            frame_skip=2,
            continuous=True,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False),
            save_replay_gif=False,
            replay_path_gif='./replay_gif',
        ),
        policy=dict(
            multi_gpu=False, # TODO: nable multi-GPU for DDP
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=200000))),
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
                    max_tokens=2 * num_unroll_steps,  # 每个时间步有2个token: obs 和 action
                    context_length=2 * infer_context_length,
                    device='cuda',
                    # device='cpu', # TODO
                    # num_layers=2,
                    num_layers=4, # TODO
                    num_heads=8,
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
            use_priority=False,
            print_task_priority_logs=False,
            cuda=True,
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            game_segment_length=100,
            update_per_collect=update_per_collect, # TODO
            replay_ratio=0.25, # TODO
            batch_size=batch_size,
            optim_type='AdamW',
            num_segments=num_segments,
            num_simulations=num_simulations,
            n_episode=n_episode,
            replay_buffer_size=int(1e6),
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
        seed=seed,
    ))


def generate_configs(env_id_list, seed, collector_env_num, evaluator_env_num, n_episode, num_simulations,
                     batch_size, num_unroll_steps, infer_context_length, norm_type,
                     buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition, num_segments, update_per_collect):
    """
    Generate configurations for all DMC tasks in the environment list.
    """
    configs = []
    exp_name_prefix = f'data_suz_mt_20241224/{len(env_id_list)}tasks_brf{buffer_reanalyze_freq}/'
    action_space_size_list = [dmc_state_env_action_space_map[env_id] for env_id in env_id_list]
    observation_shape_list = [dmc_state_env_obs_space_map[env_id] for env_id in env_id_list]
    for task_id, env_id in enumerate(env_id_list):
        config = create_config(
            env_id,
            action_space_size_list,
            observation_shape_list,
            collector_env_num,
            evaluator_env_num,
            n_episode,
            num_simulations,
            batch_size,
            num_unroll_steps,
            infer_context_length,
            norm_type,
            buffer_reanalyze_freq,
            reanalyze_batch_size,
            reanalyze_partition,
            num_segments,
            seed,
            update_per_collect
        )

        # 设置多任务相关的配置
        config.policy.task_num = len(env_id_list)
        config.policy.task_id = task_id

        # 生成实验名称前缀
        config.exp_name = exp_name_prefix + f'mt_unizero_seed{seed}'
        configs.append([task_id, [config, create_env_manager()]])
    return configs
    # return [[i, [deepcopy(config), create_env_manager()]] for i in range(len(env_id_list))]


def create_env_manager():
    """
    Create the environment manager configuration.
    """
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
    from lzero.entry import train_unizero_multitask_segment_serial

    import argparse

    parser = argparse.ArgumentParser(description='Train multi-task DMC Unizero model.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()

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
    
    # DMC 18games
    env_id_list = [
        'acrobot-swingup',
        'cartpole-balance',
        'cartpole-balance_sparse',
        'cartpole-swingup',
        'cartpole-swingup_sparse',
        'cheetah-run',
        "ball_in_cup-catch",
        "finger-spin",
        "finger-turn_easy",
        "finger-turn_hard",
        'hopper-hop',
        'hopper-stand',
        'pendulum-swingup',
        # 'quadruped-run',
        # 'quadruped-walk',
        'reacher-easy',
        'reacher-hard',
        'walker-run',
        'walker-stand',
        'walker-walk',
        # 'humanoid-run',
    ]

    # 获取各环境的 action_space_size 和 observation_shape
    action_space_size_list = [dmc_state_env_action_space_map[env_id] for env_id in env_id_list]
    observation_shape_list = [dmc_state_env_obs_space_map[env_id] for env_id in env_id_list]

    # 定义关键参数
    seed = args.seed
    collector_env_num = 8
    evaluator_env_num = 3
    num_segments = 8
    n_episode = 8
    num_simulations = 50
    batch_size = [64 for _ in range(len(env_id_list))]
    num_unroll_steps = 5
    infer_context_length = 2
    norm_type = 'LN'
    buffer_reanalyze_freq = 1 / 100000
    reanalyze_batch_size = 160
    reanalyze_partition = 0.75
    max_env_step = int(5e5)
    update_per_collect = 100

    # ========== TODO: debug config ============
    # collector_env_num = 2
    # evaluator_env_num = 2
    # num_segments = 2
    # n_episode = 2
    # num_simulations = 2
    # batch_size = [4,4]  # 可以根据需要调整或者设置为列表
    # update_per_collect = 1

    # 生成配置
    configs = generate_configs(
        env_id_list=env_id_list,
        seed=seed,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_episode=n_episode,
        num_simulations=num_simulations,
        batch_size=batch_size,
        num_unroll_steps=num_unroll_steps,
        infer_context_length=infer_context_length,
        norm_type=norm_type,
        buffer_reanalyze_freq=buffer_reanalyze_freq,
        reanalyze_batch_size=reanalyze_batch_size,
        reanalyze_partition=reanalyze_partition,
        num_segments=num_segments,
        update_per_collect=update_per_collect
    )

    # 启动多任务训练
    train_unizero_multitask_segment_serial(configs, seed=seed, max_env_step=max_env_step)