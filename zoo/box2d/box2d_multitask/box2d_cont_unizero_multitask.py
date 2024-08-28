from easydict import EasyDict
from copy import deepcopy
import torch
def create_config(env_id, observation_shapes, action_space_sizes, collector_env_num, evaluator_env_num, n_episode, num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type):
    return EasyDict(dict(
        env=dict(
            stop_value=int(1e6),
            env_id=env_id,
            continuous=True,
            manually_discretization=False,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
        ),
        policy=dict(
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=200000,),),),  # default is 10000
            grad_correct_params=dict(
                # for MoCo
                MoCo_beta=0.5,
                MoCo_beta_sigma=0.5,
                MoCo_gamma=0.1,
                MoCo_gamma_sigma=0.5,
                MoCo_rho=0,
                # for CAGrad
                calpha=0.5,
                rescale=1,
            ),
            task_num=len(env_id_list),
            task_id=0,
            model=dict(
                observation_shapes=observation_shapes,
                action_space_sizes=action_space_sizes,
                continuous_action_space=True,
                num_of_sampled_actions=20,
                model_type='mlp',
                world_model_cfg=dict(
                    obs_type='vector',
                    num_unroll_steps=num_unroll_steps,
                    policy_entropy_loss_weight=1e-4,
                    continuous_action_space=True,
                    num_of_sampled_actions=20,
                    sigma_type='conditioned',
                    norm_type=norm_type,
                    bound_type=None,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    action_space_size=action_space_sizes,
                    env_num=max(collector_env_num, evaluator_env_num),
                    task_num=len(env_id_list),
                    use_normal_head=True,
                    use_softmoe_head=False,
                    use_moe_head=False,
                    num_experts_in_moe_head=4,  # NOTE
                    moe_in_transformer=False,  # NOTE
                    multiplication_moe_in_transformer=False,  # NOTE
                    num_experts_of_moe_in_transformer=4,
                ),
            ),
            use_priority=True,
            print_task_priority_logs=False,
            cuda=True,
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            replay_ratio=0.25,
            batch_size=batch_size,
            optim_type='AdamW',
            learning_rate=1e-4,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            n_episode=n_episode,
            eval_freq=int(2e3),
            replay_buffer_size=int(1e6),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
        ),
    ))

def generate_configs(env_id_list, observation_shapes, action_space_sizes, collector_env_num, n_episode, evaluator_env_num, num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type, seed):
    configs = []
    exp_name_prefix = f'data_unizero_mt_box2d/{len(env_id_list)}games_cont_action_seed{seed}/'

    for task_id, (env_id, observation_shape, action_space_size) in enumerate(zip(env_id_list, observation_shapes, action_space_sizes)):
        config = create_config(
            env_id,
            observation_shapes, # TODO
            action_space_sizes,
            collector_env_num,
            evaluator_env_num,
            n_episode,
            num_simulations,
            reanalyze_ratio,
            batch_size,
            num_unroll_steps,
            infer_context_length,
            norm_type
        )
        config.policy.task_id = task_id
        config.exp_name = exp_name_prefix + f"{env_id.split('-v')[0]}_unizero_mt_seed{seed}"

        configs.append([task_id, [config, create_env_manager(env_name=env_id)]])
    return configs

def create_env_manager():
    return EasyDict(dict(
        env=dict(
            type='box2d',
            import_names=['zoo.box2d.lunarlander.envs.lunarlander_env', 'zoo.box2d.bipedalwalker.envs.bipedalwalker_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(
            type='sampled_unizero_multitask',
            import_names=['lzero.policy.sampled_unizero_multitask'],
        ),
    ))

def create_env_manager(env_name: str):
    if env_name == 'LunarLanderContinuous-v2':
        return EasyDict(dict(
            env=dict(
                type='lunarlander',
                import_names=[f'zoo.box2d.lunarlander.envs.lunarlander_env'],
            ),
            env_manager=dict(type='subprocess'),
            policy=dict(
                type='sampled_unizero_multitask',
                import_names=['lzero.policy.sampled_unizero_multitask'],
            ),
        ))
    elif env_name == 'BipedalWalker-v3':
        return EasyDict(dict(
            env=dict(
                type='bipedalwalker',
                import_names=[f'zoo.box2d.bipedalwalker.envs.bipedalwalker_env'],
            ),
            env_manager=dict(type='subprocess'),
            policy=dict(
                type='sampled_unizero_multitask',
                import_names=['lzero.policy.sampled_unizero_multitask'],
            ),
        ))

if __name__ == "__main__":
    from lzero.entry import train_unizero_multitask

    env_id_list = [
        'LunarLanderContinuous-v2',
        'BipedalWalker-v3',
    ]

    observation_shapes = [
        8,  # LunarLanderContinuous-v2
        24, # BipedalWalker-v3
    ]

    action_space_sizes = [
        2,  # LunarLanderContinuous-v2
        4,  # BipedalWalker-v3
    ]

    seed = 0
    collector_env_num = 8
    n_episode = 8
    evaluator_env_num = 3
    num_simulations = 50
    max_env_step = int(1e6)
    reanalyze_ratio = 0.
    max_batch_size = 1000
    batch_size = [int(max_batch_size/len(env_id_list)) for i in range(len(env_id_list))]
    num_unroll_steps = 10
    infer_context_length = 4
    norm_type = 'LN'

    configs = generate_configs(env_id_list, observation_shapes, action_space_sizes, collector_env_num, n_episode, evaluator_env_num, num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type, seed)

    train_unizero_multitask(configs, seed=seed, max_env_step=max_env_step)