from easydict import EasyDict
from copy import deepcopy

def create_config(env_id, action_space_size, collector_env_num, evaluator_env_num, n_episode, num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type):
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
        ),
        policy=dict(
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=200000))),
            grad_correct_params=dict(
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
                world_model_cfg=dict(
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    device='cuda',
                    action_space_size=action_space_size,
                    num_layers=2,
                    num_heads=8,
                    embed_dim=768,
                    obs_type='image',
                    env_num=8,
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
            update_per_collect=1000,
            replay_ratio=0.25,
            batch_size=batch_size,
            optim_type='AdamW',
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            n_episode=n_episode,
            replay_buffer_size=int(1e6),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
        ),
    ))

def generate_configs(env_id_list, action_space_size, collector_env_num, n_episode, evaluator_env_num, num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type, seed):
    configs = []
    exp_name_prefix = f'data_unizero_mt/{len(env_id_list)}games_1-encoder-{norm_type}_26-head_lsd768-nlayer2-nh8_max-bs2000_upc1000_seed{seed}/'

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
            norm_type
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
    from lzero.entry import train_unizero_multitask_serial

    env_id_list = [
        'PongNoFrameskip-v4',
        'MsPacmanNoFrameskip-v4',
        'SeaquestNoFrameskip-v4',
        'BoxingNoFrameskip-v4',
        'AlienNoFrameskip-v4',
        'AmidarNoFrameskip-v4',
        'AssaultNoFrameskip-v4',
        'AsterixNoFrameskip-v4',
        'BankHeistNoFrameskip-v4',
        'BattleZoneNoFrameskip-v4',
        'ChopperCommandNoFrameskip-v4',
        'CrazyClimberNoFrameskip-v4',
        'DemonAttackNoFrameskip-v4',
        'FreewayNoFrameskip-v4',
        'FrostbiteNoFrameskip-v4',
        'GopherNoFrameskip-v4',
        'HeroNoFrameskip-v4',
        'JamesbondNoFrameskip-v4',
        'KangarooNoFrameskip-v4',
        'KrullNoFrameskip-v4',
        'KungFuMasterNoFrameskip-v4',
        'PrivateEyeNoFrameskip-v4',
        'RoadRunnerNoFrameskip-v4',
        'UpNDownNoFrameskip-v4',
        'QbertNoFrameskip-v4',
        'BreakoutNoFrameskip-v4',
    ]

    action_space_size = 18
    seed = 0
    collector_env_num = 8
    n_episode = 8
    evaluator_env_num = 3
    num_simulations = 50
    max_env_step = int(1e6)
    reanalyze_ratio = 0.0
    max_batch_size = 1000
    batch_size = [int(max_batch_size / len(env_id_list)) for _ in range(len(env_id_list))]
    num_unroll_steps = 10
    infer_context_length = 4
    norm_type = 'LN'

    configs = generate_configs(env_id_list, action_space_size, collector_env_num, n_episode, evaluator_env_num, num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type, seed)

    # train_unizero_multitask_serial(configs[:4], seed=seed, max_env_step=max_env_step) # multitask learning on first four tasks
    train_unizero_multitask_serial(configs, seed=seed, max_env_step=max_env_step)