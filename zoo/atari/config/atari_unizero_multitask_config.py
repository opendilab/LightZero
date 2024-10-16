from easydict import EasyDict
from copy import deepcopy
# from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map

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
            manager=dict(shared_memory=False, ),
            full_action_space=True,
            # ===== only for debug =====
            # collect_max_episode_steps=int(30),
            # eval_max_episode_steps=int(30),
            # collect_max_episode_steps=int(50),
            # eval_max_episode_steps=int(50),
            # collect_max_episode_steps=int(500),
            # eval_max_episode_steps=int(500),
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
                observation_shape=(3, 64, 64),
                action_space_size=action_space_size,
                norm_type=norm_type,
                world_model_cfg=dict(
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    # device='cpu',  # 'cuda',
                    device='cuda',  # 'cuda',
                    action_space_size=action_space_size,
                    # num_layers=4,  # NOTE
                    num_layers=2,  # NOTE
                    num_heads=8,
                    embed_dim=768,
                    obs_type='image',
                    # env_num=max(collector_env_num, evaluator_env_num),
                    env_num=8,  # TODO: the max of all tasks
                    # collector_env_num=collector_env_num,
                    # evaluator_env_num=evaluator_env_num,
                    task_num=len(env_id_list),
                    use_normal_head=True,
                    # use_normal_head=False,
                    use_softmoe_head=False,
                    # use_moe_head=True,
                    use_moe_head=False,
                    num_experts_in_moe_head=4,  # NOTE
                    # moe_in_transformer=True,
                    moe_in_transformer=False,  # NOTE
                    # multiplication_moe_in_transformer=True,
                    multiplication_moe_in_transformer=False,  # NOTE
                    num_experts_of_moe_in_transformer=4,
                    # num_experts_of_moe_in_transformer=2,
                ),
            ),
            use_priority=False,
            # print_task_priority_logs=False,
            # use_priority=True,  # TODO
            print_task_priority_logs=False,
            cuda=True,
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            # update_per_collect=None,
            update_per_collect=1000,
            # update_per_collect=500,
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
    # TODO
    # exp_name_prefix = f'data_unizero_mt_0711/{len(env_id_list)}games_{"-".join(env_id_list)}_1-head-softmoe4_1-encoder-{norm_type}_lsd768-nlayer4-nh8_seed{seed}/'
    # exp_name_prefix = f'data_unizero_mt_0716/{len(env_id_list)}games_4-head_1-encoder-{norm_type}_MoCo_lsd768-nlayer4-nh8_max-bs1500_seed{seed}/'
    # exp_name_prefix = f'data_unizero_mt_0719/{len(env_id_list)}games_pong-boxing-envnum2_4-head_1-encoder-{norm_type}_trans-ffw-moe4_lsd768-nlayer2-nh8_max-bs1500_upc1000_seed{seed}/'
    # exp_name_prefix = f'data_unizero_mt_0716/{len(env_id_list)}games_1-head_1-encoder-{norm_type}_trans-ffw-moe4_lsd768-nlayer4-nh8_max-bs1500_seed{seed}/'
    # exp_name_prefix = f'data_unizero_mt_0722_debug/{len(env_id_list)}games_1-encoder-{norm_type}_trans-ffw-moeV2-expert4_4-head_lsd768-nlayer2-nh8_max-bs2000_upc1000_seed{seed}/'
    # exp_name_prefix = f'data_unizero_mt_0722_profile/lineprofile_{len(env_id_list)}games_1-encoder-{norm_type}_4-head_lsd768-nlayer2-nh8_max-bs2000_upc1000_seed{seed}/'
    exp_name_prefix = f'data_unizero_mt_1016_origcollect/{len(env_id_list)}games_1-encoder-{norm_type}_4-head_lsd768-nlayer2-nh8_max-bs2000_upc1000_seed{seed}/'

    for task_id, env_id in enumerate(env_id_list):
        config = create_config(
            env_id,
            action_space_size,
            # collector_env_num if env_id not in ['PongNoFrameskip-v4', 'BoxingNoFrameskip-v4'] else 2,  # TODO: different collector_env_num for Pong and Boxing
            # evaluator_env_num if env_id not in ['PongNoFrameskip-v4', 'BoxingNoFrameskip-v4'] else 2,
            # n_episode if env_id not in ['PongNoFrameskip-v4', 'BoxingNoFrameskip-v4'] else 2,
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
    from lzero.entry import train_unizero_multitask
    # TODO
    env_id_list = [
        'PongNoFrameskip-v4',
        'MsPacmanNoFrameskip-v4',
        'SeaquestNoFrameskip-v4',
        'BoxingNoFrameskip-v4'
    ]

    # env_id_list = [
    #     'PongNoFrameskip-v4',
    #     'MsPacmanNoFrameskip-v4',
    #     'SeaquestNoFrameskip-v4',
    #     'BoxingNoFrameskip-v4',
    #     'AlienNoFrameskip-v4',
    #     'CrazyClimberNoFrameskip-v4',
    #     'BreakoutNoFrameskip-v4',
    #     'QbertNoFrameskip-v4',
    # ]

    action_space_size = 18  # Full action space
    seed = 0
    collector_env_num = 8
    n_episode = 8
    evaluator_env_num = 3
    num_simulations = 50
    max_env_step = int(1e6)
    reanalyze_ratio = 0.
    # batch_size = [32, 32, 32, 32]
    max_batch_size = 2000
    batch_size = [int(max_batch_size/len(env_id_list)) for i in range(len(env_id_list))]
    num_unroll_steps = 10
    infer_context_length = 4
    norm_type = 'LN'
    # # norm_type = 'BN'  # bad performance now


    # ======== TODO: only for debug ========
    # collector_env_num = 3
    # n_episode = 3
    # evaluator_env_num = 2
    # num_simulations = 2
    # batch_size = [4, 4, 4, 4]

    configs = generate_configs(env_id_list, action_space_size, collector_env_num, n_episode, evaluator_env_num, num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type, seed)

    # Uncomment the desired training run
    # train_unizero_multitask(configs[:1], seed=seed, max_env_step=max_env_step)  # Pong
    # train_unizero_multitask(configs[:2], seed=seed, max_env_step=max_env_step)  # Pong, MsPacman
    train_unizero_multitask(configs, seed=seed, max_env_step=max_env_step)      # Pong, MsPacman, Seaquest, Boxing

    # only for cprofile
    # def run(max_env_step: int):
    #     train_unizero_multitask(configs, seed=seed, max_env_step=max_env_step)      # Pong, MsPacman, Seaquest, Boxing
    # import cProfile
    # cProfile.run(f"run({20000})", filename="unizero_mt_4games_cprofile_20k_envstep", sort="cumulative")