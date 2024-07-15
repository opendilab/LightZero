from easydict import EasyDict
from copy import deepcopy
# from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map

def create_config(env_id, action_space_size, collector_env_num, evaluator_env_num, n_episode, num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length):
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
            # collect_max_episode_steps=int(50),
            # eval_max_episode_steps=int(50),
            # collect_max_episode_steps=int(500),
            # eval_max_episode_steps=int(500),
        ),
        policy=dict(
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
                world_model_cfg=dict(
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    # device='cpu',  # 'cuda',
                    device='cuda',  # 'cuda',
                    action_space_size=action_space_size,
                    num_layers=4,  # NOTE
                    num_heads=8,
                    embed_dim=768,
                    obs_type='image',
                    env_num=max(collector_env_num, evaluator_env_num),
                    # collector_env_num=collector_env_num,
                    # evaluator_env_num=evaluator_env_num,
                    task_num=len(env_id_list),
                    num_experts_in_softmoe=-1,  # NOTE
                    # num_experts_in_softmoe=-1,  # NOTE
                    num_fc_gating_layers=2,
                    base_layers_num=5
                ),
            ),
            cuda=True,
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            update_per_collect=None,
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

def generate_configs(env_id_list, action_space_size, collector_env_num, n_episode, evaluator_env_num, num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, seed):
    configs = []
    # exp_name_prefix = f'data_unizero_mt_0711/{len(env_id_list)}games_{"-".join(env_id_list)}_1-head-softmoe4_1-encoder-LN_lsd768-nlayer4-nh8_seed{seed}/'
    exp_name_prefix = f'data_unizero_mt_0711_debug/{len(env_id_list)}games_1-head-softmoe4_1-encoder-LN_lsd768-nlayer4-nh8_seed{seed}/'

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
            infer_context_length
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

    env_id_list = [
        'PongNoFrameskip-v4',
        'MsPacmanNoFrameskip-v4',
        'SeaquestNoFrameskip-v4',
        'BoxingNoFrameskip-v4'
    ]

    action_space_size = 18  # Full action space
    seed = 0
    collector_env_num = 8
    n_episode = 8
    evaluator_env_num = 3
    num_simulations = 50
    max_env_step = int(1e6)
    reanalyze_ratio = 0.
    batch_size = [32, 32, 32, 32]
    num_unroll_steps = 10
    infer_context_length = 4

    # ======== only for debug ========
    collector_env_num = 3
    n_episode = 3
    evaluator_env_num = 2
    num_simulations = 5
    batch_size = [4, 4, 4, 4]

    configs = generate_configs(env_id_list, action_space_size, collector_env_num, n_episode, evaluator_env_num, num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, seed)

    # Uncomment the desired training run
    # train_unizero_multitask(configs[:1], seed=seed, max_env_step=max_env_step)  # Pong
    # train_unizero_multitask(configs[:2], seed=seed, max_env_step=max_env_step)  # Pong, MsPacman
    train_unizero_multitask(configs, seed=seed, max_env_step=max_env_step)      # Pong, MsPacman, Seaquest, Boxing