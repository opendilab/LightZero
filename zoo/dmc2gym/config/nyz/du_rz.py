from easydict import EasyDict
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================

from zoo.dmc2gym.config.dmc_state_env_space_map import dmc_state_env_action_space_map, dmc_state_env_obs_space_map

# env_id = 'humanoid-run'  # 'cartpole-swingup'  # You can specify any DMC task here
# env_id = 'cheetah-run'  # 'cartpole-swingup'  # You can specify any DMC task here
# env_id = 'hopper-hop'  # 'cartpole-swingup'  # You can specify any DMC task here
# env_id =  'cartpole-swingup' # 'cartpole-swingup'  # You can specify any DMC task here

def main(env_id, seed):
    action_space_size = dmc_state_env_action_space_map[env_id]
    obs_space_size = dmc_state_env_obs_space_map[env_id]
    print(f'env_id: {env_id}, action_space_size: {action_space_size}, obs_space_size: {obs_space_size}')

    domain_name = env_id.split('-')[0]
    task_name = env_id.split('-')[1]

    continuous_action_space = True
    K = 20  # num_of_sampled_actions
    # K = 5  # num_of_sampled_actions

    collector_env_num = 8
    n_episode = 8
    num_segments = 8
    game_segment_length=20
    evaluator_env_num = 8
    num_simulations = 50
    replay_ratio = 0.25

    # if env_id ==  'cartpole-swingup':
    #     max_env_step = int(1e6)
    # else:
    #     max_env_step = int(4e6)

    max_env_step = int(5e5)
    reanalyze_ratio = 0.0

    batch_size = 64


    num_layers = 2
    num_unroll_steps = 5
    infer_context_length = 2
    norm_type = 'LN'

    # buffer_reanalyze_freq = 1/10  # modify according to num_segments
    buffer_reanalyze_freq = 1/1000000  # modify according to num_segments

    # 20*8*10/5=320
    reanalyze_batch_size = 160   # in total of num_unroll_steps
    # reanalyze_partition=3/4
    reanalyze_partition=1

    # ==============================================================
    # end of the most frequently changed config specified by the user
    # ==============================================================

    dmc2gym_state_cont_sampled_unizero_config = dict(
        env=dict(
            env_id='dmc2gym-v0',
            domain_name=domain_name,
            task_name=task_name,
            from_pixels=False,  # vector/state obs
            # from_pixels=True,  # vector/state obs
            frame_skip=2,
            # frame_skip=8,
            continuous=True,
            save_replay_gif=False,
            # save_replay_gif=True,
            replay_path_gif='./replay_gif',
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
            # TODO: only for debug: 目前不起作用
            # collect_max_episode_steps=int(20),
            # eval_max_episode_steps=int(20),
        ),
        policy=dict(
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000000,),),),  # default is 10000
            model=dict(
                observation_shape=obs_space_size,
                action_space_size=action_space_size,
                continuous_action_space=continuous_action_space,
                num_of_sampled_actions=K,
                model_type='mlp',
                norm_type = norm_type,
                world_model_cfg=dict(
                    policy_loss_type='kl', # 'simple'
                    obs_type='vector',
                    num_unroll_steps=num_unroll_steps,
                    policy_entropy_weight=5e-3,
                    continuous_action_space=continuous_action_space,
                    num_of_sampled_actions=K,
                    sigma_type='conditioned',
                    # sigma_type='fixed',
                    # fixed_sigma_value=fixed_sigma_value,
                    fixed_sigma_value=0.5,
                    bound_type=None,
                    model_type='mlp',
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,  # NOTE: each timestep has 2 tokens: obs and action
                    context_length=2 * infer_context_length,
                    # device='cpu',
                    device='cuda',
                    action_space_size=action_space_size,
                    num_layers=num_layers,
                    num_heads=8,
                    embed_dim=768,
                    env_num=max(collector_env_num, evaluator_env_num),
                ),
            ),
            # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            cuda=True,
            use_root_value=False, # TODO
            use_augmentation=False,
            use_priority=False,
            env_type='not_board_games',
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            discount_factor=1,
            td_steps=5,
            lr_piecewise_constant_decay=False,
            learning_rate=0.0001,
            grad_clip_value=5, # TODO
            # grad_clip_value=20,
            # manual_temperature_decay=True,  # TODO
            manual_temperature_decay=False,  # TODO
            # cos_lr_scheduler=True,
            cos_lr_scheduler=False,
            num_segments=num_segments,
            train_start_after_envsteps=2000,
            # train_start_after_envsteps=0, # TODO: for debug
            game_segment_length=game_segment_length, # debug
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            n_episode=n_episode,
            eval_freq=int(5e3),
            replay_buffer_size=int(1e6),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            # ============= The key different params for ReZero =============
            buffer_reanalyze_freq=buffer_reanalyze_freq, # 1 means reanalyze one times per epoch, 2 means reanalyze one times each two epoch
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
        ),
    )

    dmc2gym_state_cont_sampled_unizero_config = EasyDict(dmc2gym_state_cont_sampled_unizero_config)
    main_config = dmc2gym_state_cont_sampled_unizero_config

    dmc2gym_state_cont_sampled_unizero_create_config = dict(
        env=dict(
            type='dmc2gym_lightzero',
            import_names=['zoo.dmc2gym.envs.dmc2gym_lightzero_env'],
        ),
        env_manager=dict(type='subprocess'),
        # env_manager=dict(type='base'),
        policy=dict(
            type='sampled_unizero',
            import_names=['lzero.policy.sampled_unizero'],
        ),
    )
    dmc2gym_state_cont_sampled_unizero_create_config = EasyDict(dmc2gym_state_cont_sampled_unizero_create_config)
    create_config = dmc2gym_state_cont_sampled_unizero_create_config
    
    # 调整train_unizero里面的collector
    #main_config.exp_name=f'data_sampled_unizero_0930/ucb-uniform-prior_fs2_seg-collector_fixvalueV8/dmc2gym_{env_id}_state_cont_suz_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_K{K}_ns{num_simulations}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_{norm_type}_seed{seed}_learnsigma'
    main_config.exp_name=f'data_sampled_unizero/rezero_{env_id}'
    from lzero.entry import train_unizero_reanalyze
    train_unizero_reanalyze([main_config, create_config], model_path=main_config.policy.model_path, seed=seed, max_env_step=max_env_step)

    # main_config.exp_name=f'data_efficiency0829_plus_tune-suz_0926/ucb-uniform-prior_fs2_seg-collector-origsctre/dmc2gym_{env_id}_state_cont_suz_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-only{reanalyze_partition}_nlayer{num_layers}_collect{collector_env_num}-numsegments-{num_segments}_gsl{game_segment_length}_K{K}_ns{num_simulations}_upc{update_per_collect}-rr{replay_ratio}_rer{reanalyze_ratio}_H{num_unroll_steps}-eval{infer_context_length}_bs{batch_size}_{norm_type}_seed{seed}_fixsigma'
    # from lzero.entry import train_rezero_uz # MuZeroSegmentCollector
    # train_rezero_uz([main_config, create_config], seed=seed, model_path=main_config.policy.model_path, max_env_step=max_env_step)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some environment.')
    
    parser.add_argument('--env', type=str, help='The environment to use', default='cartpole-swingup')
    
    args = parser.parse_args()
    for i in range(4):
        main(args.env, i)
