from easydict import EasyDict

from zoo.dmc2gym.config.dmc_state_env_space_map import dmc_state_env_action_space_map, dmc_state_env_obs_space_map

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================

env_id =  'cartpole-swingup' # 'cartpole-swingup'  # You can specify any DMC task here

action_space_size = dmc_state_env_action_space_map[env_id]
obs_space_size = dmc_state_env_obs_space_map[env_id]
print(f'env_id: {env_id}, action_space_size: {action_space_size}, obs_space_size: {obs_space_size}')

domain_name = env_id.split('-')[0]
task_name = env_id.split('-')[1]

continuous_action_space = True
K = 20  # num_of_sampled_actions
collector_env_num = 8
n_episode = 8
num_segments = 8
game_segment_length = 100
evaluator_env_num = 3
num_simulations = 50
update_per_collect = None
max_env_step = int(5e5)
reanalyze_ratio = 0
batch_size = 64
num_layers = 2
num_unroll_steps = 10
infer_context_length = 4
replay_ratio = 0.25
norm_type = 'LN'
seed = 0

# for debug
# collector_env_num = 2
# n_episode = 2
# evaluator_env_num = 2
# num_simulations = 3
# batch_size = 3
# reanalyze_batch_size = 1
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
        continuous=True,
        save_replay_gif=False,
        replay_path_gif='./replay_gif',
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        # TODO: only for debug
        # collect_max_episode_steps=int(20),
        # eval_max_episode_steps=int(20),
    ),
    policy=dict(
        model=dict(
            observation_shape=obs_space_size,
            action_space_size=action_space_size,
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            model_type='mlp',
            world_model_cfg=dict(


                obs_type='vector',
                num_unroll_steps=num_unroll_steps,
                policy_entropy_weight=5e-3,
                continuous_action_space=continuous_action_space,
                num_of_sampled_actions=K,
                sigma_type='fixed',
                fixed_sigma_value=0.5,
                bound_type=None,
                model_type='mlp',
                norm_type=norm_type,
                max_blocks=num_unroll_steps,
                max_tokens=2 * num_unroll_steps,  # NOTE: each timestep has 2 tokens: obs and action
                context_length=2 * infer_context_length,
                device='cuda',
                action_space_size=action_space_size,
                num_layers=num_layers,
                num_heads=8,
                embed_dim=768,
                env_num=max(collector_env_num, evaluator_env_num),
            ),
        ),
        learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000000,),),),  # default is 10000
        model_path=None,
        num_unroll_steps=num_unroll_steps,
        cuda=True,
        use_augmentation=False,
        env_type='not_board_games',
        replay_ratio=replay_ratio,
        batch_size=batch_size,
        discount_factor=0.99,
        lr_piecewise_constant_decay=False,
        learning_rate=1e-4,
        grad_clip_value=5, # TODO
        manual_temperature_decay=False,
        cos_lr_scheduler=False,
        num_segments=num_segments,
        train_start_after_envsteps=2000,
        game_segment_length=game_segment_length,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(5e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
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
    policy=dict(
        type='sampled_unizero',
        import_names=['lzero.policy.sampled_unizero'],
    ),
)
dmc2gym_state_cont_sampled_unizero_create_config = EasyDict(dmc2gym_state_cont_sampled_unizero_create_config)
create_config = dmc2gym_state_cont_sampled_unizero_create_config
    

if __name__ == "__main__":
    from lzero.entry import train_unizero
    main_config.exp_name=f'data_suz/dmc2gym_{env_id}_state_cont_suz_nlayer{num_layers}_gsl{game_segment_length}_K{K}_ns{num_simulations}_upc{update_per_collect}-rr{replay_ratio}_rer{reanalyze_ratio}_H{num_unroll_steps}-eval{infer_context_length}_bs{batch_size}_{norm_type}_seed{seed}'
    train_unizero([main_config, create_config], model_path=main_config.policy.model_path, seed=seed, max_env_step=max_env_step)
