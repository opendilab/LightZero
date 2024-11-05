from easydict import EasyDict
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================

from zoo.dmc2gym.config.dmc_state_env_space_map import dmc_state_env_action_space_map, dmc_state_env_obs_space_map

env_id = 'cartpole-swingup'  # You can specify any DMC task here
action_space_size = dmc_state_env_action_space_map[env_id]
obs_space_size = dmc_state_env_obs_space_map[env_id]
print(f'env_id: {env_id}, action_space_size: {action_space_size}, obs_space_size: {obs_space_size}')

domain_name = env_id.split('-')[0]
task_name = env_id.split('-')[1]

continuous_action_space = True
K = 20  # num_of_sampled_actions
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
update_per_collect = None
replay_ratio = 0.25
max_env_step = int(1e6)
batch_size = 64
num_unroll_steps = 5
infer_context_length = 2
norm_type = 'LN'
seed = 0

# for debug
# collector_env_num = 2
# n_episode = 2
# evaluator_env_num = 1
# num_simulations = 2
# batch_size = 2
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

dmc2gym_pixels_cont_sampled_unizero_config = dict(
    exp_name=f'data_suz/dmc2gym_{env_id}_image_cont_sampled_unizero_ns{num_simulations}_upc{update_per_collect}-rr{replay_ratio}_rer{reanalyze_ratio}_H{num_unroll_steps}_bs{batch_size}_{norm_type}_seed{seed}',
    env=dict(
        env_id='dmc2gym-v0',
        continuous=True,
        domain_name=domain_name,
        task_name=task_name,
        from_pixels=True,  # pixel/image obs
        frame_skip=2,
        warp_frame=True,
        scale=True,
        frame_stack_num=1,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=(3, 84, 84),
            action_space_size=action_space_size,
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            world_model_cfg=dict(
                obs_type='image',
                num_unroll_steps=num_unroll_steps,
                policy_entropy_weight=5e-2,
                continuous_action_space=continuous_action_space,
                num_of_sampled_actions=K,
                sigma_type='conditioned',
                fixed_sigma_value=0.5,
                bound_type=None,
                model_type='conv',
                max_blocks=num_unroll_steps,
                max_tokens=2 * num_unroll_steps,  # NOTE: each timestep has 2 tokens: obs and action
                context_length=2 * infer_context_length,
                device='cuda',
                action_space_size=action_space_size,
                num_layers=2,
                num_heads=8,
                embed_dim=768,
                env_num=max(collector_env_num, evaluator_env_num),
            ),
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        num_unroll_steps=num_unroll_steps,
        cuda=True,
        use_augmentation=False,
        env_type='not_board_games',
        game_segment_length=100,
        replay_ratio=replay_ratio,
        batch_size=batch_size,
        discount_factor=0.99,
        td_steps=5,
        lr_piecewise_constant_decay=False,
        learning_rate=1e-4,
        grad_clip_value=5,
        manual_temperature_decay=True,
        threshold_training_steps_for_final_temperature=int(2.5e4),
        cos_lr_scheduler=True,
        optim_type='AdamW',
        num_simulations=num_simulations,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

dmc2gym_pixels_cont_sampled_unizero_config = EasyDict(dmc2gym_pixels_cont_sampled_unizero_config)
main_config = dmc2gym_pixels_cont_sampled_unizero_config

dmc2gym_pixels_cont_sampled_unizero_create_config = dict(
    env=dict(
        type='dmc2gym_lightzero',
        import_names=['zoo.dmc2gym.envs.dmc2gym_lightzero_env'],
    ),
    # env_manager=dict(type='subprocess'),
    env_manager=dict(type='base'),
    policy=dict(
        type='sampled_unizero',
        import_names=['lzero.policy.sampled_unizero'],
    ),
)
dmc2gym_pixels_cont_sampled_unizero_create_config = EasyDict(dmc2gym_pixels_cont_sampled_unizero_create_config)
create_config = dmc2gym_pixels_cont_sampled_unizero_create_config

if __name__ == "__main__":
    from lzero.entry import train_unizero
    train_unizero([main_config, create_config], seed=seed, max_env_step=max_env_step)
