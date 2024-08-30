from easydict import EasyDict
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================

from zoo.dmc2gym.config.dmc_state_env_space_map import dmc_state_env_action_space_map, dmc_state_env_obs_space_map

env_id = 'humanoid-run'  # 'cartpole-swingup'  # You can specify any DMC task here
# env_id =  'cartpole-swingup' # 'cartpole-swingup'  # You can specify any DMC task here

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
# num_simulations = 100
update_per_collect = None
# replay_ratio = 0.25
replay_ratio = 1
max_env_step = int(5e6)
reanalyze_ratio = 0
batch_size = 64
num_unroll_steps = 10
infer_context_length = 4
norm_type = 'LN'
seed = 0

# for debug
# collector_env_num = 2
# n_episode = 2
# evaluator_env_num = 2
# num_simulations = 1
# batch_size = 2
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

dmc2gym_state_cont_sampled_unizero_config = dict(
    exp_name=f'data_sampled_unizero/dmc2gym_{env_id}_state_cont_sampled_unizero_ns{num_simulations}_upc{update_per_collect}-rr{replay_ratio}_rer{reanalyze_ratio}_H{num_unroll_steps}_bs{batch_size}_{norm_type}_seed{seed}_policy-head-layer-num2_pew5e-3_disfac099_rbs5e5',
    env=dict(
        env_id='dmc2gym-v0',
        domain_name=domain_name,
        task_name=task_name,
        from_pixels=False,  # vector/state obs
        # from_pixels=True,  # vector/state obs
        frame_skip=2,
        continuous=True,
        save_replay_gif=True,
        replay_path_gif='./replay_gif_humanoid_0830',
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
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
                policy_entropy_loss_weight=5e-3,
                continuous_action_space=continuous_action_space,
                num_of_sampled_actions=K,
                sigma_type='conditioned',
                fixed_sigma_value=0.3,
                bound_type=None,
                model_type='mlp',
                max_blocks=num_unroll_steps,
                max_tokens=2 * num_unroll_steps,  # NOTE: each timestep has 2 tokens: obs and action
                context_length=2 * infer_context_length,
                # device='cpu',
                device='cuda',
                action_space_size=action_space_size,
                num_layers=2,
                num_heads=8,
                embed_dim=768,
                env_num=max(collector_env_num, evaluator_env_num),
            ),
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        # model_path=None,
        model_path='/Users/puyuan/code/LightZero/data_suz_0830/dmc2gym_humanoid-run_state_cont_sampled_unizero_ns50_upcNone-rr1_rer0_H10_bs64_LN_seed0_policy-head-layer-num2_pew5e-3_disfac099_rbs1e6/ckpt/ckpt_best.pth.tar',
        num_unroll_steps=num_unroll_steps,
        cuda=True,
        use_augmentation=False,
        env_type='not_board_games',
        game_segment_length=100,
        replay_ratio=replay_ratio,
        batch_size=batch_size,
        discount_factor=0.99,
        optim_type='AdamW',
        lr_piecewise_constant_decay=False,
        learning_rate=0.0001,
        target_update_freq=100,
        grad_clip_value=5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),
        # replay_buffer_size=int(5e5), # TODO
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
    # env_manager=dict(type='subprocess'),
    env_manager=dict(type='base'),
    policy=dict(
        type='sampled_unizero',
        import_names=['lzero.policy.sampled_unizero'],
    ),
)
dmc2gym_state_cont_sampled_unizero_create_config = EasyDict(dmc2gym_state_cont_sampled_unizero_create_config)
create_config = dmc2gym_state_cont_sampled_unizero_create_config

if __name__ == "__main__":
    from lzero.entry import train_unizero

    train_unizero([main_config, create_config], seed=seed, max_env_step=max_env_step)
