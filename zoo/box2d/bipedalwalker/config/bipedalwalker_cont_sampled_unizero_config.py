from easydict import EasyDict
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
continuous_action_space = True
K = 20  # num_of_sampled_actions
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
update_per_collect = None
replay_ratio = 0.25
max_env_step = int(1e6)
reanalyze_ratio = 0
batch_size = 64
num_unroll_steps = 10
infer_context_length = 4
norm_type = 'LN'
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

bipedalwalker_cont_sampled_unizero_config = dict(
    exp_name=f'data_suz/bipedalwalker_cont_sampled_unizero_ns{num_simulations}_upc{update_per_collect}-rr{replay_ratio}_rer{reanalyze_ratio}_H{num_unroll_steps}-icl{infer_context_length}_bs{batch_size}_{norm_type}_seed0',
    env=dict(
        env_id='BipedalWalker-v3',
        env_type='normal',
        continuous=True,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=24,
            action_space_size=4,
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            model_type='mlp',
            world_model_cfg=dict(
                obs_type='vector',
                num_unroll_steps=num_unroll_steps,
                policy_entropy_weight=5e-3,
                continuous_action_space=continuous_action_space,
                num_of_sampled_actions=K,
                sigma_type='conditioned',
                fixed_sigma_value=0.5,
                norm_type=norm_type,
                bound_type=None,
                max_blocks=num_unroll_steps,
                max_tokens=2 * num_unroll_steps,  # NOTE: each timestep has 2 tokens: obs and action
                context_length=2 * infer_context_length,
                device='cuda',
                action_space_size=4,
                env_num=max(collector_env_num, evaluator_env_num),
            ),
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        num_unroll_steps=num_unroll_steps,
        cuda=True,
        env_type='not_board_games',
        game_segment_length=200,
        replay_ratio=replay_ratio,
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
)

bipedalwalker_cont_sampled_unizero_config = EasyDict(bipedalwalker_cont_sampled_unizero_config)
main_config = bipedalwalker_cont_sampled_unizero_config

bipedalwalker_cont_sampled_unizero_create_config = dict(
    env=dict(
        type='bipedalwalker',
        import_names=['zoo.box2d.bipedalwalker.envs.bipedalwalker_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sampled_unizero',
        import_names=['lzero.policy.sampled_unizero'],
    ),
)
bipedalwalker_cont_sampled_unizero_create_config = EasyDict(bipedalwalker_cont_sampled_unizero_create_config)
create_config = bipedalwalker_cont_sampled_unizero_create_config

if __name__ == "__main__":
    from lzero.entry import train_unizero
    train_unizero([main_config, create_config], seed=0, max_env_step=max_env_step)
