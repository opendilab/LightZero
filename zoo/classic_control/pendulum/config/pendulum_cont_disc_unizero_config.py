from easydict import EasyDict
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 25
update_per_collect = None
replay_ratio = 0.25
max_env_step = int(1e6)
reanalyze_ratio = 0
batch_size = 256
num_unroll_steps = 10
infer_context_length = 4
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

pendulum_cont_disc_unizero_config = dict(
    exp_name=f'data_unizero/pendulum_cont_disc_unizero_ns{num_simulations}_upc{update_per_collect}-rr{replay_ratio}_rer{reanalyze_ratio}_H{num_unroll_steps}_bs{batch_size}_seed0',
    env=dict(
        env_id='Pendulum-v1',
        continuous=False,
        manually_discretization=True,
        each_dim_disc_size=11,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=3,
            action_space_size=11,
            model_type='mlp',
            world_model_cfg=dict(
                obs_type='vector',
                max_blocks=num_unroll_steps,
                max_tokens=2 * num_unroll_steps,  # NOTE: each timestep has 2 tokens: obs and action
                context_length=2 * infer_context_length,
                device='cuda',
                action_space_size=11,
                env_num=collector_env_num,
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
                norm_type='LN',
            ),
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        num_unroll_steps=num_unroll_steps,
        cuda=True,
        use_augmentation=False,
        env_type='not_board_games',
        game_segment_length=50,
        replay_ratio=replay_ratio,
        batch_size=batch_size,
        optim_type='AdamW',
        learning_rate=0.0001,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(1e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

pendulum_cont_disc_unizero_config = EasyDict(pendulum_cont_disc_unizero_config)
main_config = pendulum_cont_disc_unizero_config

pendulum_cont_disc_unizero_create_config = dict(
    env=dict(
        type='pendulum_lightzero',
        import_names=['zoo.classic_control.pendulum.envs.pendulum_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='unizero',
        import_names=['lzero.policy.unizero'],
    ),
)
pendulum_cont_disc_unizero_create_config = EasyDict(pendulum_cont_disc_unizero_create_config)
create_config = pendulum_cont_disc_unizero_create_config

if __name__ == "__main__":
    from lzero.entry import train_unizero
    train_unizero([main_config, create_config], seed=0, max_env_step=max_env_step)
