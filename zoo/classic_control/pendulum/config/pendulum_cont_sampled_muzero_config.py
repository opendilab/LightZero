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
batch_size = 256
max_env_step = int(2e5)
reanalyze_ratio = 0.
norm_type = 'LN'
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

pendulum_sampled_muzero_config = dict(
    exp_name=f'data_smz/pendulum_sampled_muzero_k{K}_ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}_norm-{norm_type}_seed0',
    env=dict(
        env_id='Pendulum-v1',
        continuous=True,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=3,
            action_space_size=1,
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            sigma_type='conditioned',
            model_type='mlp', 
            latent_state_dim=128,
            norm_type=norm_type,
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        cuda=True,
        env_type='not_board_games',
        game_segment_length=50,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='AdamW',
        cos_lr_scheduler=True,
        learning_rate=0.0001,
        piecewise_decay_lr_scheduler=False,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
pendulum_sampled_muzero_config = EasyDict(pendulum_sampled_muzero_config)
main_config = pendulum_sampled_muzero_config

pendulum_sampled_muzero_create_config = dict(
    env=dict(
        type='pendulum_lightzero',
        import_names=['zoo.classic_control.pendulum.envs.pendulum_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sampled_muzero',
        import_names=['lzero.policy.sampled_muzero'],
    ),
)
pendulum_sampled_muzero_create_config = EasyDict(pendulum_sampled_muzero_create_config)
create_config = pendulum_sampled_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)
