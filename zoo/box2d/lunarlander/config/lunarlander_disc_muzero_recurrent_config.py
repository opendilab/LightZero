from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
seed = 0
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
update_per_collect = 200
batch_size = 256
max_env_step = int(3e6)
reanalyze_ratio = 0.

# collector_env_num = 1
# n_episode = 1
# evaluator_env_num = 1
# num_simulations = 5
# update_per_collect = 2
# batch_size = 4
# max_env_step = int(3e6)
# reanalyze_ratio = 0.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

lunarlander_muzero_recurrent_config = dict(
    exp_name=f'data_mz_ctree/lunarlander_muzero_recurrent_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_adamw3e-3_sslw2_seed{seed}',
    env=dict(
        stop_value=int(1e6),
        env_name='LunarLander-v2',
        continuous=False,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=8,
            action_space_size=4,
            model_type='mlp', 
            lstm_hidden_size=256,
            latent_state_dim=256,
            discrete_action_encoding_type='one_hot',
            res_connection_in_dynamics=True,
            norm_type='BN',
            self_supervised_learning_loss=True,  # NOTE: default is False.
        ),
        cuda=True,
        env_type='not_board_games',
        game_segment_length=200,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='AdamW',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        ssl_loss_weight=2,  # NOTE: default is 0.
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(1e3),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
lunarlander_muzero_recurrent_config = EasyDict(lunarlander_muzero_recurrent_config)
main_config = lunarlander_muzero_recurrent_config

lunarlander_muzero_recurrent_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['zoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero_recurrent',
        import_names=['lzero.policy.muzero_recurrent'],
    ),
    collector=dict(
        type='episode_muzero',
        get_train_sample=True,
        import_names=['lzero.worker.muzero_collector'],
    )
)
lunarlander_muzero_recurrent_create_config = EasyDict(lunarlander_muzero_recurrent_create_config)
create_config = lunarlander_muzero_recurrent_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=seed, max_env_step=max_env_step)
