from easydict import EasyDict
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
update_per_collect = None
replay_ratio = 0.25
batch_size = 256
max_env_step = int(1e6)
# ============= The key different params for ReZero =============
reuse_search = True
collect_with_pure_policy = False
buffer_reanalyze_freq = 1
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

lunarlander_muzero_config = dict(
    exp_name=f'data_rezero-ez/lunarlander_rezero-ez_ns{num_simulations}_upc{update_per_collect}_brf{buffer_reanalyze_freq}_seed0',
    env=dict(
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
            self_supervised_learning_loss=True,
            discrete_action_encoding_type='one_hot',
            res_connection_in_dynamics=True,
            norm_type='BN', 
        ),
        cuda=True,
        env_type='not_board_games',
        game_segment_length=200,
        update_per_collect=update_per_collect,
        replay_ratio=replay_ratio,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.003,
        ssl_loss_weight=2,
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        n_episode=n_episode,
        eval_freq=int(1e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        reanalyze_noise=True,
        # ============= The key different params for ReZero =============
        reuse_search=reuse_search,
        collect_with_pure_policy=collect_with_pure_policy,
        buffer_reanalyze_freq=buffer_reanalyze_freq,
    ),
)
lunarlander_muzero_config = EasyDict(lunarlander_muzero_config)
main_config = lunarlander_muzero_config

lunarlander_muzero_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['zoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='efficientzero',
        import_names=['lzero.policy.efficientzero'],
    ),
)
lunarlander_muzero_create_config = EasyDict(lunarlander_muzero_create_config)
create_config = lunarlander_muzero_create_config

if __name__ == "__main__":
    # Define a list of seeds for multiple runs
    seeds = [0]  # You can add more seed values here
    for seed in seeds:
        # Update exp_name to include the current seed
        main_config.exp_name = f'data_rezero_ez/lunarlander_rezero-ez_ns{num_simulations}_upc{update_per_collect}_brf{buffer_reanalyze_freq}_seed{seed}'
        from lzero.entry import train_rezero
        train_rezero([main_config, create_config], seed=seed, max_env_step=max_env_step)
