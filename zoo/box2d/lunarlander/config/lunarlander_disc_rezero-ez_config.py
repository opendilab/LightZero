from easydict import EasyDict
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
update_per_collect = None
model_update_ratio = 0.25
batch_size = 256
max_env_step = int(1e6)
reanalyze_ratio = 1.
reuse_search = True
collect_with_pure_policy = True
use_priority = False
buffer_reanalyze_freq = 1
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

lunarlander_muzero_config = dict(
    exp_name=f'data_rezero-ez/lunarlander_rezero-ez_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
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
        # NOTE: not save ckpt
        learn=dict(
            learner=dict(
                train_iterations=1000000000,
                dataloader=dict(
                    num_workers=0,
                ),
                log_policy=True,
                hook=dict(
                    load_ckpt_before_run='',
                    log_show_after_iter=1000,
                    save_ckpt_after_iter=1000000,
                    save_ckpt_after_run=True,
                ),
                cfg_type='BaseLearnerDict',
            ),
        ),
        model=dict(
            observation_shape=8,
            action_space_size=4,
            model_type='mlp', 
            lstm_hidden_size=256,
            latent_state_dim=256,
            self_supervised_learning_loss=True,  # NOTE: default is False.
            discrete_action_encoding_type='one_hot',
            res_connection_in_dynamics=True,
            norm_type='BN', 
        ),
        cuda=True,
        env_type='not_board_games',
        game_segment_length=200,
        update_per_collect=update_per_collect,
        model_update_ratio=model_update_ratio,
        batch_size=batch_size,
        optim_type='Adam',
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
        reanalyze_noise=True,
        reuse_search=reuse_search,
        collect_with_pure_policy=collect_with_pure_policy,
        use_priority=use_priority,
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
        main_config.exp_name = f'data_rezero_ez/lunarlander_rezero-ez_ns{main_config.policy.num_simulations}_upc{main_config.policy.update_per_collect}_rr{main_config.policy.reanalyze_ratio}_seed{seed}'
        from lzero.entry import train_rezero
        train_rezero([main_config, create_config], seed=seed, max_env_step=max_env_step)
