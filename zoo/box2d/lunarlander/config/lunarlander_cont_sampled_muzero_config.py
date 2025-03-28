from easydict import EasyDict
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
continuous_action_space = True
K = 20  # num_of_sampled_actions
num_simulations = 50
update_per_collect = None
replay_ratio = 0.25
batch_size = 1024
max_env_step = int(1e6)
reanalyze_ratio = 0.
norm_type = 'LN'
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

lunarlander_cont_sampled_muzero_config = dict(
    exp_name=f'data_smz/lunarlander_cont_sampled_muzero_k{K}_ns{num_simulations}_upc{update_per_collect}-rr{replay_ratio}_rer{reanalyze_ratio}_norm-{norm_type}_seed0',
    env=dict(
        env_id='LunarLanderContinuous-v2',
        continuous=True,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=8,
            action_space_size=2,
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            sigma_type='fixed',
            fixed_sigma_value=0.5,
            model_type='mlp',
            norm_type=norm_type,
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        cuda=True,
        game_segment_length=200,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='AdamW',
        cos_lr_scheduler=True,
        learning_rate=1e-4,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        random_collect_episode_num=0,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_ratio=replay_ratio,
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
lunarlander_cont_sampled_muzero_config = EasyDict(lunarlander_cont_sampled_muzero_config)
main_config = lunarlander_cont_sampled_muzero_config

lunarlander_cont_sampled_muzero_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['zoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sampled_muzero',
        import_names=['lzero.policy.sampled_muzero'],
    ),
)
lunarlander_cont_sampled_muzero_create_config = EasyDict(lunarlander_cont_sampled_muzero_create_config)
create_config = lunarlander_cont_sampled_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)
