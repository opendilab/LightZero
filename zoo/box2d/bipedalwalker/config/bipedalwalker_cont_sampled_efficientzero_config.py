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
update_per_collect = 200
batch_size = 256
max_env_step = int(5e6)
reanalyze_ratio = 0.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

bipedalwalker_cont_sampled_efficientzero_config = dict(
    exp_name=
    f'data_sez_ctree/bipedalwalker_cont_sampled_efficientzero_k{K}_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_sslw2_gcv05_lsd256_bias-t_seed0',
    env=dict(
        env_name='BipedalWalker-v3',
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
            self_supervised_learning_loss=True,
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            sigma_type='conditioned',  # options={'conditioned', 'fixed'}
            model_type='mlp',  # options={'mlp', 'conv'}
            bias=True,
            lstm_hidden_size=256,
            # The mlp model.
            latent_state_dim=256,
            # The conv model.
            # num_res_blocks=1,
            # num_channels=32,
        ),
        cuda=True,
        env_type='not_board_games',
        game_segment_length=200,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        ssl_loss_weight=2,
        grad_clip_value=0.5,  # NOTE: this parameter is important for stability in bipedalwalker.
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        # NOTE: for continuous gaussian policy, we use the policy_entropy_loss as in the original Sampled MuZero paper.
        policy_entropy_loss_weight=5e-3,
        policy_loss_type='cross_entropy',  # options={'cross_entropy', 'KL'}
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
bipedalwalker_cont_sampled_efficientzero_config = EasyDict(bipedalwalker_cont_sampled_efficientzero_config)
main_config = bipedalwalker_cont_sampled_efficientzero_config

bipedalwalker_cont_sampled_efficientzero_create_config = dict(
    env=dict(
        type='bipedalwalker',
        import_names=['zoo.box2d.bipedalwalker.envs.bipedalwalker_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sampled_efficientzero',
        import_names=['lzero.policy.sampled_efficientzero'],
    ),
    collector=dict(
        type='episode_muzero',
        get_train_sample=True,
        import_names=['lzero.worker.muzero_collector'],
    )
)
bipedalwalker_cont_sampled_efficientzero_create_config = EasyDict(
    bipedalwalker_cont_sampled_efficientzero_create_config
)
create_config = bipedalwalker_cont_sampled_efficientzero_create_config

if __name__ == "__main__":
    # Users can use different train entry by specifying the entry_type.
    entry_type = "train_muzero"  # options={"train_muzero", "train_muzero_with_gym_env"}

    if entry_type == "train_muzero":
        from lzero.entry import train_muzero
    elif entry_type == "train_muzero_with_gym_env":
        """
        The ``train_muzero_with_gym_env`` entry means that the environment used in the training process is generated by wrapping the original gym environment with LightZeroEnvWrapper.
        Users can refer to lzero/envs/wrappers for more details.
        """
        from lzero.entry import train_muzero_with_gym_env as train_muzero

    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
