from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 25
update_per_collect = 100
batch_size = 256
max_env_step = int(1e5)
reanalyze_ratio = 0
enable_chance = False
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

cartpole_stochastic_muzero_config = dict(
    exp_name=f'data_stochastic_mz/cartpole_stochastic_muzero_ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}_seed0',
    env=dict(
        env_id='CartPole-v0',
        continuous=False,
        enable_chance=enable_chance,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=4,
            action_space_size=2,
            chance_space_size=3,
            model_type='mlp',
            lstm_hidden_size=128,
            latent_state_dim=128,
            self_supervised_learning_loss=True,  # NOTE: default is False.
            discrete_action_encoding_type='one_hot',
            norm_type='BN', 
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        cuda=True,
        env_type='not_board_games',
        game_segment_length=50,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        ssl_loss_weight=2,  # NOTE: default is 0.
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        use_ture_chance_label_in_chance_encoder=enable_chance,
        n_episode=n_episode,
        eval_freq=int(2e2),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

cartpole_stochastic_muzero_config = EasyDict(cartpole_stochastic_muzero_config)
main_config = cartpole_stochastic_muzero_config

cartpole_stochastic_muzero_create_config = dict(
    env=dict(
        type='cartpole_lightzero',
        import_names=['zoo.classic_control.cartpole.envs.cartpole_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='stochastic_muzero',
        import_names=['lzero.policy.stochastic_muzero'],
    ),
)
cartpole_stochastic_muzero_create_config = EasyDict(cartpole_stochastic_muzero_create_config)
create_config = cartpole_stochastic_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)
