from easydict import EasyDict
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
reanalyze_ratio = 0.
update_per_collect = None
replay_ratio = 0.25
max_env_step = int(1e6)
batch_size = 256
num_unroll_steps = 5

# debug
# collector_env_num = 2
# n_episode = 2
# evaluator_env_num = 2
# num_simulations = 5
# batch_size = 2
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

lunarlander_unizero_config = dict(
    exp_name=f'data_unizero/lunarlander_unizero_ns{num_simulations}_upc{update_per_collect}-rr{replay_ratio}_rer{reanalyze_ratio}_H{num_unroll_steps}_bs{batch_size}_seed0',
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
            latent_state_dim=256,
            self_supervised_learning_loss=True,  # NOTE: default is False.
            discrete_action_encoding_type='one_hot',
            res_connection_in_dynamics=True,
            norm_type='LN',
            world_model=dict(
                max_blocks=10,
                max_tokens=2 * 10,
                context_length=2 * 4,
                context_length_for_recurrent=2 * 4,
                gru_gating=False,
                device='cpu',
                action_shape=4,  # TODO
                group_size=8,  # NOTE: sim_norm
                num_layers=4,
                num_heads=4,
                embed_dim=256,
                env_num=collector_env_num,
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
                predict_latent_loss_type='group_kl',
                obs_type='vector',
                norm_type='LN',
            ),
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        num_unroll_steps=num_unroll_steps,
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
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
lunarlander_unizero_config = EasyDict(lunarlander_unizero_config)
main_config = lunarlander_unizero_config

lunarlander_unizero_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['zoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='unizero',
        import_names=['lzero.policy.unizero'],
    ),
)
lunarlander_unizero_create_config = EasyDict(lunarlander_unizero_create_config)
create_config = lunarlander_unizero_create_config

if __name__ == "__main__":
    from lzero.entry import train_unizero
    train_unizero([main_config, create_config], seed=0, max_env_step=max_env_step)
