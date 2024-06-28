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
max_env_step = int(2e5)
reanalyze_ratio = 0
batch_size = 256
num_unroll_steps = 5
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

cartpole_unizero_config = dict(
    exp_name=f'data_unizero/cartpole_unizero_ns{num_simulations}_upc{update_per_collect}-rr{replay_ratio}_rer{reanalyze_ratio}_H{num_unroll_steps}_bs{batch_size}_seed0',
    env=dict(
        env_name='CartPole-v0',
        continuous=False,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model_path=None,
        train_start_after_envsteps=int(0),
        num_unroll_steps=num_unroll_steps,
        model=dict(
            analysis_sim_norm=False,
            observation_shape=4,
            action_space_size=2,
            self_supervised_learning_loss=True,  # NOTE: default is False.
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            model_type='mlp',
            world_model=dict(
                max_blocks=10,
                max_tokens=2 * 10,
                context_length=2 * 4,
                context_length_for_recurrent=2 * 4,
                gru_gating=False,
                device='cpu',
                analysis_sim_norm=False,
                analysis_dormant_ratio=False,
                action_shape=2,
                group_size=8,
                attention='causal',
                num_layers=2,
                num_heads=2,
                embed_dim=64,
                env_num=collector_env_num,
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
                latent_recon_loss_weight=0.,
                perceptual_loss_weight=0.,
                policy_entropy_weight=1e-4,
                predict_latent_loss_type='group_kl',
                obs_type='vector',
                norm_type='BN',
            ),
        ),
        cuda=True,
        use_augmentation=False,
        env_type='not_board_games',
        game_segment_length=50,
        replay_ratio=replay_ratio,
        batch_size=batch_size,
        optim_type='AdamW',
        lr_piecewise_constant_decay=False,
        learning_rate=0.0001,
        target_update_freq=100,
        grad_clip_value=5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(1e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

cartpole_unizero_config = EasyDict(cartpole_unizero_config)
main_config = cartpole_unizero_config

cartpole_unizero_create_config = dict(
    env=dict(
        type='cartpole_lightzero',
        import_names=['zoo.classic_control.cartpole.envs.cartpole_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='unizero',
        import_names=['lzero.policy.unizero'],
    ),
)
cartpole_unizero_create_config = EasyDict(cartpole_unizero_create_config)
create_config = cartpole_unizero_create_config

if __name__ == "__main__":
    from lzero.entry import train_unizero
    train_unizero([main_config, create_config], seed=0, max_env_step=max_env_step)
