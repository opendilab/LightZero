from easydict import EasyDict
# import torch
# torch.cuda.set_device(0)
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
reanalyze_ratio = 0
batch_size = 64
num_unroll_steps = 5
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
        # TODO
        analysis_sim_norm=False,
        cal_dormant_ratio=False,
        model_path=None,
        train_start_after_envsteps=int(0),
        num_unroll_steps=num_unroll_steps,
        model=dict(
            analysis_sim_norm=False,
            observation_shape=8,
            action_space_size=4,
            model_type='mlp', 
            latent_state_dim=256,
            self_supervised_learning_loss=True,  # NOTE: default is False.
            discrete_action_encoding_type='one_hot',
            res_connection_in_dynamics=True,
            norm_type='LN',
            reward_support_size=101,
            value_support_size=101,
            support_scale=50,
            world_model=dict(
                tokens_per_block=2,
                max_blocks=10,
                max_tokens=2 * 10,
                context_length=2 * 4,
                context_length_for_recurrent=2 * 4,
                gru_gating=False,
                device='cpu',
                analysis_sim_norm=False,
                analysis_dormant_ratio=False,
                action_shape=4,  # TODO
                group_size=8,  # NOTE: sim_norm
                attention='causal',
                num_layers=4,  # TODO
                num_heads=4,
                embed_dim=256,  # TODO
                embed_pdrop=0.1,
                resid_pdrop=0.1,
                attn_pdrop=0.1,
                support_size=101,
                max_cache_size=5000,
                env_num=8,
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
                latent_recon_loss_weight=0.,
                perceptual_loss_weight=0.,
                policy_entropy_weight=1e-4,
                predict_latent_loss_type='group_kl',
                obs_type='vector',
                gamma=1,
                dormant_threshold=0.025,
            ),
        ),
        use_priority=False,  # TODO
        use_augmentation=False,  # TODO
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
    collector=dict(
        type='episode_muzero',
        get_train_sample=True,
        import_names=['lzero.worker.muzero_collector'],
    )
)
lunarlander_unizero_create_config = EasyDict(lunarlander_unizero_create_config)
create_config = lunarlander_unizero_create_config

if __name__ == "__main__":
    from lzero.entry import train_unizero
    train_unizero([main_config, create_config], seed=0, max_env_step=max_env_step)
