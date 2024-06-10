from easydict import EasyDict

env_id = 'PongNoFrameskip-v4'  # 6
# env_id = 'SeaquestNoFrameskip-v4' # 18
# env_id = 'MsPacmanNoFrameskip-v4' # 9
# env_id = 'BoxingNoFrameskip-v4' # 18

if env_id == 'PongNoFrameskip-v4':
    action_space_size = 6
elif env_id == 'SeaquestNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'MsPacmanNoFrameskip-v4':
    action_space_size = 9
elif env_id == 'BoxingNoFrameskip-v4':
    action_space_size = 18

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
update_per_collect = None
model_update_ratio = 0.25
reanalyze_ratio = 0
batch_size = 64
num_unroll_steps = 10
max_env_step = int(5e5)
num_simulations = 50
eps_greedy_exploration_in_collect = True
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_muzero_config = dict(
    exp_name=f'data_unizero/{env_id[:-14]}_stack4/',
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        observation_shape=(4, 64, 64),
        gray_scale=True,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        collect_max_episode_steps=int(2e4),
        eval_max_episode_steps=int(1e4),
        clip_rewards=True,
    ),
    policy=dict(
        analysis_sim_norm=False,
        cal_dormant_ratio=False,
        model_path=None,
        tokenizer_start_after_envsteps=int(0),
        transformer_start_after_envsteps=int(0),
        update_per_collect_transformer=update_per_collect,
        update_per_collect_tokenizer=update_per_collect,
        num_unroll_steps=num_unroll_steps,
        model=dict(
            analysis_sim_norm=False,
            observation_shape=(4, 64, 64),
            image_channel=1,
            frame_stack_num=4,
            gray_scale=True,
            action_space_size=action_space_size,
            downsample=True,
            self_supervised_learning_loss=True,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            reward_support_size=101,
            value_support_size=101,
            support_scale=50,
            world_model=dict(
                tokens_per_block=2,
                max_blocks=10,
                max_tokens=2 * 10,
                context_length=2 * 4,
                context_length_for_recurrent=2 * 4,
                recurrent_keep_deepth=100,
                gru_gating=False,
                device='cpu',
                analysis_sim_norm=False,
                analysis_dormant_ratio=False,
                action_shape=6,  # TODO：for pong qbert
                group_size=8,  # NOTE: sim_norm
                attention='causal',
                num_layers=4,  # TODO：for atari debug
                num_heads=8,
                embed_dim=768,  # TODO：for atari
                embed_pdrop=0.1,
                resid_pdrop=0.1,
                attn_pdrop=0.1,
                support_size=101,  # TODO
                max_cache_size=5000,
                env_num=8,
                latent_recon_loss_weight=0.,
                perceptual_loss_weight=0.,  # for stack1 rgb obs
                policy_entropy_weight=1e-4,
                predict_latent_loss_type='group_kl',
                obs_type='image',  # 'vector', 'image'
                gamma=1,
                dormant_threshold=0.025,
            ),
        ),
        use_priority=False,
        use_augmentation=False,
        cuda=True,
        env_type='not_board_games',
        game_segment_length=400,
        random_collect_episode_num=0,
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            type='linear',
            start=1.,
            end=0.01,
            decay=int(2e4),
        ),
        update_per_collect=update_per_collect,
        model_update_ratio=model_update_ratio,
        batch_size=batch_size,
        optim_type='AdamW',
        lr_piecewise_constant_decay=False,
        learning_rate=0.0001,
        target_update_freq=100,
        grad_clip_value=5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
atari_muzero_config = EasyDict(atari_muzero_config)
main_config = atari_muzero_config

atari_muzero_create_config = dict(
    env=dict(
        type='atari_lightzero',
        import_names=['zoo.atari.envs.atari_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='unizero',
        import_names=['lzero.policy.unizero'],
    ),
)
atari_muzero_create_config = EasyDict(atari_muzero_create_config)
create_config = atari_muzero_create_config

if __name__ == "__main__":
    seeds = [0, 1, 2]  # You can add more seed values here
    for seed in seeds:
        # Update exp_name to include the current seed
        main_config.exp_name = f'data_unizero_stack4/{env_id[:-14]}_stack4_unizero_upc{update_per_collect}-mur{model_update_ratio}_H{num_unroll_steps}_bs{batch_size}_seed{seed}'
        from lzero.entry import train_unizero
        train_unizero([main_config, create_config], seed=seed, max_env_step=max_env_step)
