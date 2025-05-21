from easydict import EasyDict
from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map


def main(env_id, seed):
    action_space_size = atari_env_action_space_map[env_id]

    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    collector_env_num = 8
    num_segments = 8
    game_segment_length = 20
    evaluator_env_num = 3
    num_simulations = 50
    # max_env_step = int(4e5)
    # max_env_step = int(1e6)
    max_env_step = int(100e6)

    batch_size = 64
    num_layers = 2
    # replay_ratio = 0.25
    replay_ratio = 0.1

    num_unroll_steps = 10
    infer_context_length = 4

    # Defines the frequency of reanalysis. E.g., 1 means reanalyze once per epoch, 2 means reanalyze once every two epochs.
    # buffer_reanalyze_freq = 1/50
    buffer_reanalyze_freq = 1/10000
    # Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
    reanalyze_batch_size = 160
    # The partition of reanalyze. E.g., 1 means reanalyze_batch samples from the whole buffer, 0.5 means samples from the first half of the buffer.
    reanalyze_partition = 0.75

    # ====== only for debug =====
    # collector_env_num = 2
    # num_segments = 2
    # evaluator_env_num = 2
    # num_simulations = 5
    # batch_size = 5
    # buffer_reanalyze_freq = 1/1000000
    # ==============================================================
    # end of the most frequently changed config specified by the user
    # ==============================================================

    atari_unizero_config = dict(
        env=dict(
            stop_value=int(1e6),
            env_id=env_id,
            observation_shape=(3, 64, 64),
            gray_scale=False,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
            # collect_max_episode_steps=int(5e3),
            # eval_max_episode_steps=int(5e3),
            # TODO: only for debug
            # collect_max_episode_steps=int(20),
            # eval_max_episode_steps=int(20),
        ),
        policy=dict(
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000000, ), ), ),  # default is 10000
            model=dict(
                observation_shape=(3, 64, 64),
                action_space_size=action_space_size,
                support_scale=300,
                world_model_cfg=dict(
                    # final_norm_option_in_obs_head='LayerNorm',
                    # final_norm_option_in_encoder='LayerNorm',
                    # predict_latent_loss_type='mse', # TODO: only for latent state layer_norm
                    
                    final_norm_option_in_obs_head='SimNorm',
                    final_norm_option_in_encoder='SimNorm',
                    predict_latent_loss_type='group_kl', # TODO: only for latent state sim_norm
                    
                    # analysis_dormant_ratio_weight_rank=True, # TODO

                    analysis_dormant_ratio_weight_rank=False, # TODO
                    dormant_threshold=0.025,
                    task_embed_option=None,   # ==============TODO: none ==============
                    use_task_embed=False, # ==============TODO==============
                    use_shared_projection=False,
                    support_size=601,
                    policy_entropy_weight=5e-3,
                    continuous_action_space=False,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,  # NOTE: each timestep has 2 tokens: obs and action
                    context_length=2 * infer_context_length,
                    device='cuda',
                    action_space_size=action_space_size,
                    num_layers=num_layers,
                    num_heads=8,
                    embed_dim=768,
                    obs_type='image',
                    
                    encoder_type='vit',
                    # encoder_type='resnet',

                    env_num=max(collector_env_num, evaluator_env_num),
                    num_simulations=num_simulations,
                    rotary_emb=False,
                    use_normal_head=True,
                    use_softmoe_head=False,
                    use_moe_head=False,
                    num_experts_in_moe_head=4,
                    moe_in_transformer=False,
                    multiplication_moe_in_transformer=False,
                    num_experts_of_moe_in_transformer=4,
                    # LoRA 参数：
                    lora_r= 0,
                    lora_alpha =1,
                    lora_dropout= 0.0,
                ),
            ),
            # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
            model_path=None,
            use_augmentation=False,
            manual_temperature_decay=False,
            threshold_training_steps_for_final_temperature=int(2.5e4),
            use_priority=False,
            num_unroll_steps=num_unroll_steps,
            update_per_collect=None,
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            optim_type='AdamW',
            learning_rate=0.0001,
            num_simulations=num_simulations,
            num_segments=num_segments,
            td_steps=5,
            train_start_after_envsteps=0, # only for debug
            # train_start_after_envsteps=2000,
            game_segment_length=game_segment_length,
            grad_clip_value=5,
            replay_buffer_size=int(1e6),
            # replay_buffer_size=int(5e5),
            # eval_freq=int(5e3),
            eval_freq=int(1e4),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            # ============= The key different params for reanalyze =============
            # Defines the frequency of reanalysis. E.g., 1 means reanalyze once per epoch, 2 means reanalyze once every two epochs.
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            # Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
            reanalyze_batch_size=reanalyze_batch_size,
            # The partition of reanalyze. E.g., 1 means reanalyze_batch samples from the whole buffer, 0.5 means samples from the first half of the buffer.
            reanalyze_partition=reanalyze_partition,
        ),
    )
    atari_unizero_config = EasyDict(atari_unizero_config)
    main_config = atari_unizero_config

    atari_unizero_create_config = dict(
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
    atari_unizero_create_config = EasyDict(atari_unizero_create_config)
    create_config = atari_unizero_create_config

    # ============ use muzero_segment_collector instead of muzero_collector =============
    from lzero.entry import train_unizero_segment
    main_config.exp_name = f'data_unizero_20250521/{env_id[:-14]}/{env_id[:-14]}_uz_vit-encoder-ps8-finalsimnorm_obs-kl-loss_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'
    train_unizero_segment([main_config, create_config], seed=seed, model_path=main_config.policy.model_path, max_env_step=max_env_step)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process different environments and seeds.')
    parser.add_argument('--env', type=str, help='The environment to use', default='MsPacmanNoFrameskip-v4')
    parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    args = parser.parse_args()
    main(args.env, args.seed)
