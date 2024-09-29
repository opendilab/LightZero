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
    evaluator_env_num = 5
    num_simulations = 50
    max_env_step = int(2e5)
    batch_size = 64

    num_layers = 4
    replay_ratio = 1
    num_unroll_steps = 10
    infer_context_length = 4

    # num_layers = 2
    # replay_ratio = 0.25
    # num_unroll_steps = 10
    # infer_context_length = 4

    # Defines the frequency of reanalysis. E.g., 1 means reanalyze once per epoch, 2 means reanalyze once every two epochs.
    buffer_reanalyze_freq = 1/10
    # Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
    reanalyze_batch_size = 160
    # The partition of reanalyze. E.g., 1 means reanalyze_batch samples from the whole buffer, 0.5 means samples from the first half of the buffer.
    reanalyze_partition=1

    # ====== only for debug =====
    # collector_env_num = 2
    # num_segments = 2
    # evaluator_env_num = 2
    # num_simulations = 2
    # update_per_collect = 2
    # batch_size = 2
    # ==============================================================
    # end of the most frequently changed config specified by the user
    # ==============================================================

    atari_unizero_config = dict(
        env=dict(
            stop_value=int(1e6),
            env_id=env_id,
            observation_shape=(3, 96, 96),
            gray_scale=False,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
            # TODO: only for debug
            # collect_max_episode_steps=int(20),
            # eval_max_episode_steps=int(20),
        ),
        policy=dict(
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000000, ), ), ),  # default is 10000
            model=dict(
                observation_shape=(3, 96, 96),
                action_space_size=action_space_size,
                world_model_cfg=dict(
                    policy_entropy_weight=0,
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
                    env_num=max(collector_env_num, evaluator_env_num),
                ),
            ),
            # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
            model_path=None,
            use_augmentation=False,
            manual_temperature_decay=False,
            threshold_training_steps_for_final_temperature=int(5e4),
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
            train_start_after_envsteps=2000,
            # train_start_after_envsteps=0,  # TODO
            game_segment_length=game_segment_length,
            grad_clip_value=20,
            replay_buffer_size=int(1e6),
            eval_freq=int(5e3),
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

    main_config.exp_name = f'data_unizero_reanalyze_0929/{env_id[:-14]}/{env_id[:-14]}_uz_fixvaluebugV8_td5_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'
    # main_config.exp_name = f'data_unizero_reanalyze_0929/{env_id[:-14]}/{env_id[:-14]}_uz_origin-buffer_td5_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'
    
    # ============ use muzero_segment_collector instead of muzero_collector =============
    from lzero.entry import train_unizero_reanalyze
    train_unizero_reanalyze([main_config, create_config], seed=seed, model_path=main_config.policy.model_path, max_env_step=max_env_step)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process different environments and seeds.')
    parser.add_argument('--env', type=str, help='The environment to use', default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    args = parser.parse_args()

    # args.env = 'QbertNoFrameskip-v4' # TODO

    main(args.env, args.seed)
