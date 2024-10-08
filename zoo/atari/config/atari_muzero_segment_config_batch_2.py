from easydict import EasyDict
from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map
# env_id = 'PongNoFrameskip-v4'  # You can specify any Atari game here

def main(env_id, seed):
    action_space_size = atari_env_action_space_map[env_id]

    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    collector_env_num = 8
    num_segments = 8
    game_segment_length=20

    n_episode = 8
    evaluator_env_num = 3
    num_simulations = 50
    update_per_collect = None
    replay_ratio = 0.25
    # replay_ratio = 1

    batch_size = 256
    max_env_step = int(1e5)
    reanalyze_ratio = 0.
    buffer_reanalyze_freq = 1/10  # modify according to num_segments
    reanalyze_batch_size = 160   # in total of num_unroll_steps
    reanalyze_partition=1
    num_unroll_steps = 5

    # =========== for debug ===========
    # collector_env_num = 1
    # n_episode = 1
    # evaluator_env_num = 1
    # num_simulations = 2
    # update_per_collect = 2
    # batch_size = 2
    # ==============================================================
    # end of the most frequently changed config specified by the user
    # ==============================================================

    atari_muzero_config = dict(
        env=dict(
            stop_value=int(1e6),
            env_id=env_id,
            # observation_shape=(4, 64, 64),  # (4, 96, 96)
            observation_shape=(4, 96, 96),  # (4, 96, 96)
            frame_stack_num=4,
            gray_scale=True,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
            # TODO: debug
            # collect_max_episode_steps=int(50),
            # eval_max_episode_steps=int(50),
        ),
        policy=dict(
            analysis_sim_norm=False,
            cal_dormant_ratio=False,
            model=dict(
                # observation_shape=(4, 64, 64),  # (4, 96, 96)
                observation_shape=(4, 96, 96),  # (4, 96, 96)
                image_channel=1,
                frame_stack_num=4,
                gray_scale=True,
                action_space_size=action_space_size,
                downsample=True,
                self_supervised_learning_loss=True,  # default is False
                discrete_action_encoding_type='one_hot',
                norm_type='BN',
                use_sim_norm=True,
                # use_sim_norm=False,
                use_sim_norm_kl_loss=False,
                model_type='conv'
            ),
            cuda=True,
            env_type='not_board_games',
            num_segments=num_segments,
            train_start_after_envsteps=2000,
            game_segment_length=game_segment_length,
            random_collect_episode_num=0,
            # use_augmentation=True,
            use_augmentation=False,
            use_priority=False,
            replay_ratio=replay_ratio,
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            optim_type='SGD',
            lr_piecewise_constant_decay=True,
            manual_temperature_decay=False,  # TODO
            learning_rate=0.2,
            target_update_freq=100,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            ssl_loss_weight=2,
            n_episode=n_episode,
            eval_freq=int(5e3),
            replay_buffer_size=int(1e6),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            threshold_training_steps_for_final_temperature=int(5e4),
            # ============= The key different params for ReZero =============
            num_unroll_steps=num_unroll_steps,
            buffer_reanalyze_freq=buffer_reanalyze_freq, # 1 means reanalyze one times per epoch, 2 means reanalyze one times each two epoch
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,
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
            type='muzero',
            import_names=['lzero.policy.muzero'],
        ),
        collector=dict(
            type='segment_muzero',
            import_names=['lzero.worker.muzero_segment_collector'],
        ),
    )
    atari_muzero_create_config = EasyDict(atari_muzero_create_config)
    create_config = atari_muzero_create_config

    main_config.exp_name = f'data_efficiency0829_plus_tune-mz_0924/{env_id[:-14]}/{env_id[:-14]}_mz_origcollect_haveseginit_temp0.25_rr{replay_ratio}_simnorm_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-only{reanalyze_partition}_eval5_collect{collector_env_num}-numsegments-{num_segments}_gsl{game_segment_length}_rer{reanalyze_ratio}_H{num_unroll_steps}_bs{batch_size}_seed{seed}'
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=seed, max_env_step=max_env_step)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some environment.')
    
    parser.add_argument('--env', type=str, help='The environment to use')
    parser.add_argument('--seed', type=int, help='The environment to use')
    
    args = parser.parse_args()
    main(args.env, args.seed)