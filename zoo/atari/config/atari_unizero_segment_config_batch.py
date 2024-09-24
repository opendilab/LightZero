from easydict import EasyDict
from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map

def main(env_id, seed):

    action_space_size = atari_env_action_space_map[env_id]

    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    update_per_collect = None
    # replay_ratio = 0.25
    replay_ratio = 1
    # replay_ratio = 0.5

    collector_env_num = 8 # TODO
    num_segments = 8

    # collector_env_num = 4 # TODO
    # num_segments = 4
    # game_segment_length=10
    # collector_env_num = 1 # TODO
    # num_segments = 1

    game_segment_length=20

    evaluator_env_num = 5  # TODO
    num_simulations = 50
    max_env_step = int(5e5)  # TODO

    reanalyze_ratio = 0.

    batch_size = 64
    num_unroll_steps = 10
    infer_context_length = 4

    # num_unroll_steps = 5
    # infer_context_length = 4

    num_layers = 4
    buffer_reanalyze_freq = 1/10  # modify according to num_segments
    # buffer_reanalyze_freq = 1/5  # modify according to num_segments
    # buffer_reanalyze_freq = 1/2  # modify according to num_segments

    reanalyze_batch_size = 160   # in total of num_unroll_steps
    # reanalyze_batch_size = 640   # in total of num_unroll_steps
    # reanalyze_partition=3/4
    reanalyze_partition=1



    # ====== only for debug =====
    # collector_env_num = 8
    # num_segments = 8
    # evaluator_env_num = 2
    # num_simulations = 5
    # max_env_step = int(2e5)
    # reanalyze_ratio = 0.1
    # batch_size = 64
    # num_unroll_steps = 10
    # replay_ratio = 0.01

    # ==============================================================
    # end of the most frequently changed config specified by the user
    # ==============================================================

    atari_unizero_config = dict(
        env=dict(
            stop_value=int(1e6),
            env_id=env_id,
            # observation_shape=(3, 64, 64),
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
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000000,),),),  # default is 10000
            model=dict(
                # observation_shape=(3, 64, 64),
                observation_shape=(3, 96, 96),
                action_space_size=action_space_size,
                world_model_cfg=dict(
                    policy_entropy_weight=0,  # NOTE
                    # policy_entropy_weight=1e-4,
                    continuous_action_space=False,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,  # NOTE: each timestep has 2 tokens: obs and action
                    context_length=2 * infer_context_length,
                    device='cuda',
                    # device='cpu',
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
            # model_path='/mnt/afs/niuyazhe/code/LightZero/data_efficiency0829_plus_tune-uz_0914/numsegments-8_gsl20_origin-target-value-policy/Pong_stack1_unizero_upcNone-rr0.25_H10_bs64_seed0_nlayer2/ckpt/ckpt_best.pth.tar',
            # use_augmentation=True,
            use_augmentation=False,

            # manual_temperature_decay=True,  # TODO
            manual_temperature_decay=False,  # TODO
            # threshold_training_steps_for_final_temperature=int(2.5e4),
            threshold_training_steps_for_final_temperature=int(5e4),
            # manual_temperature_decay=False,  # TODO

            # use_priority=True, # TODO
            use_priority=False, # TODO

            num_unroll_steps=num_unroll_steps,
            update_per_collect=update_per_collect,
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            optim_type='AdamW',
            learning_rate=0.0001,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            num_segments=num_segments,
            train_start_after_envsteps=2000,
            game_segment_length=game_segment_length, # debug
            grad_clip_value=20,
            replay_buffer_size=int(1e6),
            eval_freq=int(5e3),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            # ============= The key different params for ReZero =============
            buffer_reanalyze_freq=buffer_reanalyze_freq, # 1 means reanalyze one times per epoch, 2 means reanalyze one times each two epoch
            reanalyze_batch_size=reanalyze_batch_size,
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
        collector=dict(
            type='segment_muzero',
            import_names=['lzero.worker.muzero_segment_collector'],
        ),
        evaluator=dict(
            type='muzero',
            import_names=['lzero.worker.muzero_evaluator'],
        )
    )
    atari_unizero_create_config = EasyDict(atari_unizero_create_config)
    create_config = atari_unizero_create_config


    # main_config.exp_name = f'data_efficiency0829_plus_tune-uz_0920/{env_id[:-14]}/{env_id[:-14]}_uz_nlayer{num_layers}_eval5_collect{collector_env_num}-numsegments-{num_segments}_gsl{game_segment_length}_temp025_upc{update_per_collect}-rr{replay_ratio}_rer{reanalyze_ratio}_H{num_unroll_steps}-infer{infer_context_length}_bs{batch_size}_seed{seed}'
    # from lzero.entry import train_unizero
    # train_unizero([main_config, create_config], seed=seed, model_path=main_config.policy.model_path, max_env_step=max_env_step)

    main_config.exp_name = f'data_efficiency0829_plus_tune-uz_0923/{env_id[:-14]}/{env_id[:-14]}_uz_temp025_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-only{reanalyze_partition}_nlayer{num_layers}_eval5_collect{collector_env_num}-numsegments-{num_segments}_gsl{game_segment_length}_upc{update_per_collect}-rr{replay_ratio}_rer{reanalyze_ratio}_H{num_unroll_steps}-infer{infer_context_length}_bs{batch_size}_seed{seed}'
    from lzero.entry import train_rezero_uz
    train_rezero_uz([main_config, create_config], seed=seed, model_path=main_config.policy.model_path, max_env_step=max_env_step)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some environment.')
    
    parser.add_argument('--env', type=str, help='The environment to use')
    parser.add_argument('--seed', type=int, help='The environment to use')
    
    args = parser.parse_args()
    main(args.env, args.seed)

