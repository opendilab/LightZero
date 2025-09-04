from easydict import EasyDict

from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map


def main(env_id='PongNoFrameskip-v4', seed=0):
    action_space_size = atari_env_action_space_map[env_id]

    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    collector_env_num = 8
    n_episode = 8
    evaluator_env_num = 3

    # collector_env_num = 1
    # n_episode = 1
    # evaluator_env_num = 1

    num_simulations = 50
    collect_num_simulations = 25
    # collect_num_simulations = 50
    eval_num_simulations = 50
    max_env_step = int(5e5)
    # max_env_step = int(50e6)
    batch_size = 256
    # batch_size = 64 # debug
    # batch_size = 4 # debug

    num_layers = 2
    # replay_ratio = 0.25
    replay_ratio = 0.1

    game_segment_length = 20
    num_unroll_steps = 10
    infer_context_length = 4

    # game_segment_length = 40
    # num_unroll_steps = 20
    # infer_context_length = 8

    # game_segment_length = 200
    # num_unroll_steps = 16
    # infer_context_length = 8

    # num_unroll_steps = 4 # TODO
    # infer_context_length = 2

    # Defines the frequency of reanalysis. E.g., 1 means reanalyze once per epoch, 2 means reanalyze once every two epochs.
    # buffer_reanalyze_freq = 1/50
    # buffer_reanalyze_freq = 1/10
    buffer_reanalyze_freq = 1/1000000000000

    # Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
    reanalyze_batch_size = 160
    # The partition of reanalyze. E.g., 1 means reanalyze_batch samples from the whole buffer, 0.5 means samples from the first half of the buffer.
    reanalyze_partition = 0.75

    # norm_type ="BN"
    norm_type ="LN"

    # ====== only for debug =====
    # collector_env_num = 2
    # num_segments = 2
    # evaluator_env_num = 2
    # num_simulations = 10
    # batch_size = 5
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
            # TODO: only for debug
            # collect_max_episode_steps=int(50),
            # eval_max_episode_steps=int(50),
        ),
        policy=dict(
            # learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000000, ), ), ),  # default is 10000
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=100000, ), ), ),  # 100k

            model=dict(
                observation_shape=(3, 64, 64),
                action_space_size=action_space_size,
                reward_support_range=(-300., 301., 1.),
                value_support_range=(-300., 301., 1.),
                world_model_cfg=dict(
                    game_segment_length=game_segment_length,

                    norm_type=norm_type,
                    num_res_blocks=2,
                    num_channels=128,
                    # num_res_blocks=1, # TODO
                    # num_channels=64,
                    support_size=601,
                    policy_entropy_weight=5e-3,
                    # policy_entropy_weight=5e-2, # TODO(pu)
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
                    encoder_type="resnet",
                    env_num=max(collector_env_num, evaluator_env_num),
                    num_simulations=num_simulations,
                    rotary_emb=False,
                    # rotary_emb=True,
                    # final_norm_option_in_encoder='LayerNorm_Tanh',
                    # final_norm_option_in_obs_head="LayerNorm",
                    # predict_latent_loss_type='mse',

                    # final_norm_option_in_encoder='L2Norm',
                    # final_norm_option_in_obs_head="L2Norm",
                    # predict_latent_loss_type='mse',

                    final_norm_option_in_encoder="LayerNorm",
                    final_norm_option_in_obs_head="LayerNorm",
                    predict_latent_loss_type='mse',

                    # final_norm_option_in_encoder="SimNorm",
                    # final_norm_option_in_obs_head="SimNorm",
                    # predict_latent_loss_type='group_kl',

                    # weight_decay=1e-2,
                    latent_norm_loss=True,

                    # latent_norm_loss=False,
                    weight_decay=1e-4, # TODO

                    use_priority=True, # TODO(pu): test
                ),
            ),
            # gradient_scale=True, #TODO
            gradient_scale=False, #TODO
            # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
            model_path=None,
            use_augmentation=False, # TODO

            use_priority=True, # TODO(pu): test
            priority_prob_alpha=1,
            priority_prob_beta=1,

            manual_temperature_decay=False,
            threshold_training_steps_for_final_temperature=int(2.5e4),
            num_unroll_steps=num_unroll_steps,
            update_per_collect=None,
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            optim_type='AdamW',
            # target_model_update_option="hard",
            target_update_freq=100,

            target_model_update_option="soft",
            # target_update_theta=0.005, # TODO
            # target_update_theta=0.01,
            target_update_theta=0.05,

            learning_rate=0.0001,
            # learning_rate=0.0003, # TODO

            num_simulations=50, # for reanalyze
            collect_num_simulations=collect_num_simulations,
            eval_num_simulations=eval_num_simulations,
            # num_segments=num_segments,
            n_episode=n_episode,
            td_steps=5,
            train_start_after_envsteps=0,
            # train_start_after_envsteps=2000, # TODO
            game_segment_length=game_segment_length,
            replay_buffer_size=int(1e6),
            eval_freq=int(5e3),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
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

    main_config.exp_name = f'data_unizero_longrun_20250904/{env_id[:-14]}/{env_id[:-14]}_uz_episode_envnum{collector_env_num}_nlayer{num_layers}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'
    from lzero.entry import train_unizero
    train_unizero([main_config, create_config], seed=seed, model_path=main_config.policy.model_path, max_env_step=max_env_step)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some environment.')
    parser.add_argument('--env', type=str, help='The environment to use', default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    args = parser.parse_args()




    # args.env = 'PongNoFrameskip-v4'

    args.env = 'MsPacmanNoFrameskip-v4'

    # args.env = 'QbertNoFrameskip-v4'
    # args.env = 'SeaquestNoFrameskip-v4' 

    # args.env = 'SpaceInvadersNoFrameskip-v4'
    # args.env = 'BeamRiderNoFrameskip-v4'
    # args.env = 'GravitarNoFrameskip-v4'

    # args.env = 'BreakoutNoFrameskip-v4'


    args.seed = 0


    main(args.env, args.seed)

    """
    export CUDA_VISIBLE_DEVICES=1
    cd /mnt/nfs/zhangjinouwen/puyuan/LightZero
    python /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/config/atari_unizero_config.py
    """
