from easydict import EasyDict
from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map
# # 在main文件开始，通过全局变量来控制是否处于调试状态
# global DEBUG_ENABLED;DEBUG_ENABLED = True 

def main(env_id, seed):
    action_space_size = atari_env_action_space_map[env_id]

    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    collector_env_num = 8
    num_segments = 8
    evaluator_env_num = 3

    # collector_env_num = 1
    # num_segments = 1
    # evaluator_env_num = 1

    num_simulations = 50
    collect_num_simulations = 25
    # collect_num_simulations = 50
    eval_num_simulations = 50
    max_env_step = int(5e6)
    # max_env_step = int(50e6)
    batch_size = 256
    # batch_size = 16 # debug
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
            # observation_shape=(3, 96, 96),

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
            # learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=100000, ), ), ),  # 100k
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=50000, ), ), ),  # 50k

            # sample_type='episode',  # NOTE: very important for memory env
            model=dict(
                observation_shape=(3, 64, 64),
                # observation_shape=(3, 96, 96),
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

                    # latent_norm_loss=True,
                    latent_norm_loss=False,
                    weight_decay=1e-4, # TODO

                    use_priority=True, # TODO(pu): test

                    # optim_type='SGD',
                    # learning_rate=0.01,

                    optim_type='AdamW',
                    # learning_rate=0.001,
                    learning_rate=0.0001,

                    # entry_norm=True, # TODO========
                    entry_norm=False, # TODO========

                    use_temperature_scaling=True,

                    # optim_type='AdamW_mix_lr',
                    # learning_rate=0.001,

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

            # manual_temperature_decay=True,
            # threshold_training_steps_for_final_temperature=int(5e4),

            manual_temperature_decay=False,
            threshold_training_steps_for_final_temperature=int(2.5e4),
            
            num_unroll_steps=num_unroll_steps,
            update_per_collect=None,
            replay_ratio=replay_ratio,
            batch_size=batch_size,

            piecewise_decay_lr_scheduler=False,
            optim_type='AdamW',
            # learning_rate=0.001,
            learning_rate=0.0001,


            # latent_norm_clip_threshold=3, # 768dim
            latent_norm_clip_threshold=5, # 768dim latent encoder

            # latent_norm_clip_threshold=20, # 768dim
            # latent_norm_clip_threshold=30, # 768dim

            # logit_clip_threshold=10, # value reward
            # policy_logit_clip_threshold=5, # policy

            logit_clip_threshold=9999, # value reward
            policy_logit_clip_threshold=99999, # policy

            # piecewise_decay_lr_scheduler=False,
            # optim_type='AdamW_mix_lr',
            # learning_rate=0.001,


            # optim_type='SGD', # TODO
            # piecewise_decay_lr_scheduler=True,
            # # learning_rate=0.2,
            # learning_rate=0.01,


            # target_model_update_option="hard",
            target_update_freq=100,

            target_model_update_option="soft",
            # target_update_theta=0.005, # TODO
            # target_update_theta=0.01,
            target_update_theta=0.05,



            num_simulations=50, # for reanalyze
            collect_num_simulations=collect_num_simulations,
            eval_num_simulations=eval_num_simulations,
            num_segments=num_segments,
            td_steps=5,
            # train_start_after_envsteps=0,
            train_start_after_envsteps=2000, # TODO
            game_segment_length=game_segment_length,
            grad_clip_value=5,

            backbone_grad_clip_value=5,
            # head_grad_clip_value=0.5,
            head_grad_clip_value=5,  # TODO


            # replay_buffer_size=int(1e6),
            replay_buffer_size=int(5e5), # TODO

            eval_freq=int(5e3),
            # eval_freq=int(1e4), # TODO
            # eval_freq=int(2e4),
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
    main_config.exp_name = f'data_unizero_longrun_20250918/{env_id[:-14]}/{env_id[:-14]}_uz_adamw1e-4_temp-scale_encoder-clip-5_fix-clip_reinit-value-reward-policy-50k_envnum{collector_env_num}_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_c25_seed{seed}'

    # main_config.exp_name = f'data_unizero_longrun_20250918_debug/{env_id[:-14]}/{env_id[:-14]}_uz_adamw1e-4_encoder-clip-5_fix-clip_reinit-value-reward-policy-50k_envnum{collector_env_num}_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_c25_seed{seed}'

    # main_config.exp_name = f'data_unizero_longrun_20250910/{env_id[:-14]}/{env_id[:-14]}_uz_adamw1e-4_encoder-clip-5-true_head-clip10-policy5_muzerohead_fix-clip_reinit-value-reward-policy-50k_envnum{collector_env_num}_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_c25_seed{seed}'


    # main_config.exp_name = f'data_unizero_longrun_20250910/{env_id[:-14]}/{env_id[:-14]}_uz_adamw1e-4_encoder-clip-5_head-clip10-policy05_entry-norm_clipgrad-backbone5-head5_reinit-value-reward-policy-50k_envnum{collector_env_num}_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_c25_seed{seed}'

    # main_config.exp_name = f'data_unizero_longrun_20250910/{env_id[:-14]}/{env_id[:-14]}_uz_adamw1e-4_encoder-clip-5_entry-norm_clipgrad-backbone5-head05_grad-scale_reinit-value-reward-policy-50k_head-clip10-policy05_envnum{collector_env_num}_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_c25_seed{seed}'

    # main_config.exp_name = f'data_unizero_longrun_20250910/{env_id[:-14]}/{env_id[:-14]}_uz_adamw1e-4_encoder-clip-5_entry-norm_clipgrad-backbone5-head05_grad-scale_reinit-value-reward-policy-50k_clear{game_segment_length}_originlossweight_spsi{game_segment_length}_envnum{collector_env_num}_soft-target-005_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_c25_seed{seed}'

    # main_config.exp_name = f'data_unizero_longrun_20250910/{env_id[:-14]}/{env_id[:-14]}_uz_adamw1e-4_encoder-clip-5_entry-norm_clipgrad-backbone5-head05_clear{game_segment_length}_originlossweight_spsi{game_segment_length}_envnum{collector_env_num}_soft-target-005_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_c25_seed{seed}'


    # main_config.exp_name = f'data_unizero_longrun_20250910/{env_id[:-14]}/{env_id[:-14]}_uz_adamwlr1e-4_96_clear{game_segment_length}_originlossweight_spsi{game_segment_length}_envnum{collector_env_num}_soft-target-005_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_c25_seed{seed}'

    # main_config.exp_name = f'data_unizero_longrun_20250910/{env_id[:-14]}/{env_id[:-14]}_uz_adammixlr1e-3_in-value-reward-head-ln2_per_lnlw1e-4_enc-LN_fix-init-recur_encoder-head-ln_clear{game_segment_length}_originlossweight_spsi{game_segment_length}_envnum{collector_env_num}_soft-target-005_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_c25_seed{seed}'

    # main_config.exp_name = f'data_unizero_longrun_20250910/{env_id[:-14]}/{env_id[:-14]}_uz_sgd001_clear{game_segment_length}_originlossweight_spsi{game_segment_length}_envnum{collector_env_num}_soft-target-005_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_c25_seed{seed}'

    # main_config.exp_name = f'data_unizero_longrun_20250910/{env_id[:-14]}/{env_id[:-14]}_uz_adamw1e-3_in-value-reward-head-ln2_per_lnlw1e-4_enc-LN_fix-init-recur_encoder-head-ln_clear{game_segment_length}_originlossweight_spsi{game_segment_length}_envnum{collector_env_num}_soft-target-005_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_c25_seed{seed}'
    
    # main_config.exp_name = f'data_unizero_longrun_20250901_debug/{env_id[:-14]}/{env_id[:-14]}_uz_per_lnlw1e-4_enc-BN_fix-init-recur_encoder-head-ln_clear{game_segment_length}_originlossweight_spsi{game_segment_length}_envnum{collector_env_num}_soft-target-005_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_c25_seed{seed}'

    # main_config.exp_name = f'data_unizero_longrun_20250901/{env_id[:-14]}/{env_id[:-14]}_uz_per_grad-scale_pew005_fix-init-recur_encoder-head-ln_clear{game_segment_length}_originlossweight_spsi{game_segment_length}_envnum{collector_env_num}_soft-target-005_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_c25_seed{seed}'

    # main_config.exp_name = f'data_unizero_longrun_20250827/{env_id[:-14]}/{env_id[:-14]}_uz_per_lnlw0001_pew005_fix-init-recur_encoder-head-ln_clear{game_segment_length}_originlossweight_spsi{game_segment_length}_envnum{collector_env_num}_soft-target-005_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_c25_seed{seed}'

    # main_config.exp_name = f'data_unizero_longrun_20250827/{env_id[:-14]}/{env_id[:-14]}_uz_current-next-latent_fix-init-recur_encoder-head-simnorm_clear{game_segment_length}_originlossweight_spsi{game_segment_length}_envnum{collector_env_num}_soft-target-005_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_c25_seed{seed}'

    # main_config.exp_name = f'data_unizero_longrun_20250827/{env_id[:-14]}/{env_id[:-14]}_uz_current-next-latent_latent-norm-weight001_fix-init-recur_clear{game_segment_length}_originlossweight_spsi{game_segment_length}_envnum{collector_env_num}_encoder-head-ln_soft-target-005_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_c50_seed{seed}'

    train_unizero_segment([main_config, create_config], seed=seed, model_path=main_config.policy.model_path, max_env_step=max_env_step)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process different environments and seeds.')
    parser.add_argument('--env', type=str, help='The environment to use', default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    args = parser.parse_args()

    args.env = 'PongNoFrameskip-v4'

    # args.env = 'MsPacmanNoFrameskip-v4'

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

    python /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/config/atari_unizero_segment_config.py > /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/logs/unizero_adamw1e-4_64_encoder-clip-5_fix-clip_temp-scale_reinit-value-reward-policy-50k_pong.log 2>&1
    

    python /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/config/atari_unizero_segment_config.py > /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/logs/unizero_adamw1e-4_64_encoder-clip-5-true_head-clip10-pol5_fix-clip_mz-head_reinit-value-reward-policy-50k_msp.log 2>&1

    python /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/config/atari_unizero_segment_config.py > /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/logs/unizero_adamw1e-4_64_encoder-clip-5_head-clip10-pol5_fix-clip_reinit-value-reward-policy-50k.log 2>&1
    
    python /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/config/atari_unizero_segment_config.py > /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/logs/unizero_adamw1e-4_64_encoder-clip-5_entry-norm_clipgrad-backbone5-head5_reinit-value-reward-policy-50k_head-clip10-pol05_msp.log 2>&1

    python /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/config/atari_unizero_segment_config.py > /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/logs/unizero_adamw1e-4_64_encoder-clip-5_entry-norm_clipgrad-backbone5-head05_grad-scale_reinit-value-reward-policy-50k_head-clip10-pol05_pong.log 2>&1

    python /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/config/atari_unizero_segment_config.py > /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/logs/unizero_adamw1e-4_96.log 2>&1

    python /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/config/atari_unizero_segment_config.py > /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/logs/unizero_sgd_02-0002.log 2>&1
    python /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/config/atari_unizero_segment_config.py > /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/logs/unizero_sgd_001.log 2>&1

    python /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/config/atari_unizero_segment_config.py > /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/logs/unizero_adamw1e-3.log 2>&1
    python /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/config/atari_unizero_segment_config.py > /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/logs/unizero_adamw1e-4_96.log 2>&1

    python /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/config/atari_unizero_segment_config.py > /mnt/nfs/zhangjinouwen/puyuan/LightZero/zoo/atari/logs/unizero_adamw-mix-1e-3.log 2>&1


    """
