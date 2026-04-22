import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from easydict import EasyDict
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
num_segments = 8
evaluator_env_num = 3
num_simulations = 50
reanalyze_ratio = 0.
update_per_collect = None
replay_ratio = 0.25
# replay_ratio = 0.1
max_env_step = int(1e6)
batch_size = 256
num_unroll_steps = 10
infer_context_length = 4
num_layers = 2
norm_type = 'BN'
game_segment_length = 200

buffer_reanalyze_freq = 1/5000000000
reanalyze_batch_size = 160
reanalyze_partition = 0.75

# debug
# collector_env_num = 2
# num_segments = 2
# evaluator_env_num = 2
# num_simulations = 5
# batch_size = 2
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

lunarlander_image_unizero_config = dict(
    exp_name=f'data_unizero_0422/lunarlander_image_unizero_ns{num_simulations}_upc{update_per_collect}-rr{replay_ratio}_rer{reanalyze_ratio}_H{num_unroll_steps}-infer{infer_context_length}_bs{batch_size}_{norm_type}_seed0',
    env=dict(
        env_id='LunarLander-v2',
        observation_shape=(3, 64, 64),
        gray_scale=False,
        continuous=False,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        collect_max_episode_steps=int(1000),
        eval_max_episode_steps=int(1000),
    ),
    policy=dict(
        learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000000, ), ), ),
        model=dict(
            observation_shape=(3, 64, 64),
            action_space_size=4,
            norm_type=norm_type,
            # ====== [FIX] support range must cover LunarLander reward/value range (-200 ~ +300) ======
            reward_support_range=(-300., 301., 1.),
            value_support_range=(-300., 301., 1.),
            num_res_blocks=1,
            num_channels=64,
            world_model_cfg=dict(
                continuous_action_space=False,
                max_blocks=num_unroll_steps,
                max_tokens=2 * num_unroll_steps,
                context_length=2 * infer_context_length,
                device='cuda',
                action_space_size=4,
                num_layers=num_layers,
                num_heads=4,
                embed_dim=256,
                obs_type='image',
                encoder_type='resnet',
                group_size=8,
                norm_type=norm_type,
                env_num=max(collector_env_num, evaluator_env_num),
                support_size=601,
                # Normalization options
                final_norm_option_in_encoder='LayerNorm',
                final_norm_option_in_obs_head='LayerNorm',
                predict_latent_loss_type='mse',
                # Task embedding (single-task, disabled)
                task_embed_option=None,
                # MoE (disabled for single-task baseline)
                moe_in_transformer=False,
                multiplication_moe_in_transformer=False,
                # Misc
                policy_entropy_weight=5e-3,
                num_simulations=num_simulations,
                game_segment_length=game_segment_length,
                rotary_emb=False,
                latent_recon_loss_weight=0.,
                perceptual_loss_weight=0.,
                decode_loss_mode=None,
                use_priority=False,
                use_normal_head=True,
                use_softmoe_head=False,
                use_moe_head=False,
                # optim_type='AdamW_mix_lr_wdecay',
                optim_type='AdamW',
            ),
        ),
        model_path=None,
        num_unroll_steps=num_unroll_steps,
        cuda=True,
        game_segment_length=game_segment_length,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        # optim_type='AdamW_mix_lr_wdecay',
        # weight_decay=1e-2,
        optim_type='AdamW',
        # weight_decay=1e-2,
        learning_rate=0.0001,
        piecewise_decay_lr_scheduler=False,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        num_segments=num_segments,
        replay_ratio=replay_ratio,
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        # ====== [FIX] grad clip: 20 -> 5, prevent gradient explosion ======
        grad_clip_value=5,
        # ====== [FIX] Priority Experience Replay ======
        # use_priority=True,
        use_priority=False,
        priority_prob_alpha=1,
        priority_prob_beta=1,
        # ====== [FIX] Adaptive entropy weight ======
        # use_adaptive_entropy_weight=True,
        use_adaptive_entropy_weight=False,
        adaptive_entropy_alpha_lr=1e-4,
        target_entropy_start_ratio=0.98,
        target_entropy_end_ratio=0.7,
        target_entropy_decay_steps=100000,
        # ====== [FIX] Encoder-clip annealing ======
        # use_encoder_clip_annealing=True,
        use_encoder_clip_annealing=False,
        encoder_clip_anneal_type='cosine',
        encoder_clip_start_value=30.0,
        encoder_clip_end_value=10.0,
        encoder_clip_anneal_steps=100000,
        # ====== [FIX] Label smoothing ======
        policy_ls_eps_start=0.05,
        policy_ls_eps_end=0.01,
        policy_ls_eps_decay_steps=50000,
        label_smoothing_eps=0.1,
        # ====== Monitor ======
        monitor_norm_freq=10000,
        eval_freq=int(5e3),
        td_steps=5,
        train_start_after_envsteps=0,
        use_augmentation=True,
        manual_temperature_decay=False,
        # ============= Reanalyze =============
        buffer_reanalyze_freq=buffer_reanalyze_freq,
        reanalyze_batch_size=reanalyze_batch_size,
        reanalyze_partition=reanalyze_partition,
    ),
)
lunarlander_image_unizero_config = EasyDict(lunarlander_image_unizero_config)
main_config = lunarlander_image_unizero_config

lunarlander_image_unizero_create_config = dict(
    env=dict(
        type='lunarlander_image',
        import_names=['zoo.box2d.lunarlander.envs.lunarlander_image_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='unizero',
        import_names=['lzero.policy.unizero'],
    ),
)
lunarlander_image_unizero_create_config = EasyDict(lunarlander_image_unizero_create_config)
create_config = lunarlander_image_unizero_create_config

if __name__ == "__main__":
    # ====== [FIX] use train_unizero_segment (segment-based collector) instead of train_unizero ======
    from lzero.entry import train_unizero_segment
    train_unizero_segment([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)
