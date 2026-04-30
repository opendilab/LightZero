from easydict import EasyDict
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
reanalyze_ratio = 0.
update_per_collect = None
replay_ratio = 0.0            # 改为 0：split training 不需要额外 replay
max_env_step = int(5e5)
batch_size = 512              # WM 的 batch size
num_unroll_steps = 10
infer_context_length = 4
norm_type = 'BN'

# debug
# collector_env_num = 2
# n_episode = 2
# evaluator_env_num = 2
# num_simulations = 5
# batch_size = 2
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

lunarlander_unizero_ppo_config = dict(
    exp_name=f'data_unizero_ppo/lunarlander_unizero_ppo_ns{num_simulations}_upc{update_per_collect}-rr{replay_ratio}_rer{reanalyze_ratio}_H{num_unroll_steps}-infer{infer_context_length}_bs{batch_size}_{norm_type}_seed0',
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
        model=dict(
            observation_shape=8,
            action_space_size=4,
            model_type='mlp', 
            norm_type=norm_type,
            world_model_cfg=dict(
                continuous_action_space=False,
                max_blocks=num_unroll_steps,
                max_tokens=2 * num_unroll_steps,  # NOTE: each timestep has 2 tokens: obs and action
                context_length=2 * infer_context_length,
                device='cuda',
                action_space_size=4,
                group_size=8,  # NOTE: sim_norm
                num_layers=4,
                num_heads=4,
                embed_dim=256,
                env_num=max(collector_env_num, evaluator_env_num),
                obs_type='vector',
                norm_type=norm_type,
            ),
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        num_unroll_steps=num_unroll_steps,
        cuda=True,
        game_segment_length=100,        # 改为 100：PPO 标准值（on-policy）
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.0005,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        replay_buffer_size=int(1e5),    # 改为 1e5：split training 需要适中的 buffer
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        # Whether to use pure policy (without MCTS) for data collection
        collect_with_pure_policy=True,
        # Whether to use pure policy (without MCTS) for evaluation
        # If not set, will use collect_with_pure_policy value
        eval_with_pure_policy=True,
        # Whether to use online learning (clear replay_buffer after each training iteration)
        online_learning=False,          # 改为 False：split training 需要保留旧数据给 WM
        # Value normalization for stable training
        value_norm=True,
        # Gradient clipping for training stability (based on ppo_bak.py defaults)
        grad_clip_value=20,             # PPO 标准值（防止梯度爆炸）
        # Weight decay (L2 regularization) for training stability
        # weight_decay=0.0,                # PPO 标准（禁用 L2 正则化）
        # PPO configuration for GAE computation
        ppo=dict(
            gamma=0.99,           # Discount factor
            gae_lambda=0.95,      # GAE lambda parameter
            clip_ratio=0.2,       # PPO clipping ratio
            value_coef=0.5,       # Value loss coefficient
            entropy_coef=0.01,   # Entropy loss coefficient
        ),
        # Split training configuration (PriorZero-style)
        # When enabled, trains World Model and PPO separately with different data
        split_ppo_wm_training=True,  # Set to True to enable split training
        wm_update_per_collect=10,    # World Model updates per collect (uses all data)
        ppo_update_per_collect=5,    # PPO updates per collect (uses only new data)
        ppo_batch_size=64,           # Batch size for PPO training (PPO 标准值)
    ),
)
lunarlander_unizero_ppo_config = EasyDict(lunarlander_unizero_ppo_config)
main_config = lunarlander_unizero_ppo_config

lunarlander_unizero_ppo_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['zoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='unizero',
        import_names=['lzero.policy.unizero'],
    ),
)
lunarlander_unizero_ppo_create_config = EasyDict(lunarlander_unizero_ppo_create_config)
create_config = lunarlander_unizero_ppo_create_config

if __name__ == "__main__":
    from lzero.entry import train_unizero
    train_unizero([main_config, create_config], seed=0, max_env_step=max_env_step)

