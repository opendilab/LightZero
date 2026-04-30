from easydict import EasyDict
# ==============================================================
# Online PPO Configuration for LunarLander
# Based on ppo_bak.py default hyperparameters
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 4
n_episode = 4
evaluator_env_num = 3
num_simulations = 50          # 保持（虽然不用，但不影响）
reanalyze_ratio = 0.          # 保持
update_per_collect = 10       # 根据 ppo_bak: epoch_per_collect=10
replay_ratio = 0.0            # 改为 0：online 不需要回放
max_env_step = int(5e5)
batch_size = 64               # 根据 ppo_bak: batch_size=64
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

lunarlander_unizero_ppo_online_config = dict(
    exp_name=f'data_unizero_ppo/lunarlander_ppo_bak_upc{update_per_collect}_bs{batch_size}_lr3e4_seed0',
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
        game_segment_length=100,        # 改为 100：online 不需要长 segment
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='AdamW',
        piecewise_decay_lr_scheduler=False,
        learning_rate=5e-4,             # 根据 ppo_bak: learning_rate=3e-4
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        replay_buffer_size=int(1e5),    # 改为 1e5：online 不需要大缓冲
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        # Whether to use pure policy (without MCTS) for data collection
        collect_with_pure_policy=True,
        # Whether to use pure policy (without MCTS) for evaluation
        # If not set, will use collect_with_pure_policy value
        eval_with_pure_policy=True,
        # Whether to use online learning (clear replay_buffer after each training iteration)
        online_learning=True,
        # Value normalization for stable training (based on ppo_bak.py)
        value_norm=True,
        # PPO configuration for GAE computation (based on ppo_bak.py defaults)
        grad_clip_value=20,   
        ppo=dict(
            gamma=0.99,           # 根据 ppo_bak: discount_factor=0.99
            gae_lambda=0.95,      # 根据 ppo_bak: gae_lambda=0.95
            clip_ratio=0.2,       # 根据 ppo_bak: clip_ratio=0.2
            value_coef=0.5,       # 根据 ppo_bak: value_weight=0.5
            entropy_coef=0.01,     # 根据 ppo_bak: entropy_weight=0.0
        ),
        # Split training configuration (PriorZero-style)
        # When enabled, trains World Model and PPO separately with different data
        split_ppo_wm_training=False,  # Set to True to enable split training
        wm_update_per_collect=None,   # World Model updates per collect (uses all data)
        ppo_update_per_collect=None,  # PPO updates per collect (uses only new data)
        ppo_batch_size=None,          # Batch size for PPO training (defaults to batch_size)
    ),
)
lunarlander_unizero_ppo_online_config = EasyDict(lunarlander_unizero_ppo_online_config)
main_config = lunarlander_unizero_ppo_online_config

lunarlander_unizero_ppo_online_create_config = dict(
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
lunarlander_unizero_ppo_online_create_config = EasyDict(lunarlander_unizero_ppo_online_create_config)
create_config = lunarlander_unizero_ppo_online_create_config

if __name__ == "__main__":
    from lzero.entry import train_unizero
    train_unizero([main_config, create_config], seed=0, max_env_step=max_env_step)


