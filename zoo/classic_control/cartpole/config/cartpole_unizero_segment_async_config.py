from easydict import EasyDict
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
# collector_env_num = 8
# num_segments = 8
# n_episode = 8

collector_env_num = 3
num_segments = 3
n_episode = 3

game_segment_length = 20
evaluator_env_num = 3
# num_simulations = 25
num_simulations = 10 # TODO

update_per_collect = None
replay_ratio = 0.25
# max_env_step = int(2e5)
max_env_step = int(3e3) # TODO

batch_size = 256
num_unroll_steps = 5
reanalyze_ratio = 0.
# Defines the frequency of reanalysis. E.g., 1 means reanalyze once per epoch, 2 means reanalyze once every two epochs.
# buffer_reanalyze_freq = 1/50
buffer_reanalyze_freq = 1/50000000
# buffer_reanalyze_freq = 1 # TODO

# Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
reanalyze_batch_size = 160
# The partition of reanalyze. E.g., 1 means reanalyze_batch samples from the whole buffer, 0.5 means samples from the first half of the buffer.
reanalyze_partition = 0.75

# ============= 异步训练相关配置 =============
# 是否启用异步训练
enable_async_training = True
# 数据缓冲队列大小
data_queue_size = 10
# 评估器检查间隔（秒）
evaluator_check_interval = 2.0
# 线程同步超时时间（秒）
thread_timeout = 2.0
# 是否输出异步训练的详细调试信息
enable_async_debug_log = True
# enable_async_debug_log = False

# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================
cartpole_unizero_config = dict(
    exp_name=f'data_unizero_async/cartpole_unizero_ns{num_simulations}_upc{update_per_collect}-rr{replay_ratio}_H{num_unroll_steps}_bs{batch_size}_seed0',
    env=dict(
        env_name='CartPole-v0',
        continuous=False,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        # learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000, ), ), ),
        learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=100, ), ), ),
        model=dict(
            observation_shape=4,
            action_space_size=2,
            self_supervised_learning_loss=True,  # NOTE: default is False.
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            model_type='mlp',
            world_model_cfg=dict(
                final_norm_option_in_obs_head='LayerNorm',
                final_norm_option_in_encoder='LayerNorm',
                predict_latent_loss_type='mse',
                max_blocks=10,
                max_tokens=2 * 10,
                context_length=2 * 4,
                context_length_for_recurrent=2 * 4,
                # device='cuda',
                device='cpu',
                action_space_size=2,
                num_layers=2,
                num_heads=2,
                embed_dim=64,
                env_num=max(collector_env_num, evaluator_env_num),
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
                obs_type='vector',
                norm_type='BN',
                # rotary_emb=True,
                rotary_emb=False,
            ),
        ),
        use_wandb=False,
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        num_unroll_steps=num_unroll_steps,
        # cuda=True,
        # device='cuda',
        cuda=False,
        device='cpu',
        use_augmentation=False,
        env_type='not_board_games',
        num_segments=num_segments,
        game_segment_length=game_segment_length,
        replay_ratio=replay_ratio,
        batch_size=batch_size,
        optim_type='AdamW',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.0001,
        target_update_freq=100,
        grad_clip_value=5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        # eval_freq=int(1e4),
        # eval_freq=int(1e3),
        eval_freq=int(50), # TODO
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        # ============= The key different params for reanalyze =============
        # Defines the frequency of reanalysis. E.g., 1 means reanalyze once per epoch, 2 means reanalyze once every two epochs.
        buffer_reanalyze_freq=buffer_reanalyze_freq,
        # Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
        reanalyze_batch_size=reanalyze_batch_size,
        # The partition of reanalyze. E.g., 1 means reanalyze_batch samples from the whole buffer, 0.5 means samples from the first half of the buffer.
        reanalyze_partition=reanalyze_partition,
        # ============= 异步训练配置 =============
        enable_async_training=enable_async_training,
        data_queue_size=data_queue_size,
        evaluator_check_interval=evaluator_check_interval,
        thread_timeout=thread_timeout,
        enable_async_debug_log=enable_async_debug_log,
    ),
)

cartpole_unizero_config = EasyDict(cartpole_unizero_config)
main_config = cartpole_unizero_config

cartpole_unizero_create_config = dict(
    env=dict(
        type='cartpole_lightzero',
        import_names=['zoo.classic_control.cartpole.envs.cartpole_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='unizero',
        import_names=['lzero.policy.unizero'],
    ),
)
cartpole_unizero_create_config = EasyDict(cartpole_unizero_create_config)
create_config = cartpole_unizero_create_config

if __name__ == "__main__":
    from lzero.entry import train_unizero_segment_async
    train_unizero_segment_async([main_config, create_config], seed=0, max_env_step=max_env_step) 