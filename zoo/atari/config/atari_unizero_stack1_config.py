from easydict import EasyDict
from env_action_space_map import env_action_space_map

norm_type = 'BN'
env_id = 'PongNoFrameskip-v4'  # You can specify any Atari game here

action_space_size = env_action_space_map[env_id]

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
update_per_collect = None
replay_ratio = 0.25
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
max_env_step = int(5e5)
reanalyze_ratio = 0.
batch_size = 64
num_unroll_steps = 10

# ====== debug =====
# collector_env_num = 2
# n_episode = 2
# evaluator_env_num = 2
# num_simulations = 5
# max_env_step = int(5e5)
# reanalyze_ratio = 0.
# batch_size = 2
# num_unroll_steps = 10
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
        collect_max_episode_steps=int(2e4),
        eval_max_episode_steps=int(1e4),
        # TODO: debug
        # collect_max_episode_steps=int(50),
        # eval_max_episode_steps=int(50),
    ),
    policy=dict(
        model_path=None,
        num_unroll_steps=num_unroll_steps,
        model=dict(
            observation_shape=(3, 64, 64),
            action_space_size=action_space_size,
            downsample=True,
            norm_type=norm_type,
            world_model=dict(
                norm_type=norm_type,
                max_blocks=10,
                max_tokens=2 * 10,  # NOTE: each timestep has 2 tokens: obs and action
                context_length=2 * 4,
                device='cuda',
                action_shape=action_space_size,
                group_size=8,
                num_layers=4,
                num_heads=8,
                embed_dim=768,
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
                policy_entropy_weight=1e-4,
                obs_type='image',
            ),
        ),
        cuda=True,
        env_type='not_board_games',
        game_segment_length=400,
        update_per_collect=update_per_collect,
        replay_ratio=replay_ratio,
        batch_size=batch_size,
        optim_type='AdamW',
        learning_rate=0.0001,
        grad_clip_value=5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),
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

if __name__ == "__main__":
    # Define a list of seeds for multiple runs
    seeds = [0]  # You can add more seed values here
    for seed in seeds:
        # Update exp_name to include the current seed
        main_config.exp_name = f'data_unizero/{env_id[:-14]}_stack1_unizero_upc{update_per_collect}-rr{replay_ratio}_H{num_unroll_steps}_bs{batch_size}_normtype-{norm_type}_seed{seed}'
        from lzero.entry import train_unizero
        train_unizero([main_config, create_config], seed=seed, model_path=main_config.policy.model_path, max_env_step=max_env_step)
