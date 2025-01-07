from easydict import EasyDict
env_id = 'visual_match'  # The name of the environment, options: 'visual_match', 'key_to_door'

# memory_length = 2 # DEBUG
memory_length = 500
# memory_length = 100

# max_env_step = int(1e6)  # for visual_match [2, 60, 100]
max_env_step = int(3e6)  # for visual_match [250,500]

# embed_dim=256 
# num_layers=2
# num_heads=2

# embed_dim=256 
# num_layers=8
# num_heads=8



embed_dim=128
num_layers=12
num_heads=8

# memory_length = 500
# max_env_step = int(3e6)  # for visual_match [100,250,500]
# embed_dim=256 # for visual_match [100,250,500]
# num_layers=8
# num_heads=8
# ==============================================================
# begin of the most frequently changed config specified by the user,
# you should change the following configs to adapt to your own task
# ==============================================================
# for key_to_door
# num_unroll_steps = 30+memory_length
# game_segment_length = 30+memory_length # TODO: for "explore": 15

# for visual_match
num_unroll_steps = 16 + memory_length
game_segment_length = 16 + memory_length  # TODO: for "explore": 1
seed = 0
collector_env_num = 8
n_episode = 8
evaluator_env_num = 10

num_simulations = 50
# update_per_collect = None
# update_per_collect = 10
update_per_collect = 50

replay_ratio = 0.1
# batch_size = 160 # 32*5 = 160
batch_size = 64 # 32*5 = 160
reanalyze_ratio = 0
# td_steps = 10
# td_steps = 5
td_steps = game_segment_length

# eps_greedy_exploration_in_collect = True

# ========= only for debug ===========
# collector_env_num = 2
# n_episode = 2
# evaluator_env_num = 2
# num_simulations = 3
# update_per_collect = None
# replay_ratio = 0.25
# batch_size = 4
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================
memory_unizero_config = dict(
    # exp_name=f'data_{env_id}_1025_clean/{env_id}_memlen-{memory_length}_unizero_H{num_unroll_steps}_bs{batch_size}_seed{seed}',
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        flatten_observation=False,  # Whether to flatten the observation
        max_frames={
            # ================ Maximum frames per phase =============
            # "explore": 15, # TODO: for key_to_door
            "explore": 1,  # for visual_match
            "distractor": memory_length,
            "reward": 15
        },
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000000,),),),  # default is 10000
        sample_type='episode',  # NOTE: very important for memory env
        model=dict(
            observation_shape=(3, 5, 5),
            action_space_size=4,
            world_model_cfg=dict(
                # In order to preserve the observation data of the first frame in a memory environment,
                # we must ensure that we do not exceed the episode_length during the MCTS of the last frame.
                # Therefore, we set a longer context_length than during training to ensure that the observation data of the first frame is not lost.
                max_blocks=num_unroll_steps + 5,
                max_tokens=2 * (num_unroll_steps + 5),
                context_length=2 * (num_unroll_steps + 5),
                # device='cpu',
                device='cuda',
                action_space_size=4,
                num_layers=num_layers,
                num_heads=num_heads,
                embed_dim=embed_dim,
                env_num=max(collector_env_num, evaluator_env_num),
                obs_type='image_memory',
                policy_entropy_weight=5e-3,
            ),
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        num_unroll_steps=num_unroll_steps,
        td_steps=td_steps,
        # discount_factor=1,
        discount_factor=0.99,
        # cuda=True,
        game_segment_length=game_segment_length,
        replay_ratio=replay_ratio,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='AdamW',
        learning_rate=1e-4,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(5e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

memory_unizero_config = EasyDict(memory_unizero_config)
main_config = memory_unizero_config

memory_unizero_create_config = dict(
    env=dict(
        type='memory_lightzero',
        import_names=['zoo.memory.envs.memory_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='unizero',
        import_names=['lzero.policy.unizero'],
    ),
)
memory_unizero_create_config = EasyDict(memory_unizero_create_config)
create_config = memory_unizero_create_config

if __name__ == "__main__":
    # seeds = [0, 1, 2]  # You can add more seed values here
    seeds = [1]  # You can add more seed values here
    for seed in seeds:
        main_config.exp_name = f'data_{env_id}_1202/{env_id}_memlen-{memory_length}_fixedcolormap_obs10value05_td{td_steps}_layer{num_layers}-head{num_heads}_unizero_edim{embed_dim}_H{num_unroll_steps}_bs{batch_size}_upc{update_per_collect}_seed{seed}'
        # main_config.exp_name = f'data_{env_id}_1122/{env_id}_memlen-{memory_length}_randomcolormap/obs10value05_td{td_steps}_layer{num_layers}-head{num_heads}_unizero_edim{embed_dim}_H{num_unroll_steps}_bs{batch_size}_upc{update_per_collect}_seed{seed}'
        from lzero.entry import train_unizero
        train_unizero([main_config, create_config], seed=seed, model_path=main_config.policy.model_path, max_env_step=max_env_step)