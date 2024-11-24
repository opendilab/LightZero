from easydict import EasyDict
env_id = 'visual_match'  # The name of the environment, options: 'visual_match', 'key_to_door'

# memory_length = 60
# max_env_step = int(5e5)  # for visual_match [2, 60]

memory_length = 100
max_env_step = int(1e6)  # for visual_match [2, 60]
num_layers = 16
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
# seed = 0
collector_env_num = 8
n_episode = 8
evaluator_env_num = 10

num_simulations = 50
update_per_collect = 50
replay_ratio = 0.25
batch_size = 64
reanalyze_ratio = 0
# td_steps = 5
td_steps = game_segment_length


# ========= only for debug ===========
# collector_env_num = 2
# n_episode = 2
# evaluator_env_num = 2
# num_simulations = 5
# update_per_collect = None
# replay_ratio = 0.25
# batch_size = 4
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================
memory_unizero_config = dict(
    # exp_name=f'data_{env_id}/{env_id}_memlen-{memory_length}_unizero_H{num_unroll_steps}_bs{batch_size}_seed{seed}',
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
                num_heads=num_layers,
                embed_dim=256,
                env_num=max(collector_env_num, evaluator_env_num),
                obs_type='image_memory',

                use_normal_head=True,
                use_softmoe_head=False,
                use_moe_head=False,
                num_experts_in_moe_head=4,  # NOTE
                moe_in_transformer=False,  # NOTE
                multiplication_moe_in_transformer=False,  # NOTE
                num_experts_of_moe_in_transformer=4,
            ),
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        num_unroll_steps=num_unroll_steps,
        td_steps=td_steps,
        discount_factor=0.9,
        # cuda=True,
        game_segment_length=game_segment_length,
        replay_ratio=replay_ratio,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='AdamW',
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
    seeds = [0, 1, 2]  # You can add more seed values here
    # seeds = [0]  # You can add more seed values here
    for seed in seeds:
        main_config.exp_name = f'data_{env_id}_1124/{env_id}_memlen-{memory_length}_nlayer{num_layers}_unizero_H{num_unroll_steps}_bs{batch_size}_seed{seed}'
        from lzero.entry import train_unizero
        train_unizero([main_config, create_config], seed=seed, model_path=main_config.policy.model_path, max_env_step=max_env_step)