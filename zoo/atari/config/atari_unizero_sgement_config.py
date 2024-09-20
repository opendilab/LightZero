from easydict import EasyDict
from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map

env_id = 'PongNoFrameskip-v4'  # You can specify any Atari game here
# env_id = 'SeaquestNoFrameskip-v4'  # You can specify any Atari game here
# env_id = 'QbertNoFrameskip-v4'  # You can specify any Atari game here


action_space_size = atari_env_action_space_map[env_id]

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
update_per_collect = None
replay_ratio = 0.25
# replay_ratio = 0.1

# replay_ratio = 1

collector_env_num = 8
num_segments = 8

# collector_env_num = 4
# num_segments = 4


# num_segments = 1
game_segment_length=20
# game_segment_length=15
# game_segment_length=50
# game_segment_length=100
# game_segment_length=400

evaluator_env_num = 3
num_simulations = 50
max_env_step = int(2e5)

# reanalyze_ratio = 0.1
reanalyze_ratio = 0.

batch_size = 64
num_unroll_steps = 10
infer_context_length = 4

num_layers = 2

# ====== only for debug =====
collector_env_num = 8
num_segments = 8
evaluator_env_num = 2
num_simulations = 5
max_env_step = int(2e5)
reanalyze_ratio = 0.
batch_size = 64
num_unroll_steps = 10
replay_ratio = 0.05

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
        collect_max_episode_steps=int(20),
        eval_max_episode_steps=int(20),
    ),
    policy=dict(
        learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=100000,),),),  # default is 10000
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
        threshold_training_steps_for_final_temperature=int(2.5e4),
        # manual_temperature_decay=False,  # TODO

        # use_priority=True, # TODO
        use_priority=False, # TODO

        num_unroll_steps=num_unroll_steps,
        update_per_collect=update_per_collect,
        replay_ratio=replay_ratio,
        batch_size=batch_size,
        optim_type='AdamW',
        # learning_rate=0.0001,
        learning_rate=0.1,  # TODO
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        num_segments=num_segments,
        train_start_after_envsteps=2000,
        game_segment_length=game_segment_length, # debug
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

if __name__ == "__main__":
    # Define a list of seeds for multiple runs
    seeds = [0]  # You can add more seed values here
    for seed in seeds:
        # Update exp_name to include the current seed
        # main_config.exp_name = f'data_efficiency0829_plus_tune-uz_0920/{env_id[:-14]}/{env_id[:-14]}_uz_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_temp025_upc{update_per_collect}-rr{replay_ratio}_rer{reanalyze_ratio}_H{num_unroll_steps}-infer{infer_context_length}_bs{batch_size}_seed{seed}'
        # main_config.exp_name = f'data_efficiency0829_plus_tune-uz_0917/numsegments-{num_segments}_gsl{game_segment_length}_origin-target-value-policy_pew0_fixsample_temp025_useprio/{env_id[:-14]}_stack1_unizero_upc{update_per_collect}-rr{replay_ratio}_rer{reanalyze_ratio}_H{num_unroll_steps}_bs{batch_size}_seed{seed}_nlayer2'

        main_config.exp_name = f'data_efficiency0829_plus_tune-uz_debug/numsegments-{num_segments}_gsl{game_segment_length}_fix/obshape96_use-augmentation-obsw10/{env_id[:-14]}_stack1_unizero_upc{update_per_collect}-rr{replay_ratio}_H{num_unroll_steps}_bs{batch_size}_seed{seed}_nlayer2'

        from lzero.entry import train_unizero
        train_unizero([main_config, create_config], seed=seed, model_path=main_config.policy.model_path, max_env_step=max_env_step)


    # from lzero.entry import train_unizero
    # main_config.exp_name = f'data_unizero_efficiency_cprofile_250k/{env_id[:-14]}_stack1_unizero_upc{update_per_collect}-rr{replay_ratio}_H{num_unroll_steps}_bs{batch_size}_seed0_nlayer2_all-share-pool-_copy_0827'
    # def run(max_env_step: int):
    #     train_unizero([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)
    # import cProfile
    # cProfile.run(f"run({250000})", filename="pong_uz_cprofile_250k_envstep_allpool_s0", sort="cumulative")
