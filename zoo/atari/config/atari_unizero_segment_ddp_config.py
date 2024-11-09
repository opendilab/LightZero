from easydict import EasyDict
from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map
import torch.distributed as dist


def main(env_id, seed):
    action_space_size = atari_env_action_space_map[env_id]

    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    gpu_num = 4
    collector_env_num = 8
    num_segments = int(collector_env_num*gpu_num)
    game_segment_length = 20  # segment_length固定，但可能最后一局的长度不足，导致upc不一样，所以需要对ddp下面的upc进行特殊处理
    evaluator_env_num = 10 # TODO
    num_simulations = 50
    max_env_step = int(5e5) # TODO

    batch_size = 64
    num_layers = 2
    replay_ratio = 0.25
    # replay_ratio = 0.5

    # num_layers = 4
    # replay_ratio = 1

    num_unroll_steps = 10
    infer_context_length = 4

    # Defines the frequency of reanalysis. E.g., 1 means reanalyze once per epoch, 2 means reanalyze once every two epochs.
    buffer_reanalyze_freq = 1/100000
    # buffer_reanalyze_freq = 1/10

    # Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
    reanalyze_batch_size = 160
    # The partition of reanalyze. E.g., 1 means reanalyze_batch samples from the whole buffer, 0.5 means samples from the first half of the buffer.
    reanalyze_partition = 0.75

    # ====== only for debug =====
    # evaluator_env_num = 2
    # num_simulations = 10
    # batch_size = 5
    # gpu_num = 4
    # collector_env_num = 2
    # num_segments = int(collector_env_num*gpu_num)
    # ==============================================================
    # end of the most frequently changed config specified by the user
    # ==============================================================

    atari_unizero_config = dict(
        env=dict(
            stop_value=int(1e6),
            env_id=env_id,
            observation_shape=(3, 96, 96),
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
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000000, ), ), ),  # default is 10000
            model=dict(
                observation_shape=(3, 96, 96),
                action_space_size=action_space_size,
                support_scale=300, # TODO
                world_model_cfg=dict(
                    support_size=601,  # TODO
                    policy_entropy_weight=5e-3, # TODO
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
                ),
            ),
            # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
            model_path=None,
            use_augmentation=False,
            manual_temperature_decay=False,
            # manual_temperature_decay=True,
            threshold_training_steps_for_final_temperature=int(2.5e4),
            use_priority=False,
            num_unroll_steps=num_unroll_steps,
            update_per_collect=None,
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            optim_type='AdamW',
            learning_rate=0.0001,
            num_simulations=num_simulations,
            num_segments=num_segments,
            td_steps=5, # TODO
            train_start_after_envsteps=2000,
            game_segment_length=game_segment_length,
            grad_clip_value=5,
            replay_buffer_size=int(1e6),
            eval_freq=int(5e3),
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
    from ding.utils import DDPContext
    from lzero.config.utils import lz_to_ddp_config
    with DDPContext():
        main_config = lz_to_ddp_config(main_config)
        # 确保每个 Rank 分配到正确的 collector_env_num
        print(f"Rank {dist.get_rank()} Collector Env Num: {main_config.policy.collector_env_num}")
        from lzero.entry import train_unizero_segment
        main_config.exp_name = f'data_unizero_ddp/{env_id[:-14]}/{env_id[:-14]}_uz_ddp_{gpu_num}gpu_scale300_pew5e-3_obs10value01_brf{buffer_reanalyze_freq}-rbs{reanalyze_batch_size}-rp{reanalyze_partition}_nlayer{num_layers}_numsegments-{num_segments}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'
        train_unizero_segment([main_config, create_config], seed=seed, model_path=main_config.policy.model_path, max_env_step=max_env_step)


if __name__ == "__main__":
    """
    Overview:
        This script should be executed with <nproc_per_node> GPUs.
        Run the following command to launch the script:
        python -m torch.distributed.launch --nproc_per_node=4 ./zoo/atari/config/atari_unizero_segment_ddp_config.py
        torchrun --nproc_per_node=2 ./zoo/atari/config/atari_unizero_multigpu_ddp_config.py
    """

    # import argparse
    # parser = argparse.ArgumentParser(description='Process different environments and seeds.')
    # parser.add_argument('--env', type=str, help='The environment to use', default='PongNoFrameskip-v4')
    # parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    # args = parser.parse_args()

    # args.env = 'QbertNoFrameskip-v4'
    # args.env = 'MsPacmanNoFrameskip-v4'
    # args.env = 'RoadRunnerNoFrameskip-v4'

    # main(args.env, args.seed)

    main('PongNoFrameskip-v4', 0)



