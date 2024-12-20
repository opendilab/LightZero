import os
from easydict import EasyDict
# import os
# os.environ["HF_HOME"] = "/mnt/afs/zhangshenghan/.cache/huggingface/hub"

def main(env_id='detective.z5', seed=0):
    # action_space_size = 50
    action_space_size = 10
    max_steps = 50

    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    # collector_env_num = 8
    # n_episode = 8
    collector_env_num = 4
    n_episode = 4
    evaluator_env_num = 2
    num_simulations = 50
    max_env_step = int(10e6)

    # batch_size = 8
    # num_unroll_steps = 10
    # infer_context_length = 4

    batch_size = 16
    num_unroll_steps = 5
    infer_context_length = 2

    num_layers = 2
    replay_ratio = 0.25
    update_per_collect = None # NOTE: very important for ddp
    embed_dim = 768
    # Defines the frequency of reanalysis. E.g., 1 means reanalyze once per epoch, 2 means reanalyze once every two epochs.
    # buffer_reanalyze_freq = 1/10
    buffer_reanalyze_freq = 1/100000
    # Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
    reanalyze_batch_size = 160
    # The partition of reanalyze. E.g., 1 means reanalyze_batch samples from the whole buffer, 0.5 means samples from the first half of the buffer.
    reanalyze_partition = 0.75
    model_name = 'BAAI/bge-base-en-v1.5'
    # model_name = 'google-bert/bert-base-uncased'
    # =========== TODO: only for debug  =========== 
    # collector_env_num = 2
    # num_segments = 2
    # game_segment_length = 20
    # evaluator_env_num = 2
    # max_env_step = int(5e5)
    # batch_size = 10
    # num_simulations = 5
    # num_unroll_steps = 5
    # infer_context_length = 2
    # max_steps = 10
    # num_layers = 1
    # replay_ratio = 0.05
    # embed_dim = 768
    # TODO: MCTS内部的action_space受限于root节点的legal action

    # ==============================================================
    # end of the most frequently changed config specified by the user
    # ==============================================================
    jericho_unizero_config = dict(
        env=dict(
            stop_value=int(1e6),
            observation_shape=512,
            max_steps=max_steps,
            max_action_num=action_space_size,
            tokenizer_path=model_name,
            # tokenizer_path="/mnt/afs/zhangshenghan/.cache/huggingface/hub/models--google-bert--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594",
            max_seq_len=512,
            # game_path="z-machine-games-master/jericho-game-suite/" + env_id,
            game_path="/mnt/afs/niuyazhe/code/LightZero/zoo/jericho/envs/z-machine-games-master/jericho-game-suite/"+ env_id,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, )
        ),
        policy=dict(
            multi_gpu=False, # ======== Very important for ddp =============
            # multi_gpu=True, # ======== Very important for ddp =============
            # default is 10000
            use_wandb=False,
            learn=dict(learner=dict(
                hook=dict(save_ckpt_after_iter=1000000, ), ), ),
            model=dict(
                observation_shape=512,
                action_space_size=action_space_size,
                encoder_url=model_name,
                # encoder_url='google-bert/bert-base-uncased',
                # encoder_url='/mnt/afs/zhangshenghan/.cache/huggingface/hub/models--google-bert--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594',
                # The input of the model is text, whose shape is identical to the mlp model.
                model_type='mlp',
                continuous_action_space=False,
                world_model_cfg=dict(
                    policy_entropy_weight=5e-2,
                    continuous_action_space=False,
                    max_blocks=num_unroll_steps,
                    # NOTE: each timestep has 2 tokens: obs and action
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    device='cuda',
                    action_space_size=action_space_size,
                    num_layers=num_layers,
                    num_heads=8,
                    embed_dim=embed_dim,
                    obs_type='text',  # TODO: Change it.
                    env_num=max(collector_env_num, evaluator_env_num),
                ),
            ),
            update_per_collect=update_per_collect,
            action_type='varied_action_space',
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            reanalyze_ratio=0,
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            learning_rate=0.0001,
            cos_lr_scheduler=True,
            manual_temperature_decay=True,
            threshold_training_steps_for_final_temperature=int(2.5e4),
            num_simulations=num_simulations,
            # num_segments=num_segments,
            n_episode=n_episode,
            train_start_after_envsteps=0, # TODO
            # game_segment_length=game_segment_length,
            # replay_buffer_size=int(1e6),
            replay_buffer_size=int(1e5),
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
    jericho_unizero_config = EasyDict(jericho_unizero_config)

    jericho_unizero_create_config = dict(
        env=dict(
            type='jericho',
            import_names=['zoo.jericho.envs.jericho_env'],
        ),
        # NOTE: use base env manager to avoid the bug of subprocess env manager.
        env_manager=dict(type='base'),
        # env_manager=dict(type='subprocess'),
        policy=dict(
            type='unizero',
            import_names=['lzero.policy.unizero'],
        ),
    )
    jericho_unizero_create_config = EasyDict(jericho_unizero_create_config)
    main_config = jericho_unizero_config
    create_config = jericho_unizero_create_config

    main_config.exp_name = f'data_unizero_detective_20250107/{model_name}/{env_id[:8]}_ms{max_steps}_action-space-{action_space_size}_uz_nlayer{num_layers}_rr{replay_ratio}-upc{update_per_collect}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'
    from lzero.entry import train_unizero
    train_unizero([main_config, create_config], seed=seed,
                  model_path=main_config.policy.model_path, max_env_step=max_env_step)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some environment.')
    parser.add_argument('--env', type=str,
                help='The environment to use', default='detective.z5') # 'detective.z5'  'zork1.z5'                 
    parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    args = parser.parse_args()

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main(args.env, args.seed)
