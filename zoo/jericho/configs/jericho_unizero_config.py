import os
from easydict import EasyDict


def main(env_id='zork1.z5', seed=0):
    action_space_size = 50

    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    collector_env_num = 8
    game_segment_length = 20
    evaluator_env_num = 5
    num_simulations = 50
    max_env_step = int(5e5)
    batch_size = 64
    num_unroll_steps = 10
    infer_context_length = 4
    num_layers = 2
    replay_ratio = 0.25
    # ==============================================================
    # end of the most frequently changed config specified by the user
    # ==============================================================
    jericho_unizero_config = dict(
        env=dict(
            stop_value=int(1e6),
            max_steps=100,
            max_action_num=action_space_size,
            tokenizer_path="google-bert/bert-base-uncased",
            max_seq_len=512,
            game_path="z-machine-games-master/jericho-game-suite/" + env_id,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, )
        ),
        policy=dict(
            # default is 10000
            learn=dict(learner=dict(
                hook=dict(save_ckpt_after_iter=1000000, ), ), ),
            model=dict(
                observation_shape=512,
                action_space_size=action_space_size,
                encoder_url='google-bert/bert-base-uncased',
                # The input of the model is text, whose shape is identical to the mlp model.
                model_type='mlp',
                world_model_cfg=dict(
                    policy_entropy_weight=5e-3,
                    continuous_action_space=False,
                    max_blocks=num_unroll_steps,
                    # NOTE: each timestep has 2 tokens: obs and action
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    device='cuda',
                    action_space_size=action_space_size,
                    num_layers=num_layers,
                    num_heads=8,
                    embed_dim=768,
                    obs_type='text',  # TODO: Change it.
                    env_num=max(collector_env_num, evaluator_env_num),
                ),
            ),
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            learning_rate=0.0001,
            num_simulations=num_simulations,
            train_start_after_envsteps=2000,
            game_segment_length=game_segment_length,
            replay_buffer_size=int(1e6),
            eval_freq=int(5e3),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
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
        policy=dict(
            type='unizero',
            import_names=['lzero.policy.unizero'],
        ),
    )
    jericho_unizero_create_config = EasyDict(jericho_unizero_create_config)
    main_config = jericho_unizero_config
    create_config = jericho_unizero_create_config

    main_config.exp_name = f'data_unizero/{env_id[:-14]}/{env_id[:-14]}_uz_nlayer{num_layers}_gsl{game_segment_length}_rr{replay_ratio}_Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'
    from lzero.entry import train_unizero
    train_unizero([main_config, create_config], seed=seed,
                  model_path=main_config.policy.model_path, max_env_step=max_env_step)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some environment.')
    parser.add_argument('--env', type=str,
                        help='The environment to use', default='zork1.z5')
    parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    args = parser.parse_args()

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main(args.env, args.seed)
