from easydict import EasyDict

from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map


def main(env_id='PongNoFrameskip-v4', seed=0, max_env_step_override=int(5e5)):
    action_space_size = atari_env_action_space_map[env_id]
    collector_env_num = 8
    evaluator_env_num = 3
    game_segment_length = 20
    batch_size = 64
    num_unroll_steps = 10
    infer_context_length = 4
    num_layers = 2
    replay_ratio = 0.25

    main_config = EasyDict(dict(
        exp_name=(
            f'data_lz/data_unizero_ppo/{env_id[:-14]}/{env_id[:-14]}_uz_ppo_'
            f'nlayer{num_layers}_gsl{game_segment_length}_rr{replay_ratio}_'
            f'Htrain{num_unroll_steps}-Hinfer{infer_context_length}_bs{batch_size}_seed{seed}'
        ),
        env=dict(
            stop_value=int(1e6),
            env_id=env_id,
            observation_shape=(3, 64, 64),
            gray_scale=False,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False),
        ),
        policy=dict(
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000000))),
            model=dict(
                observation_shape=(3, 64, 64),
                action_space_size=action_space_size,
                world_model_cfg=dict(
                    policy_entropy_weight=1e-4,
                    continuous_action_space=False,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    device='cuda',
                    action_space_size=action_space_size,
                    num_layers=num_layers,
                    num_heads=8,
                    embed_dim=768,
                    obs_type='image',
                    encoder_type='resnet',
                    env_num=max(collector_env_num, evaluator_env_num),
                    rotary_emb=False,
                ),
            ),
            model_path=None,
            policy_improvement='ppo',
            collect_with_pure_policy=True,
            num_unroll_steps=num_unroll_steps,
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            # Kept identical to atari_unizero_config.py for a controlled comparison.
            learning_rate=0.0001,
            train_start_after_envsteps=2000,
            game_segment_length=game_segment_length,
            replay_buffer_size=int(1e6),
            eval_freq=int(5e3),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            ppo=dict(
                gamma=0.997,
                gae_lambda=0.95,
                clip_ratio=0.2,
                entropy_weight=0.01,
                epochs=4,
                minibatch_size=batch_size,
                normalize_advantage=True,
                target_kl=0.03,
                fresh_ratio_tolerance=1e-5,
                world_model_update_per_collect=None,
            ),
        ),
    ))
    create_config = EasyDict(dict(
        env=dict(
            type='atari_lightzero',
            import_names=['zoo.atari.envs.atari_lightzero_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(type='unizero', import_names=['lzero.policy.unizero']),
    ))

    from lzero.entry import train_unizero
    train_unizero(
        [main_config, create_config], seed=seed,
        model_path=main_config.policy.model_path, max_env_step=max_env_step_override,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train UniZero+PPO on Atari.')
    parser.add_argument('--env', default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-env-step', type=int, default=int(5e5))
    args = parser.parse_args()
    main(args.env, args.seed, args.max_env_step)
