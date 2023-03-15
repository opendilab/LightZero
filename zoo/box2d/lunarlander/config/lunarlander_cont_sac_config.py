"""
lunarlander_cont env:
    - obs_shape: 8
    - action_shape: 2
"""
from easydict import EasyDict

lunarlander_cont_sac_config = dict(
    exp_name='lunarlander_cont_sac_seed0',
    env=dict(
        stop_value=int(1e6),
        env_name='LunarLanderContinuous-v2',
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=10000,
        model=dict(
            obs_shape=8,
            action_shape=2,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=256,
            learning_rate_q=1e-3,
            learning_rate_policy=1e-3,
            learning_rate_alpha=3e-4,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            reparameterization=True,
            auto_alpha=False,
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(evaluator=dict(eval_freq=5000, )),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    ),
)
lunarlander_cont_sac_config = EasyDict(lunarlander_cont_sac_config)
main_config = lunarlander_cont_sac_config

lunarlander_cont_sac_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sac',
        import_names=['ding.policy.sac'],
    ),
    replay_buffer=dict(type='naive', ),
)
lunarlander_cont_sac_create_config = EasyDict(lunarlander_cont_sac_create_config)
create_config = lunarlander_cont_sac_create_config


def train(args):
    main_config.exp_name = 'data_lunarlander/sac_seed' + f'{args.seed}' + '_3M'
    serial_pipeline([copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed, max_env_step=int(3e6))


if __name__ == "__main__":
    import copy
    import argparse
    from ding.entry import serial_pipeline

    for seed in [0, 1, 2]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()
        train(args)
