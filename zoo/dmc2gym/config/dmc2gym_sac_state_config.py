from easydict import EasyDict

dmc2gym_sac_config = dict(
    exp_name='dmc2gym_cartpole_balance_sac_state_seed0',
    env=dict(
        env_id='dmc2gym-v0',
        stop_value=900,
        domain_name='cartpole',  # obs shape: 5, action shape: 1
        task_name="balance",
        # task_name="swingup",

        # domain_name="acrobot",  # obs shape: 6, action shape: 1
        # task_name="swingup",

        # domain_name="hopper",  # obs shape: 15, action shape: 4
        # task_name="hop",
        #
        # domain_name="manipulator",  # obs shape: 37, action shape: 5
        # task_name="bring_ball",
        frame_skip=8,
        frame_stack=1,
        from_pixels=False,  # state obs
        channels_first=False,  # obs shape (height, width, 3)
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        battle_mode='one_player_mode',
        prob_random_agent=0.,
        collect_max_episode_steps=int(1.08e4),
        eval_max_episode_steps=int(1.08e5),
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model_type='state',
        cuda=True,
        random_collect_size=10000,
        model=dict(
            obs_shape=5,
            action_shape=1,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        learn=dict(
            ignore_done=True,
            update_per_collect=1,
            batch_size=256,
            learning_rate_q=1e-3,
            learning_rate_policy=1e-3,
            learning_rate_alpha=3e-4,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            reparameterization=True,
            auto_alpha=True,
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(evaluator=dict(eval_freq=int(500), )),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    ),
)

dmc2gym_sac_config = EasyDict(dmc2gym_sac_config)
main_config = dmc2gym_sac_config

dmc2gym_sac_create_config = dict(
    env=dict(
        type='dmc2gym',
        import_names=['dizoo.dmc2gym.envs.dmc2gym_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sac',
        import_names=['ding.policy.sac'],
    ),
    replay_buffer=dict(type='naive', ),
)
dmc2gym_sac_create_config = EasyDict(dmc2gym_sac_create_config)
create_config = dmc2gym_sac_create_config


if __name__ == "__main__":
    import copy
    import argparse
    from ding.entry import serial_pipeline

    # for seed in [0, 1, 2]:
    for seed in [0]:

        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()

        main_config.exp_name = 'dmc2gym_cartpole_balance_sac_state_' + 'seed' + f'{args.seed}'
        serial_pipeline([copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed,
                        max_env_step=int(1e6))
