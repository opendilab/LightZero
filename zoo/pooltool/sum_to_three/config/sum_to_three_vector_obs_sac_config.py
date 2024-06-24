from easydict import EasyDict

sum_to_three_sac_config = dict(
    exp_name='sum_to_three_vector-obs_cont_sac_seed0',
    env=dict(
        env_name="PoolTool-SumToThree",
        env_type="not_board_games",
        observation_type="coordinate",
        continuous=True,
        manually_discretization=False,
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=10,
        raw_observation=True,
    ),
    policy=dict(
        cuda=False,
        random_collect_size=1000,
        model=dict(
            obs_shape=4,
            action_shape=2,
            twin_critic=True,
            action_space='reparameterization',
        ),
        learn=dict(
            update_per_collect=256,
            batch_size=128,
            learning_rate_q=1e-3,
            learning_rate_policy=3e-4,
            learning_rate_alpha=3e-4,
            auto_alpha=True,
        ),
        collect=dict(n_sample=80, ),
        eval=dict(evaluator=dict(eval_freq=1000, ), ),
        other=dict(replay_buffer=dict(replay_buffer_size=int(1e6), ), ),
    ),
)
sum_to_three_sac_config = EasyDict(sum_to_three_sac_config)
main_config = sum_to_three_sac_config

sum_to_three_sac_create_config = dict(
    env=dict(
        type="pooltool_sumtothree",
        import_names=["zoo.pooltool.sum_to_three.envs.sum_to_three_env"],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='sac'),
)
sum_to_three_sac_create_config = EasyDict(sum_to_three_sac_create_config)
create_config = sum_to_three_sac_create_config

if __name__ == '__main__':
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0)