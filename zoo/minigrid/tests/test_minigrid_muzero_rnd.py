import numpy as np
from easydict import EasyDict

from zoo.minigrid.config.minigrid_muzero_rnd_1m_optimized_config import (
    main_config as optimized_config,
    max_env_step as optimized_max_env_step,
)
from zoo.minigrid.config.minigrid_muzero_rnd_config import (
    main_config as baseline_config,
    max_env_step as baseline_max_env_step,
)
from zoo.minigrid.envs.minigrid_lightzero_env import MiniGridEnvLightZero


def test_keycorridor_rnd_configs():
    assert baseline_config.env.env_id == 'MiniGrid-KeyCorridorS3R3-v0'
    assert baseline_config.env.max_step == 300
    assert baseline_config.policy.use_priority is True
    assert baseline_config.policy.use_max_priority_for_new_data is True
    assert baseline_max_env_step == int(2e6)

    assert optimized_max_env_step == int(1e6)
    assert optimized_config.policy.td_steps == 20
    assert optimized_config.policy.replay_buffer_size == int(3e5)
    assert optimized_config.reward_model.intrinsic_reward_weight_final == 0.0


def test_flat_observation_space_and_episode_horizon():
    env = MiniGridEnvLightZero(
        EasyDict(
            env_id='MiniGrid-KeyCorridorS3R3-v0',
            flat_obs=True,
            max_step=7,
            save_replay_gif=False,
            replay_path_gif=None,
        )
    )
    try:
        env.seed(0, dynamic_seed=False)
        obs = env.reset()

        assert env._env.unwrapped.max_steps == 7
        assert env.observation_space.contains(obs['observation'])
        assert obs['observation'].dtype == np.uint8

        timestep = env.step(np.asarray([0], dtype=np.int64))
        assert timestep.reward.shape == (1, )
        assert timestep.reward.dtype == np.float32
    finally:
        env.close()
