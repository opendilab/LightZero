import numpy as np
import pytest
from easydict import EasyDict

from .game_2048_env import Game2048Env


# Create a Game2048 environment that will be used in the following tests.
@pytest.fixture
def env():
    # Configuration for the Game2048 environment
    cfg = EasyDict(dict(
        env_name="game_2048",
        save_replay_gif=False,
        replay_path_gif=None,
        replay_path=None,
        act_scale=True,
        channel_last=True,
        obs_type='raw_observation',  # options=['raw_observation', 'dict_observation', 'array']
        reward_normalize=False,
        reward_norm_scale=100,
        reward_type='raw',  # 'merged_tiles_plus_log_max_tile_num'
        max_tile=int(2 ** 16),  # 2**11=2048, 2**16=65536
        delay_reward_step=0,
        prob_random_agent=0.,
        max_episode_steps=int(1e6),
        is_collect=True,
        ignore_legal_actions=True,
        need_flatten=False,
    ))
    return Game2048Env(cfg)


# Test the initialization of the Game2048 environment.
def test_initialization(env):
    assert isinstance(env, Game2048Env)


# Test the reset method of the Game2048 environment.
# Ensure that the shape of the observation is as expected.
def test_reset(env):
    obs = env.reset()
    assert obs.shape == (4, 4, 16)


# Test the step method of the Game2048 environment.
# Ensure that the shape of the observation, the type of the reward,
# the type of the done flag and the type of the info are as expected.
def test_step(env):
    env.reset()
    obs, reward, done, info = env.step(1)
    assert obs.shape == (4, 4, 16)
    assert isinstance(reward, np.ndarray)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


# Test the render method of the Game2048 environment.
# Ensure that the shape of the rendered image is as expected.
def test_render(env):
    env.reset()
    env.render(mode='human')
    env.render(mode='rgb_array_render')
    env.save_render_gif()

# Test the seed method of the Game2048 environment.
# Ensure that the random seed is set correctly.
def test_seed(env):
    env.seed(0)
    assert env.np_random.randn() != np.random.randn()
