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
        save_replay=False,
        replay_format='gif',
        replay_name_suffix='eval',
        replay_path=None,
        render_real_time=False,
        act_scale=True,
        channel_last=True,
        obs_type='raw_observation',  # options=['raw_observation', 'dict_observation', 'array']
        reward_type='raw',  # options=['raw', 'merged_tiles_plus_log_max_tile_num']
        reward_normalize=False,
        reward_norm_scale=100,
        max_tile=int(2 ** 16),  # 2**11=2048, 2**16=65536
        delay_reward_step=0,
        prob_random_agent=0.,
        max_episode_steps=int(1e6),
        is_collect=True,
        ignore_legal_actions=True,
        need_flatten=False,
        num_of_possible_chance_tile=2,
        possible_tiles=np.array([2, 4]),
        tile_probabilities=np.array([0.9, 0.1]),
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
def test_step_shape(env):
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


# Test the seed method of the Game2048 environment.
# Ensure that the random seed is set correctly.
def test_seed(env):
    env.seed(0)
    assert env.np_random.randn() != np.random.randn()


def test_step_action_case1(env):
    init_board = np.array([[8, 4, 0, 0],
                           [2, 0, 0, 0],
                           [2, 0, 0, 0],
                           [2, 4, 2, 0]])

    # Test action 0 (Assuming it represents 'up' move)
    env.reset(init_board=init_board, add_random_tile_flag=False)
    obs, reward, done, info = env.step(0)
    expected_board_up = np.array([[8, 8, 2, 0],
                                  [4, 0, 0, 0],
                                  [2, 0, 0, 0],
                                  [0, 0, 0, 0]])
    assert np.array_equal(env.board, expected_board_up)

    # Test action 1 (Assuming it represents 'right' move)
    env.reset(init_board=init_board, add_random_tile_flag=False)
    obs, reward, done, info = env.step(1)
    expected_board_right = np.array([[0, 0, 8, 4],
                                     [0, 0, 0, 2],
                                     [0, 0, 0, 2],
                                     [0, 2, 4, 2]])
    assert np.array_equal(env.board, expected_board_right)

    # Test action 2 (Assuming it represents 'down' move)
    env.reset(init_board=init_board, add_random_tile_flag=False)
    obs, reward, done, info = env.step(2)
    expected_board_down = np.array([[0, 0, 0, 0],
                                    [8, 0, 0, 0],
                                    [2, 0, 0, 0],
                                    [4, 8, 2, 0]])
    assert np.array_equal(env.board, expected_board_down)

    # Test action 3 (Assuming it represents 'left' move)
    env.reset(init_board=init_board, add_random_tile_flag=False)
    obs, reward, done, info = env.step(3)
    expected_board_left = np.array([[8, 4, 0, 0],
                                    [2, 0, 0, 0],
                                    [2, 0, 0, 0],
                                    [2, 4, 2, 0]])
    assert np.array_equal(env.board, expected_board_left)


def test_step_action_case2(env):
    init_board = np.array([[8, 4, 2, 0],
                           [2, 0, 2, 0],
                           [2, 2, 4, 0],
                           [2, 4, 2, 0]])

    # Test action 0 (Assuming it represents 'up' move)
    env.reset(init_board=init_board, add_random_tile_flag=False)
    obs, reward, done, info = env.step(0)
    expected_board_up = np.array([[8, 4, 4, 0],
                                  [4, 2, 4, 0],
                                  [2, 4, 2, 0],
                                  [0, 0, 0, 0]])
    assert np.array_equal(env.board, expected_board_up)

    # Test action 1 (Assuming it represents 'right' move)
    env.reset(init_board=init_board, add_random_tile_flag=False)
    obs, reward, done, info = env.step(1)
    expected_board_right = np.array([[0, 8, 4, 2],
                                     [0, 0, 0, 4],
                                     [0, 0, 4, 4],
                                     [0, 2, 4, 2]])
    assert np.array_equal(env.board, expected_board_right)

    # Test action 2 (Assuming it represents 'down' move)
    env.reset(init_board=init_board, add_random_tile_flag=False)
    obs, reward, done, info = env.step(2)
    expected_board_down = np.array([[0, 0, 0, 0],
                                    [8, 4, 4, 0],
                                    [2, 2, 4, 0],
                                    [4, 4, 2, 0]])
    assert np.array_equal(env.board, expected_board_down)

    # Test action 3 (Assuming it represents 'left' move)
    env.reset(init_board=init_board, add_random_tile_flag=False)
    obs, reward, done, info = env.step(3)
    expected_board_left = np.array([[8, 4, 2, 0],
                                    [4, 0, 0, 0],
                                    [4, 4, 0, 0],
                                    [2, 4, 2, 0]])
    assert np.array_equal(env.board, expected_board_left)