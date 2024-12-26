import numpy as np
import pytest
from easydict import EasyDict

from .game_2048_env import Game2048Env


@pytest.mark.unittest
class TestGame2048:
    def setup_method(self, method) -> None:
        # Configuration for the Game2048 environment
        cfg = EasyDict({
            'env_id': "game_2048",
            'render_mode': None,  # Options: 'None', 'state_realtime_mode', 'image_realtime_mode', 'image_savefile_mode'
            'replay_format': 'gif',
            'replay_name_suffix': 'eval',
            'replay_path': None,
            'act_scale': True,
            'channel_last': False,
            'obs_type': 'raw_encoded_board',  # Options: 'raw_board', 'raw_encoded_board', 'dict_encoded_board'
            'reward_type': 'raw',  # Options: ['raw', 'merged_tiles_plus_log_max_tile_num']
            'reward_normalize': False,
            'reward_norm_scale': 100,
            'max_tile': int(2 ** 16),  # 2**11=2048, 2**16=65536
            'delay_reward_step': 0,
            'prob_random_agent': 0.,
            'max_episode_steps': int(1e6),
            'is_collect': True,
            'ignore_legal_actions': True,
            'need_flatten': False,
            'num_of_possible_chance_tile': 2,
            'possible_tiles': np.array([2, 4]),
            'tile_probabilities': np.array([0.9, 0.1]),
        })
        # Create a Game2048 environment that will be used in the following tests.
        self.env = Game2048Env(cfg)

    # Test the initialization of the Game2048 environment.
    def test_initialization(self):
        assert isinstance(self.env, Game2048Env), "Environment is not an instance of Game2048Env"

    # Test the reset method of the Game2048 environment.
    # Ensure that the shape of the observation is as expected.
    def test_reset(self):
        obs = self.env.reset()
        assert obs.shape == (16, 4, 4), f"Expected observation shape (16, 4, 4), got {obs.shape}"

    # Test the step method of the Game2048 environment.
    # Ensure that the shape of the observation, the type of the reward,
    # the type of the done flag and the type of the info are as expected.
    def test_step_shape(self):
        self.env.reset()
        obs, reward, done, info = self.env.step(1)
        assert obs.shape == (16, 4, 4), f"Expected observation shape (16, 4, 4), got {obs.shape}"
        assert isinstance(reward, np.ndarray), f"Expected reward type np.ndarray, got {type(reward)}"
        assert isinstance(done, bool), f"Expected done type bool, got {type(done)}"
        assert isinstance(info, dict), f"Expected info type dict, got {type(info)}"

    # Test the render method of the Game2048 environment.
    # Ensure that the shape of the rendered image is as expected.
    def test_render(self):
        # (str) The render mode. Options are 'None', 'state_realtime_mode', 'image_realtime_mode' or 'image_savefile_mode'.  If None, then the game will not be rendered.
        self.env.reset()
        self.env.render(mode='state_realtime_mode')
        self.env.render(mode='image_savefile_mode')
        self.env.render(mode='image_realtime_mode')

    # Test the seed method of the Game2048 environment.
    # Ensure that the random seed is set correctly.
    def test_seed(self):
        self.env.seed(0)
        assert self.env.np_random.choice([0,1,2,3]) != np.random.choice([0,1,2,3])

    def test_step_action_case1(self):
        init_board = np.array([
            [8, 4, 0, 0],
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [2, 4, 2, 0]
        ])

        # Test action 0 (Assuming it represents 'up' move)
        self.env.reset(init_board=init_board, add_random_tile_flag=False)
        obs, reward, done, info = self.env.step(0)
        expected_board_up = np.array([
            [8, 8, 2, 0],
            [4, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        np.testing.assert_array_equal(self.env.board, expected_board_up, "Board state after 'up' action is incorrect")

        # Test action 1 (Assuming it represents 'right' move)
        self.env.reset(init_board=init_board, add_random_tile_flag=False)
        obs, reward, done, info = self.env.step(1)
        expected_board_right = np.array([
            [0, 0, 8, 4],
            [0, 0, 0, 2],
            [0, 0, 0, 2],
            [0, 2, 4, 2]
        ])
        np.testing.assert_array_equal(self.env.board, expected_board_right,
                                      "Board state after 'right' action is incorrect")

        # Test action 2 (Assuming it represents 'down' move)
        self.env.reset(init_board=init_board, add_random_tile_flag=False)
        obs, reward, done, info = self.env.step(2)
        expected_board_down = np.array([
            [0, 0, 0, 0],
            [8, 0, 0, 0],
            [2, 0, 0, 0],
            [4, 8, 2, 0]
        ])
        np.testing.assert_array_equal(self.env.board, expected_board_down,
                                      "Board state after 'down' action is incorrect")

        # Test action 3 (Assuming it represents 'left' move)
        self.env.reset(init_board=init_board, add_random_tile_flag=False)
        obs, reward, done, info = self.env.step(3)
        expected_board_left = np.array([
            [8, 4, 0, 0],
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [2, 4, 2, 0]
        ])
        np.testing.assert_array_equal(self.env.board, expected_board_left,
                                      "Board state after 'left' action is incorrect")

    def test_step_action_case2(self):
        init_board = np.array([
            [8, 4, 2, 0],
            [2, 0, 2, 0],
            [2, 2, 4, 0],
            [2, 4, 2, 0]
        ])

        # Test action 0 (Assuming it represents 'up' move)
        self.env.reset(init_board=init_board, add_random_tile_flag=False)
        obs, reward, done, info = self.env.step(0)
        expected_board_up = np.array([
            [8, 4, 4, 0],
            [4, 2, 4, 0],
            [2, 4, 2, 0],
            [0, 0, 0, 0]
        ])
        np.testing.assert_array_equal(self.env.board, expected_board_up, "Board state after 'up' action is incorrect")

        # Test action 1 (Assuming it represents 'right' move)
        self.env.reset(init_board=init_board, add_random_tile_flag=False)
        obs, reward, done, info = self.env.step(1)
        expected_board_right = np.array([
            [0, 8, 4, 2],
            [0, 0, 0, 4],
            [0, 0, 4, 4],
            [0, 2, 4, 2]
        ])
        np.testing.assert_array_equal(self.env.board, expected_board_right,
                                      "Board state after 'right' action is incorrect")

        # Test action 2 (Assuming it represents 'down' move)
        self.env.reset(init_board=init_board, add_random_tile_flag=False)
        obs, reward, done, info = self.env.step(2)
        expected_board_down = np.array([
            [0, 0, 0, 0],
            [8, 4, 4, 0],
            [2, 2, 4, 0],
            [4, 4, 2, 0]
        ])
        np.testing.assert_array_equal(self.env.board, expected_board_down,
                                      "Board state after 'down' action is incorrect")

        # Test action 3 (Assuming it represents 'left' move)
        self.env.reset(init_board=init_board, add_random_tile_flag=False)
        obs, reward, done, info = self.env.step(3)
        expected_board_left = np.array([
            [8, 4, 2, 0],
            [4, 0, 0, 0],
            [4, 4, 0, 0],
            [2, 4, 2, 0]
        ])
        np.testing.assert_array_equal(self.env.board, expected_board_left,
                                      "Board state after 'left' action is incorrect")