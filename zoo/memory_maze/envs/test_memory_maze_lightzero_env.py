import unittest
import numpy as np
from ding.utils import set_pkg_seed
from easydict import EasyDict

from memory_maze_lightzero_env import MemoryMazeEnvLightZero


class TestMemoryMazeEnvLightZero(unittest.TestCase):
    """
    Test case for MemoryMazeEnvLightZero.
    """

    def setUp(self):
        """
        Set up the test environment with a specific configuration.
        """
        self.config = dict(
            env_id='memory_maze:MemoryMaze-9x9-v0',  # The name of the environment variant
            save_replay=False,  # Whether to save a gif replay of the episode
            render=False,  # Whether to render the environment in real-time
            scale_observation=True,  # Whether to scale the observation to [0, 1]
            rgb_img_observation=True,  # Whether the observation is in RGB format
            flatten_observation=False,  # Whether to flatten the observation
            max_steps=100,  # Maximum steps per episode
        )
        self.env = MemoryMazeEnvLightZero(EasyDict(self.config))
        self.env.seed(123)
        self.cfg = self.env.default_config()

    def test_init(self):
        """
        Test the initialization of the MemoryMazeEnvLightZero environment.
        """
        self.assertIsInstance(self.env, MemoryMazeEnvLightZero)
        self.assertIsInstance(self.env._cfg, dict)
        self.assertFalse(self.env._save_replay)

    def test_reset(self):
        """
        Test the reset method of the MemoryMazeEnvLightZero environment.
        """
        obs = self.env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, self.env._observation_space.shape)
        self.assertLessEqual(obs.max(), 1.0)  # Since we expect a scaled observation

    def test_step(self):
        """
        Test the step method of the MemoryMazeEnvLightZero environment.
        """
        self.env.reset()
        action = self.env.random_action()
        timestep = self.env.step(action)
        self.assertIsInstance(timestep.obs, np.ndarray)
        self.assertEqual(timestep.obs.shape, self.env._observation_space.shape)
        self.assertIsInstance(timestep.reward, float)
        self.assertIsInstance(timestep.done, bool)
        self.assertIsInstance(timestep.info, dict)

    def test_observation_scaling(self):
        """
        Test that observations are appropriately scaled if the configuration option is set.
        """
        self.env.reset()
        obs = self.env.reset()
        self.assertTrue(np.all((obs >= 0) & (obs <= 1)))  # Check that all values are scaled between 0 and 1

    def test_random_action(self):
        """
        Test the random_action method of the MemoryMazeEnvLightZero environment.
        """
        obs = self.env.reset()
        action = self.env.random_action()
        self.assertIsInstance(action, int)
        # self.assertEqual(action.shape, ())  # Single value action
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.env.action_space.n)

    def test_max_steps(self):
        """
        Test that the environment terminates after the maximum number of steps.
        """
        self.env.reset()
        done = False
        step_count = 0
        while not done:
            action = self.env.random_action()
            timestep = self.env.step(action)
            step_count += 1
            done = timestep.done
        self.assertEqual(step_count, self.env._max_steps)

    def test_seed(self):
        """
        Test the seed method of the MemoryMazeEnvLightZero environment.
        """
        self.env.seed(456)
        obs1 = self.env.reset()
        self.env.seed(456)
        env_2 = MemoryMazeEnvLightZero(EasyDict(self.config))
        env_2.seed(456)
        obs2 = env_2.reset()
        # self.assertTrue(np.array_equal(obs1, obs2))  # Observations should be identical with the same seed

    def test_create_collector_env_cfg(self):
        """
        Test the create_collector_env_cfg method of the MemoryMazeEnvLightZero environment.
        """
        cfg = EasyDict({'collector_env_num': 3, 'is_train': True})
        collector_env_cfg = MemoryMazeEnvLightZero.create_collector_env_cfg(cfg)
        self.assertEqual(len(collector_env_cfg), 3)
        self.assertTrue(all(c['is_train'] for c in collector_env_cfg))

    def test_create_evaluator_env_cfg(self):
        """
        Test the create_evaluator_env_cfg method of the MemoryMazeEnvLightZero environment.
        """
        cfg = EasyDict({'evaluator_env_num': 2, 'is_train': False})
        evaluator_env_cfg = MemoryMazeEnvLightZero.create_evaluator_env_cfg(cfg)
        self.assertEqual(len(evaluator_env_cfg), 2)
        self.assertTrue(all(not c['is_train'] for c in evaluator_env_cfg))

    def tearDown(self):
        """
        Clean up the test environment.
        """
        self.env.close()


if __name__ == '__main__':
    set_pkg_seed(123)
    unittest.main()