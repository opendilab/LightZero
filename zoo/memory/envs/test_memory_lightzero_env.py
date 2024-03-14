import unittest
import numpy as np
from ding.utils import set_pkg_seed
from easydict import EasyDict

from zoo.memory.envs.memory_lightzero_env import MemoryEnvLightZero


class TestMemoryEnvLightZero(unittest.TestCase):
    """
    Test case for MemoryEnvLightZero.
    """

    def setUp(self):
        """
        Set up the test environment.
        """
        self.config = dict(
            env_name='visual_match',  # The name of the environment, options: 'visual_match', 'key_to_door'
            # max_step=60,  # The maximum number of steps for each episode
            num_apples=10,  # Number of apples in the distractor phase
            # apple_reward=(1, 10),  # Range of rewards for collecting an apple
            # apple_reward=(1, 1),  # Range of rewards for collecting an apple
            apple_reward=(0, 0),  # Range of rewards for collecting an apple
            fix_apple_reward_in_episode=False,  # Whether to fix apple reward (DEFAULT_APPLE_REWARD) within an episode
            final_reward=10.0,  # Reward for choosing the correct door in the final phase
            respawn_every=300,  # Respawn interval for apples
            crop=True,  # Whether to crop the observation
            max_frames={
                "explore": 15,
                "distractor": 30,
                "reward": 15
            },  # Maximum frames per phase
            save_replay=False,  # Whether to save GIF replay
            render=False,  # Whether to enable real-time rendering
            scale_observation=True,
        )
        self.env = MemoryEnvLightZero(EasyDict(self.config))
        self.env.seed(123)
        self.cfg = self.env.default_config()

    def test_init(self):
        """
        Test the initialization of the environment.
        """
        self.assertIsInstance(self.env, MemoryEnvLightZero)
        self.assertIsInstance(self.env._cfg, dict)
        self.assertFalse(self.env._save_replay)

    def test_reset(self):
        """
        Test the reset method of the environment.
        """
        obs = self.env.reset()
        self.assertIsInstance(obs, dict)
        self.assertIn('observation', obs)
        self.assertIn('action_mask', obs)
        self.assertIn('to_play', obs)
        self.assertEqual(obs['observation'].shape, (1, 5, 5))
        self.assertEqual(obs['action_mask'].shape, (self.env.action_space.n,))
        self.assertEqual(obs['to_play'], -1)

    def test_step(self):
        """
        Test the step method of the environment.
        """
        self.env.reset()
        action = self.env.random_action()
        timestep = self.env.step(action)
        self.assertIsInstance(timestep.obs, dict)
        self.assertIn('observation', timestep.obs)
        self.assertIn('action_mask', timestep.obs)
        self.assertIn('to_play', timestep.obs)
        self.assertEqual(timestep.obs['observation'].shape, (1, 5, 5))
        self.assertEqual(timestep.obs['action_mask'].shape, (self.env.action_space.n,))
        self.assertEqual(timestep.obs['to_play'], -1)
        self.assertIsInstance(timestep.reward, np.ndarray)
        self.assertIsInstance(timestep.done, bool)
        self.assertIsInstance(timestep.info, dict)

    def test_seed(self):
        """
        Test the seed method of the environment.
        """
        self.env.seed(456)
        self.assertEqual(self.env._seed, 456)
        self.assertTrue(self.env._dynamic_seed)
        obs1 = self.env.reset()
        self.env.seed(456)
        self.env_2 = MemoryEnvLightZero(EasyDict(self.config))
        obs2 = self.env_2.reset()
        self.assertFalse(np.array_equal(obs1['observation'], obs2['observation']))

    def test_random_action(self):
        """
        Test the random_action method of the environment.
        """
        obs = self.env.reset()
        action = self.env.random_action()
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape, (1,))
        self.assertGreaterEqual(action[0], 0)
        self.assertLess(action[0], self.env.action_space.n)

    def test_create_collector_env_cfg(self):
        """
        Test the create_collector_env_cfg method of the environment.
        """
        cfg = EasyDict({'collector_env_num': 4, 'is_train': True})
        collector_env_cfg = MemoryEnvLightZero.create_collector_env_cfg(cfg)
        self.assertEqual(len(collector_env_cfg), 4)
        self.assertTrue(all(c['is_train'] for c in collector_env_cfg))

    def test_create_evaluator_env_cfg(self):
        """
        Test the create_evaluator_env_cfg method of the environment.
        """
        cfg = EasyDict({'evaluator_env_num': 2, 'is_train': False})
        evaluator_env_cfg = MemoryEnvLightZero.create_evaluator_env_cfg(cfg)
        self.assertEqual(len(evaluator_env_cfg), 2)
        self.assertTrue(all(not c['is_train'] for c in evaluator_env_cfg))

    def tearDown(self):
        """
        Clean up the test environment.
        """
        self.env.close()


if __name__ == 'main':
    set_pkg_seed(123)
    unittest.main()
