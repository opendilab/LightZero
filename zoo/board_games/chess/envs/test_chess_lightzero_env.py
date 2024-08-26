import unittest

import numpy as np
from easydict import EasyDict
from gymnasium import spaces

from zoo.board_games.chess.envs.chess_lightzero_env import ChessEnv


class TestChessEnv(unittest.TestCase):

    def setUp(self):
        self.cfg = EasyDict({
            'channel_last': False,
            'scale': False,
            'battle_mode': 'self_play_mode',
            'prob_random_agent': 0,
            'prob_expert_agent': 0,
            'agent_vs_human': False,
            'alphazero_mcts_ctree': False,
            'replay_path': None,
        })
        self.env = ChessEnv(self.cfg)

    def test_reset(self):
        obs = self.env.reset()
        self.assertIn('observation', obs)
        self.assertIn('action_mask', obs)
        self.assertIn('board', obs)
        self.assertIn('current_player_index', obs)
        self.assertIn('to_play', obs)

        self.assertEqual(obs['observation'].shape, (8, 8, 20))
        self.assertEqual(obs['action_mask'].shape, (4672,))
        self.assertEqual(len(obs['board']), 56)  # FEN string length
        self.assertIn(obs['current_player_index'], [0, 1])
        self.assertIn(obs['to_play'], [1, 2])

    def test_step(self):
        self.env.reset()
        action = self.env.random_action()
        timestep = self.env.step(action)

        self.assertIn('observation', timestep.obs)
        self.assertIn('action_mask', timestep.obs)
        self.assertIn('board', timestep.obs)
        self.assertIn('current_player_index', timestep.obs)
        self.assertIn('to_play', timestep.obs)

        self.assertIsInstance(timestep.reward, float)
        self.assertIsInstance(timestep.done, bool)
        self.assertIsInstance(timestep.info, dict)

    def test_observation_space(self):
        self.env.reset()
        obs_space = self.env.observation_space
        self.assertIn('observation', obs_space.spaces)
        self.assertIn('action_mask', obs_space.spaces)

        obs_shape = obs_space['observation'].shape
        action_mask_shape = obs_space['action_mask'].shape

        self.assertEqual(obs_shape, (8, 8, 20))
        self.assertEqual(action_mask_shape, (4672,))

    def test_action_space(self):
        self.env.reset()
        action_space = self.env.action_space
        self.assertIsInstance(action_space, spaces.Discrete)
        self.assertIsInstance(action_space.n, np.int64)
        self.assertEqual(action_space.n, 8 * 8 * 73)

    def test_simulate_action(self):
        self.env.reset()
        action = self.env.random_action()
        next_env = self.env.simulate_action(action)

        self.assertIsInstance(next_env, ChessEnv)
        self.assertNotEqual(next_env.board.fen(), self.env.board.fen())

    def test_create_collector_env_cfg(self):
        collector_env_num = 10
        self.cfg.collector_env_num = collector_env_num
        collector_env_cfg = ChessEnv.create_collector_env_cfg(self.cfg)

        self.assertIsInstance(collector_env_cfg, list)
        self.assertEqual(len(collector_env_cfg), collector_env_num)
        self.assertEqual(collector_env_cfg[0], self.cfg)

    def test_create_evaluator_env_cfg(self):
        evaluator_env_num = 5
        self.cfg.evaluator_env_num = evaluator_env_num
        evaluator_env_cfg = ChessEnv.create_evaluator_env_cfg(self.cfg)

        self.assertIsInstance(evaluator_env_cfg, list)
        self.assertEqual(len(evaluator_env_cfg), evaluator_env_num)
        self.assertEqual(evaluator_env_cfg[0].battle_mode, 'eval_mode')

if __name__ == '__main__':
    unittest.main()