import pytest
import numpy as np
from easydict import EasyDict
from gymnasium import spaces

from zoo.board_games.chess.envs.chess_lightzero_env import ChessLightZeroEnv


@pytest.fixture
def env():
    cfg = EasyDict({
        'channel_last': False,
        'scale': False,
        'battle_mode': 'self_play_mode',
        'prob_random_agent': 0,
        'prob_expert_agent': 0,
        'agent_vs_human': False,
        'alphazero_mcts_ctree': False,
        'replay_path': None,
    })
    return ChessLightZeroEnv(cfg)


@pytest.mark.envtest
def test_reset(env):
    obs = env.reset()
    assert 'observation' in obs
    assert 'action_mask' in obs
    assert 'board' in obs
    assert 'current_player_index' in obs
    assert 'to_play' in obs

    assert obs['observation'].shape == (8, 8, 20)
    # assert obs['observation'].shape == (8, 8, 111)

    assert obs['action_mask'].shape == (4672,)
    assert len(obs['board']) == 56  # FEN string length
    assert obs['current_player_index'] in [0, 1]
    assert obs['to_play'] in [1, 2]


@pytest.mark.envtest
def test_step(env):
    env.reset()
    action = env.random_action()
    timestep = env.step(action)

    assert 'observation' in timestep.obs
    assert 'action_mask' in timestep.obs
    assert 'board' in timestep.obs
    assert 'current_player_index' in timestep.obs
    assert 'to_play' in timestep.obs

    assert isinstance(timestep.reward, float)
    assert isinstance(timestep.done, bool)
    assert isinstance(timestep.info, dict)


@pytest.mark.envtest
def test_observation_space(env):
    env.reset()
    obs_space = env.observation_space
    assert 'observation' in obs_space.spaces
    assert 'action_mask' in obs_space.spaces

    obs_shape = obs_space['observation'].shape
    action_mask_shape = obs_space['action_mask'].shape

    assert obs_shape == (8, 8, 20)
    assert action_mask_shape == (4672,)


@pytest.mark.envtest
def test_action_space(env):
    env.reset()
    action_space = env.action_space
    assert isinstance(action_space, spaces.Discrete)
    assert isinstance(action_space.n, np.int64)
    assert action_space.n == 8 * 8 * 73


@pytest.mark.envtest
def test_simulate_action(env):
    env.reset()
    action = env.random_action()
    next_env = env.simulate_action(action)

    assert isinstance(next_env, ChessLightZeroEnv)
    assert next_env.board.fen() != env.board.fen()


@pytest.mark.envtest
def test_create_collector_env_cfg(env):
    collector_env_num = 10
    env.cfg.collector_env_num = collector_env_num
    collector_env_cfg = ChessLightZeroEnv.create_collector_env_cfg(env.cfg)

    assert isinstance(collector_env_cfg, list)
    assert len(collector_env_cfg) == collector_env_num
    assert collector_env_cfg[0] == env.cfg


@pytest.mark.envtest
def test_create_evaluator_env_cfg(env):
    evaluator_env_num = 5
    env.cfg.evaluator_env_num = evaluator_env_num
    evaluator_env_cfg = ChessLightZeroEnv.create_evaluator_env_cfg(env.cfg)

    assert isinstance(evaluator_env_cfg, list)
    assert len(evaluator_env_cfg) == evaluator_env_num
    assert evaluator_env_cfg[0].battle_mode == 'eval_mode'