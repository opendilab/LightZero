import numpy as np
import pytest

pytest.importorskip('ding')

from zoo.board_games.chinese_chess.envs.chinese_chess_env import ChineseChessEnv, HumanQuitError


def test_reset_contract_and_seeded_random_bot() -> None:
    env = ChineseChessEnv(dict(battle_mode='play_with_bot_mode'))
    env.seed(0)
    obs = env.reset()
    assert obs['observation'].shape == (57, 10, 9)
    assert obs['observation'].dtype == np.float32
    assert obs['action_mask'].shape == (2086, )
    assert int(obs['action_mask'].sum()) == 44
    assert obs['to_play'] == -1
    assert int(obs['timestep']) == 0

    timestep = env.step(env.random_action())
    assert env._state.ply == 2 or timestep.done
    assert timestep.obs['to_play'] == -1
    assert int(timestep.obs['timestep']) == env._state.ply
    assert -1 <= float(timestep.reward) <= 1


def test_black_actions_and_observation_use_canonical_view() -> None:
    env = ChineseChessEnv(dict(battle_mode='self_play_mode'))
    red_obs = env.reset(start_player_index=0)
    black_obs = env.reset(start_player_index=1)
    # Black's physical moves are rotated into the same canonical labels as Red.
    assert set(np.flatnonzero(black_obs['action_mask'])) == set(np.flatnonzero(red_obs['action_mask']))
    # The initial position is color/rotation symmetric; only the side plane differs.
    np.testing.assert_array_equal(red_obs['observation'][:-1], black_obs['observation'][:-1])
    assert np.all(red_obs['observation'][-1] == 1)
    assert np.all(black_obs['observation'][-1] == 0)
    env.step(int(np.flatnonzero(black_obs['action_mask'])[0]))
    assert env.current_player == 1


def test_ctree_byte_reset_preserves_position_and_history() -> None:
    env = ChineseChessEnv(dict(battle_mode='self_play_mode', alphazero_mcts_ctree=True))
    env.seed(3)
    for _ in range(3):
        env.step(env.random_action())
    obs = env._make_obs()

    restored = ChineseChessEnv(dict(battle_mode='self_play_mode', alphazero_mcts_ctree=True))
    restored_obs = restored.reset(start_player_index=obs['current_player_index'], init_state=obs['board'].tobytes())
    np.testing.assert_array_equal(restored_obs['board'], obs['board'])
    np.testing.assert_array_equal(restored_obs['observation'], obs['observation'])
    np.testing.assert_array_equal(restored_obs['action_mask'], obs['action_mask'])


def test_simulate_action_does_not_mutate_source() -> None:
    env = ChineseChessEnv(dict(battle_mode='self_play_mode'))
    original = env._serialize().copy()
    child = env.simulate_action(env.legal_actions[0])
    np.testing.assert_array_equal(env._serialize(), original)
    assert child._state.ply == env._state.ply + 1


def test_terminal_human_move_accepts_iccs(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    env = ChineseChessEnv(dict(battle_mode='eval_mode', agent_vs_human=True))
    env.reset()
    expected_action = env.legal_actions[0]
    move = env.action_to_string(expected_action)
    monkeypatch.setattr('builtins.input', lambda _: move)

    assert env.human_to_action() == expected_action
    output = capsys.readouterr().out
    assert 'a b c d e f g h i' in output
    assert 'Turn: Red' in output


def test_terminal_human_can_quit(monkeypatch: pytest.MonkeyPatch) -> None:
    env = ChineseChessEnv(dict(battle_mode='eval_mode', agent_vs_human=True))
    monkeypatch.setattr('builtins.input', lambda _: 'q')
    with pytest.raises(HumanQuitError, match='human requested terminal game exit'):
        env.human_to_action()

    env.reset()
    timestep = env.step(env.legal_actions[0])
    assert timestep.done
    assert timestep.info['human_quit'] is True
    assert float(timestep.reward) == 0.0
