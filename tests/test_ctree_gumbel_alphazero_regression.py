import math
import sys
from pathlib import Path

import pytest


BUILD_DIR = Path(__file__).resolve().parents[1] / 'lzero' / 'mcts' / 'ctree' / 'ctree_gumbel_alphazero' / 'build'
sys.path.append(str(BUILD_DIR))
mcts_gumbel_alphazero = pytest.importorskip('mcts_gumbel_alphazero')


class _ActionSpace:
    n = 49


class SparseActionEnv:
    """Small deterministic tree whose legal action ids exceed each node's child count."""

    action_space = _ActionSpace()
    battle_mode_in_simulation_env = "self_play_mode"

    def __init__(self):
        self.reset(0, None, False, None)

    def reset(self, start_player_index, init_state, katago_policy_init, katago_game_state):
        del init_state, katago_policy_init, katago_game_state
        self.path = []
        self.current_player = start_player_index + 1
        self.battle_mode = self.battle_mode_in_simulation_env

    @property
    def legal_actions(self):
        choices = ([10, 20, 30], [11, 21], [12, 22], [13, 23])
        return list(choices[min(len(self.path), len(choices) - 1)])

    def step(self, action):
        assert action in self.legal_actions
        self.path.append(action)
        self.current_player = 3 - self.current_player

    def get_done_winner(self):
        return len(self.path) >= 4, -1


class WideRootEnv(SparseActionEnv):

    @property
    def legal_actions(self):
        if not self.path:
            return list(range(40))
        return super().legal_actions


def _policy_value(env):
    weights = {action: index + 1.0 for index, action in enumerate(env.legal_actions)}
    total = sum(weights.values())
    return {action: weight / total for action, weight in weights.items()}, 0.125


def _make_mcts(env, seed=7.0):
    return mcts_gumbel_alphazero.MCTS(
        num_simulations=32,
        maxvisit_init=50,
        value_scale=0.1,
        gumbel_scale=1.0,
        gumbel_rng=seed,
        max_num_considered_actions=4,
        simulate_env=env,
    )


def _state_config():
    return {
        "start_player_index": 0,
        "init_state": None,
        "katago_policy_init": False,
        "katago_game_state": None,
    }


def test_sparse_action_ids_do_not_index_children_out_of_bounds():
    env = SparseActionEnv()
    action, visit_policy, improved_policy = _make_mcts(env).get_next_action(_state_config(), _policy_value, 1.0, False)

    assert action in [10, 20, 30]
    assert action == max(range(len(improved_policy)), key=improved_policy.__getitem__)
    assert action != max(range(len(visit_policy)), key=visit_policy.__getitem__)
    assert len(visit_policy) == env.action_space.n
    assert len(improved_policy) == env.action_space.n
    assert math.isclose(sum(visit_policy), 1.0)
    assert math.isclose(sum(improved_policy), 1.0)
    assert all(math.isfinite(probability) for probability in improved_policy)
    assert all(improved_policy[action_id] == 0.0 for action_id in range(49) if action_id not in [10, 20, 30])


def test_gumbel_stream_is_reproducible_but_not_frozen_or_hard_coded_to_36():
    first = _make_mcts(SparseActionEnv(), seed=123.0)
    second = _make_mcts(SparseActionEnv(), seed=123.0)

    first_draw = first._generate_gumbel(1.0, 123.0, 49)
    second_draw = first._generate_gumbel(1.0, 123.0, 49)
    matching_draw = second._generate_gumbel(1.0, 123.0, 49)

    assert len(first_draw) == 49
    assert first_draw == matching_draw
    assert first_draw != second_draw


def test_search_supports_more_than_36_root_actions():
    env = WideRootEnv()
    action, _, improved_policy = _make_mcts(env).get_next_action(_state_config(), _policy_value, 1.0, False)

    assert action in range(40)
    assert math.isclose(sum(improved_policy), 1.0)
    assert all(improved_policy[action_id] == 0.0 for action_id in range(40, 49))


def test_sample_flag_no_longer_adds_dirichlet_noise_to_gumbel_search():
    sampled = _make_mcts(SparseActionEnv(), seed=99.0).get_next_action(_state_config(), _policy_value, 1.0, True)
    evaluated = _make_mcts(SparseActionEnv(), seed=99.0).get_next_action(_state_config(), _policy_value, 1.0, False)

    assert sampled[1] == evaluated[1]
    assert sampled[2] == evaluated[2]
    assert sampled[0] in [10, 20, 30]
