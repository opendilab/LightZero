import pytest
from easydict import EasyDict

from lzero.mcts.ptree.ptree_az import MCTS as AlphaZeroMCTS
from lzero.mcts.ptree.ptree_az_sampled import MCTS as SampledAlphaZeroMCTS


class DummyEnv:

    def __init__(self, name):
        self.name = name


def _make_cfg():
    return EasyDict(
        dict(
            max_moves=10,
            num_simulations=1,
            pb_c_base=1,
            pb_c_init=1,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.25,
            legal_actions=[0, 1],
            action_space_size=2,
            num_of_sampled_actions=2,
            continuous_action_space=False,
        )
    )


def _check_batch_wrapper(mcts_cls, monkeypatch):
    env0 = DummyEnv('env0')
    env1 = DummyEnv('env1')
    mcts = mcts_cls(_make_cfg(), env0)
    state_configs = [EasyDict(start_player_index=1, init_state='s0'), EasyDict(start_player_index=2, init_state='s1')]
    calls = []

    def fake_get_next_action(self, temperature=1.0, sample=True, **kwargs):
        state_config = kwargs.get('state_config_for_simulate_env_reset', kwargs.get('state_config_for_env_reset'))
        policy_fn = kwargs.get('policy_forward_fn', kwargs.get('policy_value_func'))
        assert self.simulate_env in [env0, env1]
        policy_result = policy_fn(self.simulate_env)
        calls.append((state_config.init_state, self.simulate_env.name, policy_result, temperature, sample))
        return len(calls), [0.0, 1.0]

    monkeypatch.setattr(mcts_cls, 'get_next_action', fake_get_next_action)

    def fake_policy_batch(env_list):
        assert len(env_list) == 1
        return [({'env': env_list[0].name}, 0.0)]

    results = mcts.get_next_actions_batch(state_configs, fake_policy_batch, temperature=0.5, sample=False, env_list=[env0, env1])

    assert results == [(1, [0.0, 1.0]), (2, [0.0, 1.0])]
    assert calls == [
        ('s0', 'env0', ({'env': 'env0'}, 0.0), 0.5, False),
        ('s1', 'env1', ({'env': 'env1'}, 0.0), 0.5, False),
    ]
    assert mcts.simulate_env is env0


@pytest.mark.unittest
def test_alphazero_ptree_get_next_actions_batch(monkeypatch):
    _check_batch_wrapper(AlphaZeroMCTS, monkeypatch)


@pytest.mark.unittest
def test_sampled_alphazero_ptree_get_next_actions_batch(monkeypatch):
    _check_batch_wrapper(SampledAlphaZeroMCTS, monkeypatch)
