from types import SimpleNamespace

import pytest
import torch

from lzero.policy.unizero import UniZeroPolicy


def _minimal_policy(alpha: float, legacy_resume_alpha=None) -> UniZeroPolicy:
    policy = object.__new__(UniZeroPolicy)
    policy._cfg = SimpleNamespace(
        model=SimpleNamespace(analysis_sim_norm=False),
        legacy_resume_adaptive_alpha=legacy_resume_alpha,
    )
    policy._learn_model = torch.nn.Linear(2, 2)
    policy._target_model = torch.nn.Linear(2, 2)
    policy._optimizer_world_model = torch.optim.Adam(policy._learn_model.parameters(), lr=1e-3)
    policy.use_adaptive_entropy_weight = True
    policy.log_alpha = torch.nn.Parameter(torch.tensor([alpha]).log())
    policy.alpha_optimizer = torch.optim.Adam([policy.log_alpha], lr=1e-3)
    return policy


def _populate_optimizer_state(policy: UniZeroPolicy) -> None:
    policy._learn_model(torch.ones(1, 2)).sum().backward()
    policy._optimizer_world_model.step()
    policy.alpha_optimizer.zero_grad()
    policy.log_alpha.sum().backward()
    policy.alpha_optimizer.step()


def test_unizero_checkpoint_round_trip_restores_adaptive_entropy_and_optimizers():
    source = _minimal_policy(alpha=0.23)
    _populate_optimizer_state(source)
    checkpoint = source._state_dict_learn()

    restored = _minimal_policy(alpha=1.0)
    restored._load_state_dict_learn(checkpoint)

    assert restored.log_alpha.exp().item() == pytest.approx(source.log_alpha.exp().item())
    assert restored._optimizer_world_model.state_dict()['state']
    assert restored.alpha_optimizer.state_dict()['state']


def test_legacy_checkpoint_can_restore_alpha_from_explicit_log_value():
    source = _minimal_policy(alpha=0.23)
    checkpoint = source._state_dict_learn()
    checkpoint.pop('log_alpha')

    restored = _minimal_policy(alpha=1.0, legacy_resume_alpha=0.23)
    restored._load_state_dict_learn(checkpoint)

    assert restored.log_alpha.exp().item() == pytest.approx(0.23)
