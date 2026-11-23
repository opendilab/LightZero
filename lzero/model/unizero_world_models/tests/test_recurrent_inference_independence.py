"""Auxiliary recurrent supervision must not select the online cache algorithm."""
import pytest
import torch
from easydict import EasyDict

from lzero.model.unizero_world_models.world_model import WorldModel
from .test_per_sample_is_weights import _build_world_model


@pytest.fixture
def model_and_batch():
    torch.manual_seed(723)
    template = _build_world_model()
    config = EasyDict(dict(template.config, num_layers=2, context_length=10, max_tokens=20, max_blocks=10))
    wm = WorldModel(config, template.tokenizer)
    wm.open_loop_consistency_batch_size = 2
    wm.open_loop_consistency_horizon = 4
    wm.open_loop_prefix_transitions = 3
    for head in (wm.head_observations, wm.head_value, wm.head_policy, wm.head_rewards):
        final_linear = [module for module in head.modules() if isinstance(module, torch.nn.Linear)][-1]
        torch.nn.init.normal_(final_linear.weight, std=0.04)
    obs = torch.randn(2, 10, wm.embed_dim, requires_grad=True)
    target = torch.randn_like(obs, requires_grad=True)
    actions = torch.randint(0, wm.config.action_space_size, (2, 10))
    mask = torch.ones(2, 10, dtype=torch.bool)

    def labels(num_classes):
        return torch.softmax(torch.randn(actions.numel(), num_classes), dim=-1).requires_grad_()

    batch = dict(
        obs_embeddings=obs,
        target_obs_embeddings=target,
        actions=actions,
        mask_padding=mask,
        labels_rewards=labels(wm.config.support_size),
        labels_policy=labels(wm.config.action_space_size),
        labels_value=labels(wm.config.support_size),
    )
    return wm, batch


def test_recurrent_loss_and_gradients_independent_of_online_cache_flag(model_and_batch):
    wm, batch = model_and_batch
    cached = object()
    wm.past_kv_cache_recurrent_infer['sentinel'] = cached
    records = []
    for exact in (True, False):
        wm.rebuild_kv_window_from_tokens = exact
        loss, parts = wm.compute_open_loop_recurrent_loss(**batch)
        # Future decision heads must train predicted observations, even without
        # adding the latent MSE component to this differentiation.
        decision_loss = parts['policy'] + 0.5 * parts['value']
        head_parameters = list(wm.head_observations.parameters())
        grads = torch.autograd.grad(
            decision_loss, head_parameters + [batch['obs_embeddings']], retain_graph=True, allow_unused=True
        )
        assert sum(g.abs().sum().item() for g in grads[:-1] if g is not None) > 0
        assert grads[-1] is not None and torch.isfinite(grads[-1]).all()
        targets = [batch[k] for k in ('target_obs_embeddings', 'labels_rewards', 'labels_policy', 'labels_value')]
        assert all(g is None for g in torch.autograd.grad(loss, targets, allow_unused=True))
        records.append((loss.detach(), [g.detach() if g is not None else None for g in grads]))
        assert wm.training
        assert wm.rebuild_kv_window_from_tokens is exact
        assert wm.past_kv_cache_recurrent_infer['sentinel'] is cached
    torch.testing.assert_close(records[0][0], records[1][0])
    for exact_grad, approximate_grad in zip(records[0][1], records[1][1]):
        if exact_grad is None:
            assert approximate_grad is None
        else:
            torch.testing.assert_close(exact_grad, approximate_grad)


def test_recurrent_mask_and_support_guards(model_and_batch):
    wm, batch = model_and_batch
    wm.rebuild_kv_window_from_tokens = False
    batch['mask_padding'].zero_()
    loss, _ = wm.compute_open_loop_recurrent_loss(**batch)
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()
    assert batch['obs_embeddings'].grad.abs().max().item() == 0
    with pytest.raises(RuntimeError, match='rebuild_kv'):
        wm._validate_open_loop_support('diagnostic')
    wm.config.rotary_emb = True
    with pytest.raises(RuntimeError, match='rotary'):
        wm.compute_open_loop_recurrent_loss(**batch)


@pytest.mark.parametrize('exact', [False, True])
def test_decision_only_removes_only_auxiliary_latent_term(model_and_batch, exact):
    wm, batch = model_and_batch
    wm.rebuild_kv_window_from_tokens = exact
    default_total, full = wm.compute_open_loop_recurrent_loss(**batch)
    torch.testing.assert_close(
        default_total, 10 * full['latent'] + full['reward'] + 0.5 * full['value'] + full['policy']
    )
    wm.config.open_loop_recurrent_include_latent_loss = False
    decision_total, parts = wm.compute_open_loop_recurrent_loss(**batch)
    torch.testing.assert_close(decision_total, parts['reward'] + 0.5 * parts['value'] + parts['policy'])
    for name in full:
        torch.testing.assert_close(full[name], parts[name])
    torch.testing.assert_close(default_total - decision_total, 10 * parts['latent'])
    gradients = torch.autograd.grad(decision_total, list(wm.head_observations.parameters()), allow_unused=True)
    assert sum(g.abs().sum().item() for g in gradients if g is not None) > 0
    assert wm.rebuild_kv_window_from_tokens is exact
