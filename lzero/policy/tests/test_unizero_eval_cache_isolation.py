"""Tests that asynchronous UniZero eval resets preserve other env histories."""

from types import SimpleNamespace

import pytest

from lzero.policy.unizero import UniZeroPolicy


class _FakeCacheManager:

    def __init__(self, root_pools):
        self.init_pools = root_pools
        self.recur_clear_count = 0

    def clear_recur_cache(self):
        self.recur_clear_count += 1


class _FakeWorldModel:

    def __init__(self, *, exact_reset=False, raw_rebuild=False):
        self.exact_kv_window_reset = exact_reset
        self.rebuild_kv_window_from_tokens = raw_rebuild
        self.precompute_count = 0
        self.clear_count = 0

    def precompute_pos_emb_diff_kv(self):
        self.precompute_count += 1

    def clear_caches(self):
        self.clear_count += 1


@pytest.mark.unittest
@pytest.mark.parametrize('use_new_cache_manager', [False, True])
def test_finished_eval_env_does_not_clear_other_root_caches(use_new_cache_manager):
    policy = object.__new__(UniZeroPolicy)
    policy._cfg = SimpleNamespace(
        model=SimpleNamespace(observation_shape=(3, 4, 4), analysis_sim_norm=False),
        evaluator_env_num=3,
        device='cpu',
        empty_cuda_cache_on_cache_reset=False,
        kv_cache_clear_interval=2000,
    )
    policy.pad_token_id = 0
    policy.last_batch_obs_eval = None
    policy.last_batch_action_eval = [-1, -1, -1]

    root_pools = [
        {'env0-history': object()},
        {'env1-history': object()},
        {'env2-history': object()},
    ]
    world_model = SimpleNamespace(
        use_new_cache_manager=use_new_cache_manager, env_num=3, keys_values_wm_list=[object()]
    )
    if use_new_cache_manager:
        world_model.kv_cache_manager = _FakeCacheManager(root_pools)
    else:
        world_model.past_kv_cache_init_infer_envs = root_pools
        world_model.past_kv_cache_recurrent_infer = {'mcts-scratch': object()}
    policy._eval_model = SimpleNamespace(world_model=world_model)

    policy._reset_eval(env_id=[1], current_steps=None, reset_init_data=False)

    assert list(root_pools[0]) == ['env0-history']
    assert root_pools[1] == {}
    assert list(root_pools[2]) == ['env2-history']
    if use_new_cache_manager:
        assert world_model.kv_cache_manager.recur_clear_count == 1
    else:
        assert world_model.past_kv_cache_recurrent_infer == {}
    assert world_model.keys_values_wm_list == []


@pytest.mark.unittest
@pytest.mark.parametrize(
    'exact_reset,raw_rebuild,expected_precompute',
    [(False, False, 1), (True, False, 0), (False, True, 0)],
)
def test_learning_cache_reset_only_precomputes_legacy_position_differences(
    exact_reset, raw_rebuild, expected_precompute
):
    policy = object.__new__(UniZeroPolicy)
    policy._cfg = SimpleNamespace(
        model=SimpleNamespace(
            world_model_cfg=SimpleNamespace(rotary_emb=False), analysis_sim_norm=False
        ),
        empty_cuda_cache_on_cache_reset=False,
    )
    world_model = _FakeWorldModel(exact_reset=exact_reset, raw_rebuild=raw_rebuild)
    shared_model = SimpleNamespace(world_model=world_model)
    policy._learn_model = shared_model
    policy._collect_model = shared_model
    policy._eval_model = shared_model
    policy._target_model = shared_model

    policy.recompute_pos_emb_diff_and_clear_cache()

    assert world_model.precompute_count == expected_precompute
    assert world_model.clear_count == 1


@pytest.mark.unittest
def test_open_loop_diagnostics_are_registered_for_logging():
    policy = object.__new__(UniZeroPolicy)
    policy.use_head_clip = False
    policy._cfg = SimpleNamespace(model=SimpleNamespace(analysis_sim_norm=False))

    monitored = set(policy._monitor_vars_learn())

    assert {
        'analysis/open_loop_latent_mse_mean',
        'analysis/rolling_teacher_latent_mse_mean',
        'analysis/teacher_forced_latent_mse_mean',
        'analysis/rolling_context_ratio',
        'analysis/open_loop_exposure_ratio',
        'analysis/open_loop_total_ratio',
    } <= monitored
