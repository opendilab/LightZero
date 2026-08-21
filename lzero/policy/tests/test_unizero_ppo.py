from collections import deque
from types import SimpleNamespace

import numpy as np
import torch
from easydict import EasyDict

from lzero.mcts.buffer.game_buffer_unizero_ppo import UniZeroPPOGameBuffer
from lzero.policy.ppo_utils import compute_gae, normalize_advantages, ppo_policy_loss
from lzero.worker.muzero_collector import MuZeroCollector


def test_compute_gae_builds_returns_before_advantage_normalization():
    advantages, returns = compute_gae(
        rewards=[1.0, 1.0], values=[0.5, 0.25], gamma=1.0, gae_lambda=1.0
    )
    np.testing.assert_allclose(advantages, [1.5, 0.75])
    np.testing.assert_allclose(returns, [2.0, 1.0])


def test_advantages_are_normalized_across_the_whole_rollout():
    normalized, mean, std = normalize_advantages([
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.asarray([3.0], dtype=np.float32),
    ])
    flattened = np.concatenate(normalized)
    assert mean == 2.0
    assert std > 0.0
    np.testing.assert_allclose(flattened.mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(flattened.std(), 1.0, atol=1e-6)


def test_ppo_ratio_starts_at_one_for_behavior_logits_and_respects_action_mask():
    logits = torch.tensor([[[1.0, 0.0, 100.0], [0.2, 0.8, -50.0]]])
    action_mask = torch.tensor([[[True, True, False], [True, True, False]]])
    actions = torch.tensor([[0, 1]])
    masked_logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
    old_log_prob = torch.distributions.Categorical(logits=masked_logits).log_prob(actions)

    policy_loss, entropy, metrics = ppo_policy_loss(
        logits, action_mask, actions, old_log_prob,
        advantages=torch.tensor([[1.0, -1.0]]),
        valid_mask=torch.ones((1, 2), dtype=torch.bool),
        clip_ratio=0.2,
    )

    torch.testing.assert_close(policy_loss, torch.tensor([[-1.0, 1.0]]))
    assert torch.isfinite(entropy).all()
    torch.testing.assert_close(metrics['ppo_approx_kl'], torch.tensor(0.0))
    torch.testing.assert_close(metrics['ppo_clip_fraction'], torch.tensor(0.0))
    torch.testing.assert_close(metrics['ppo_ratio_mean'], torch.tensor(1.0))


def _segment(episode_id, reward, value, log_prob):
    return SimpleNamespace(
        episode_id=episode_id,
        valid_transition_count=1,
        reward_segment=np.asarray([reward], dtype=np.float32),
        root_value_segment=np.asarray([value], dtype=np.float32),
        behavior_log_prob_segment=np.asarray([log_prob], dtype=np.float32),
        behavior_action_mask_segment=np.asarray([[True, True]], dtype=np.bool_),
        behavior_policy_feature_segment=np.asarray([[reward, value]], dtype=np.float32),
        advantage_segment=np.asarray([], dtype=np.float32),
        return_segment=np.asarray([], dtype=np.float32),
    )


def test_collector_gae_keeps_interleaved_episode_segments_aligned():
    collector = MuZeroCollector.__new__(MuZeroCollector)
    collector.policy_improvement = 'ppo'
    collector.ppo_gamma = 1.0
    collector.ppo_gae_lambda = 1.0
    collector.ppo_normalize_advantage = True
    collector.policy_config = EasyDict(num_unroll_steps=2)
    collector._logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    collector._end_flag = True

    episode_zero_first = _segment(0, reward=1.0, value=0.0, log_prob=-0.1)
    episode_one = _segment(1, reward=10.0, value=0.0, log_prob=-0.2)
    episode_zero_last = _segment(0, reward=1.0, value=0.0, log_prob=-0.3)
    collector.game_segment_pool = deque([
        (episode_zero_first, None, False),
        (episode_one, None, True),
        (episode_zero_last, None, True),
    ])

    collector._finalize_ppo_rollout()

    np.testing.assert_allclose(episode_zero_first.return_segment, [2.0, 1.0])
    np.testing.assert_allclose(episode_zero_first.behavior_log_prob_segment, [-0.1, -0.3])
    np.testing.assert_allclose(episode_zero_first.behavior_policy_feature_segment, [[1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(episode_zero_last.return_segment, [1.0])
    np.testing.assert_allclose(episode_one.return_segment, [10.0])


def test_on_policy_indices_are_fresh_and_non_overlapping_and_release_is_selective():
    buffer = UniZeroPPOGameBuffer.__new__(UniZeroPPOGameBuffer)
    buffer._cfg = EasyDict(num_unroll_steps=3)
    buffer.base_idx = 0
    fresh = SimpleNamespace(
        collection_train_iter=7,
        behavior_log_prob_segment=np.ones(6, dtype=np.float32),
        behavior_action_mask_segment=np.ones((6, 2), dtype=np.bool_),
        behavior_policy_feature_segment=np.ones((6, 4), dtype=np.float32),
        advantage_segment=np.ones(6, dtype=np.float32),
        return_segment=np.ones(6, dtype=np.float32),
    )
    stale = SimpleNamespace(
        collection_train_iter=6,
        behavior_log_prob_segment=np.ones(3, dtype=np.float32),
        behavior_action_mask_segment=np.ones((3, 2), dtype=np.bool_),
        behavior_policy_feature_segment=np.ones((3, 4), dtype=np.float32),
        advantage_segment=np.ones(3, dtype=np.float32),
        return_segment=np.ones(3, dtype=np.float32),
    )
    buffer.game_segment_buffer = deque([fresh, stale])
    buffer.game_segment_game_pos_look_up = [
        *((0, position) for position in range(6)),
        *((1, position) for position in range(3)),
    ]

    np.testing.assert_array_equal(buffer.get_on_policy_indices(7), [0, 3])
    buffer.release_on_policy_data(7)
    assert fresh.behavior_policy_feature_segment.size == 0
    assert stale.behavior_policy_feature_segment.size > 0


def test_ppo_batch_skips_unused_observation_and_target_context_materialization():
    buffer = UniZeroPPOGameBuffer.__new__(UniZeroPPOGameBuffer)
    buffer._cfg = EasyDict(
        num_unroll_steps=2,
        td_steps=1,
        game_segment_length=2,
        model=dict(model_type='conv'),
    )
    buffer.action_space_size = 2

    def unexpected_observation_read(*args, **kwargs):
        raise AssertionError('PPO head-only batches must not materialize observations')

    segment = SimpleNamespace(
        action_segment=np.asarray([0, 1, 0], dtype=np.int64),
        timestep_segment=np.asarray([0, 1, 2], dtype=np.int64),
        action_space_size=2,
        advantage_segment=np.asarray([0.5, -0.5], dtype=np.float32),
        behavior_log_prob_segment=np.asarray([-0.2, -0.3], dtype=np.float32),
        return_segment=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        behavior_action_mask_segment=np.ones((2, 2), dtype=np.bool_),
        behavior_policy_feature_segment=np.ones((2, 4), dtype=np.float32),
        collection_train_iter=3,
        get_unroll_obs=unexpected_observation_read,
    )
    orig_data = (
        [segment], [0], np.asarray([4]), np.ones(1, dtype=np.float32),
        np.zeros(1, dtype=np.float64),
    )

    reward_context, policy_re_context, policy_non_re_context, current_batch = buffer._make_batch(
        1,
        reanalyze_ratio=0.0,
        orig_data=orig_data,
        include_ppo=True,
        prepare_observations=False,
        prepare_target_context=False,
    )

    assert reward_context is policy_re_context is policy_non_re_context is None
    assert current_batch[0].shape == (1, 0)
    np.testing.assert_array_equal(current_batch[1], [[0, 1]])
    np.testing.assert_array_equal(current_batch[8], [[0.5, -0.5]])


def test_world_model_batch_reuses_sample_and_preserves_reward_sequence():
    buffer = UniZeroPPOGameBuffer.__new__(UniZeroPPOGameBuffer)
    buffer._cfg = EasyDict(num_unroll_steps=2)
    buffer.action_space_size = 3
    buffer.sample_type = 'transition'
    segment = SimpleNamespace(
        reward_segment=np.asarray([1.0, -1.0], dtype=np.float32),
    )
    orig_data = (
        [segment], [1], np.asarray([9]), np.ones(1, dtype=np.float32),
        np.zeros(1, dtype=np.float64),
    )
    current_batch = [np.asarray([[123.0]], dtype=np.float32)]
    calls = []

    buffer._sample_orig_data = lambda batch_size: orig_data

    def make_batch(batch_size, reanalyze_ratio, **kwargs):
        calls.append((batch_size, reanalyze_ratio, kwargs))
        return None, None, None, current_batch

    buffer._make_batch = make_batch

    sampled_current, target_batch = buffer.sample_world_model(1)

    assert sampled_current is current_batch
    assert calls == [(1, 0.0, {'orig_data': orig_data, 'prepare_target_context': False})]
    np.testing.assert_array_equal(target_batch[0], [[-1.0, 0.0, 0.0]])
    assert target_batch[1].shape == (1, 3)
    assert target_batch[2].shape == (1, 3, 3)
