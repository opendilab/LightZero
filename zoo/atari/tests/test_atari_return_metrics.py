"""Return diagnostics preserve the learning reward and label its actual unit."""

import gym
import numpy as np
import pytest

from zoo.atari.envs import atari_lightzero_env as env_module
from zoo.atari.envs.atari_wrappers import ClipRewardWrapper, RawRewardInfoWrapper
from lzero.worker.return_metrics import explicit_return_metrics


class RewardSequence(gym.Env):
    observation_space = gym.spaces.Box(0, 255, shape=(2, 2, 3), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def reset(self):
        self.index = 0
        self.info = {'lives': 2}
        self.observation = np.zeros((2, 2, 3), dtype=np.uint8)
        return self.observation

    def step(self, action):
        reward = [10., 200., -5., 0.][self.index]
        self.index += 1
        return self.observation, reward, self.index == 4, self.info


def test_raw_annotation_does_not_change_transition_or_mutate_inner_info():
    base = RewardSequence()
    wrapped = RawRewardInfoWrapper(base)
    observation = wrapped.reset()
    actual, reward, done, info = wrapped.step(0)
    assert actual is observation and reward == 10. and done is False
    assert info == {'lives': 2, 'raw_reward': 10.}
    assert base.info == {'lives': 2}


@pytest.mark.parametrize('clipped', [True, False])
@pytest.mark.parametrize('life', [True, False])
def test_env_reports_both_scales_and_resets_accumulators(monkeypatch, clipped, life):
    def make(*args, **kwargs):
        raw = RawRewardInfoWrapper(RewardSequence())
        return ClipRewardWrapper(raw) if clipped else raw

    monkeypatch.setattr(env_module, 'wrap_lightzero', make)
    env = env_module.AtariEnvLightZero(dict(
        env_id='test', observation_shape=(3, 2, 2), clip_rewards=clipped, episode_life=life))
    try:
        for _ in range(2):
            env.reset()
            transitions = [env.step(0) for _ in range(4)]
            assert [float(t.reward) for t in transitions] == ([1., 1., -1., 0.] if clipped else [10., 200., -5., 0.])
            assert [t.done for t in transitions] == [False, False, False, True]
            info = transitions[-1].info
            assert float(info['eval_episode_return']) == (1. if clipped else 205.)
            assert info['raw_episode_return'] == 205.
            assert info['clipped_episode_return'] == 1.
            unit = 'life' if life else 'game'
            assert explicit_return_metrics(info) == {
                f'raw_{unit}_return': 205., f'clipped_{unit}_return': 1.}
    finally:
        env.close()


def test_clipped_custom_wrapper_cannot_fabricate_raw_return(monkeypatch):
    monkeypatch.setattr(env_module, 'wrap_lightzero', lambda *a, **k: ClipRewardWrapper(RewardSequence()))
    env = env_module.AtariEnvLightZero(dict(env_id='test', observation_shape=(3, 2, 2), clip_rewards=True))
    try:
        env.reset()
        for _ in range(4):
            transition = env.step(0)
        assert explicit_return_metrics(transition.info) == {'clipped_life_return': 1.}
        assert 'raw_episode_return' not in transition.info
    finally:
        env.close()


def test_unannotated_environments_have_no_inferred_unit():
    assert explicit_return_metrics({'eval_episode_return': 7., 'raw_episode_return': 7.}) == {}
