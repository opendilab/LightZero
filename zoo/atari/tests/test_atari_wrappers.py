from types import SimpleNamespace

import gym
import numpy as np
from easydict import EasyDict

from zoo.atari.envs.atari_wrappers import (
    MaxAndSkipWrapper,
    _atari_make_kwargs,
    _validate_ale_base_semantics,
)


def test_ale_v5_base_env_does_not_double_apply_frame_skip_or_sticky_actions():
    cfg = EasyDict(env_id='ALE/Pong-v5', render_mode_human=False, full_action_space=False)

    assert _atari_make_kwargs(cfg) == {
        'render_mode': 'rgb_array',
        'full_action_space': False,
        'frameskip': 1,
        'repeat_action_probability': 0.0,
    }


def test_legacy_noframeskip_env_keeps_registry_semantics():
    cfg = EasyDict(env_id='PongNoFrameskip-v4', render_mode_human=True, full_action_space=True)

    assert _atari_make_kwargs(cfg) == {
        'render_mode': 'human',
        'full_action_space': True,
    }


def test_legacy_config_without_optional_make_fields_uses_ale_defaults():
    cfg = EasyDict(env_id='PongNoFrameskip-v4')

    assert _atari_make_kwargs(cfg) == {
        'render_mode': 'rgb_array',
        'full_action_space': False,
    }


def test_invalid_ale_registry_semantics_raise_even_with_python_optimized():
    env = SimpleNamespace(
        spec=SimpleNamespace(kwargs={'frameskip': 4, 'repeat_action_probability': 0.25})
    )

    try:
        _validate_ale_base_semantics(env)
    except RuntimeError as error:
        assert 'action-repeat semantics' in str(error)
    else:
        raise AssertionError('invalid ALE base semantics must not be accepted')


def test_outer_macro_action_advances_exactly_four_frames_and_sums_rewards():
    class _CountingEnv(gym.Env):
        observation_space = gym.spaces.Box(0, 255, shape=(2, 2, 1), dtype=np.uint8)
        action_space = gym.spaces.Discrete(2)

        def __init__(self):
            self.frames = 0

        def reset(self):
            self.frames = 0
            return np.zeros((2, 2, 1), dtype=np.uint8)

        def step(self, action):
            self.frames += 1
            observation = np.full((2, 2, 1), self.frames, dtype=np.uint8)
            return observation, float(self.frames), False, {}

    base_env = _CountingEnv()
    wrapped = MaxAndSkipWrapper(base_env, skip=4)
    _, reward, done, _ = wrapped.step(0)

    assert base_env.frames == 4
    assert reward == 1.0 + 2.0 + 3.0 + 4.0
    assert done is False
