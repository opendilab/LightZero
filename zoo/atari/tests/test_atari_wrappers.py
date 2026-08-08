from easydict import EasyDict

from zoo.atari.envs.atari_wrappers import _atari_make_kwargs


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
