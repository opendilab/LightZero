"""Behavioral coverage for the Gymnasium migration, including zoo entry points."""
import ast
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from easydict import EasyDict

from lzero.envs.get_wrapped_env import get_wrappered_env
from lzero.envs.wrappers import ActionDiscretizationEnvWrapper, LightZeroEnvWrapper

pytestmark = pytest.mark.unittest


def wrapper_config(env_id, discretized=False):
    return EasyDict(
        env_id=env_id, is_train=True, continuous='Continuous' in env_id or env_id == 'Pendulum-v1',
        manually_discretization=discretized, each_dim_disc_size=3,
    )


@pytest.mark.parametrize('discretized', [False, True])
def test_factory_runs_two_complete_pendulum_episodes(discretized):
    cfg = wrapper_config('Pendulum-v1', discretized)
    cfg.continuous = not discretized
    env = get_wrappered_env(cfg, cfg.env_id)()
    try:
        env.seed(7, dynamic_seed=False)
        initial = []
        for _ in range(2):
            obs = env.reset()
            initial.append(obs['observation'].copy())
            assert isinstance(env.observation_space, gym.spaces.Dict)
            if discretized:
                assert isinstance(env.action_space, gym.spaces.Discrete)
                assert env.action_space.n == 3
                np.testing.assert_array_equal(obs['action_mask'], np.ones(3))
            else:
                assert isinstance(env.action_space, gym.spaces.Box)
                assert obs['action_mask'] is None
            total = 0.
            for step in range(200):
                timestep = env.step(env.random_action())
                total += float(timestep.reward.item())
                assert timestep.obs['observation'].shape == (3,)
                assert bool(timestep.done) == (step == 199)
            assert timestep.info['eval_episode_return'] == pytest.approx(total, rel=1e-6)
        # DI-engine's static seed behavior predates this migration.
        np.testing.assert_array_equal(*initial)
    finally:
        env.close()


@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v1'])
def test_legacy_gym_inputs_fail_with_actionable_error(env_id):
    legacy_gym = pytest.importorskip('gym')
    env = legacy_gym.make(env_id)
    try:
        with pytest.raises(TypeError, match='[Gg]ymnasium'):
            LightZeroEnvWrapper(env, wrapper_config(env_id))
        if env_id == 'Pendulum-v1':
            with pytest.raises(TypeError, match='[Gg]ymnasium'):
                ActionDiscretizationEnvWrapper(env, wrapper_config(env_id, True))
    finally:
        env.close()


@pytest.mark.parametrize('discretized', [False, True])
def test_direct_gymnasium_wrappers_handle_truncation(discretized):
    cfg = wrapper_config('Pendulum-v1', discretized)
    cfg.continuous = not discretized
    env = gym.make('Pendulum-v1', max_episode_steps=1)
    if discretized:
        env = ActionDiscretizationEnvWrapper(env, cfg)
    env = LightZeroEnvWrapper(env, cfg)
    try:
        for _ in range(2):
            assert env.reset(seed=5)['observation'].shape == (3,)
            timestep = env.step(env.action_space.sample())
            assert timestep.done
            assert timestep.info['eval_episode_return'] == pytest.approx(timestep.reward)
    finally:
        env.close()


def test_direct_gymnasium_termination():
    env = LightZeroEnvWrapper(gym.make('CartPole-v1'), wrapper_config('CartPole-v1'))
    try:
        env.reset(seed=11)
        # Put the pole outside its allowed angle so the next step terminates naturally.
        env.unwrapped.state = np.array([0., 0., 0.3, 0.])
        timestep = env.step(0)
        assert timestep.done
        assert timestep.info['eval_episode_return'] == timestep.reward
        assert env.reset()['observation'].shape == (4,)
    finally:
        env.close()


CONFIG_DIR = Path(__file__).resolve().parents[1] / 'zoo/box2d/lunarlander/config'


@pytest.mark.parametrize('path', sorted(CONFIG_DIR.glob('*config.py')), ids=lambda p: p.stem)
def test_lunarlander_config_environment_can_be_created(path):
    pytest.importorskip('Box2D')
    # Inspect the actual IDs without executing config-level GPU setup or training imports.
    ids = {node.value for node in ast.walk(ast.parse(path.read_text()))
           if isinstance(node, ast.Constant) and isinstance(node.value, str)
           and node.value.startswith('LunarLander')}
    for env_id in ids:
        env = get_wrappered_env(wrapper_config(env_id), env_id)()
        try:
            assert env.reset()['observation'].shape == (8,)
            assert env.step(env.random_action()).obs['observation'].shape == (8,)
        finally:
            env.close()


def test_mountain_car_truncation_and_repeated_recording(tmp_path, monkeypatch):
    from zoo.classic_control.mountain_car.envs.mtcar_lightzero_env import MountainCarEnv

    pytest.importorskip('pygame')
    pytest.importorskip('moviepy')
    monkeypatch.setenv('SDL_VIDEODRIVER', 'dummy')
    env = MountainCarEnv({})
    env.enable_save_replay(str(tmp_path))
    env.seed(17, dynamic_seed=False)
    try:
        first_recorder = None
        for episode in range(2):
            obs = env.reset()
            assert obs['timestep'] == 0
            assert isinstance(env._env, gym.wrappers.RecordVideo)
            if episode == 0:
                first_recorder = env._env
            else:
                assert env._env is first_recorder
            assert not isinstance(env._env.env, gym.wrappers.RecordVideo)
            for step in range(200):
                timestep = env.step(np.array([1]))  # no acceleration: force time-limit truncation
                assert bool(timestep.done) == (step == 199)
            assert timestep.info['eval_episode_return'] == -200
    finally:
        env.close()
    videos = list(tmp_path.glob('*.mp4'))
    assert len(videos) == 2
    assert all(path.stat().st_size > 0 for path in videos)


@pytest.mark.parametrize('episode_life', [False, True])
def test_atari_full_wrapper_chain_done_and_reset(monkeypatch, episode_life):
    pytest.importorskip('ale_py')
    from zoo.atari.envs import atari_wrappers as wrappers

    class FramesEnv(gym.Env):
        observation_space = gym.spaces.Box(0, 255, (96, 96, 3), np.uint8)
        action_space = gym.spaces.Discrete(3)

        def __init__(self):
            self.ale = self

        def lives(self):
            return 3

        def get_action_meanings(self):
            return ['NOOP', 'FIRE', 'RIGHT']

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.frames = 0
            return np.zeros((96, 96, 3), np.uint8), {}

        def step(self, action):
            self.frames += 1
            return np.full((96, 96, 3), self.frames % 256, np.uint8), 2., False, False, {}

    monkeypatch.setattr(wrappers.gymnasium, 'make', lambda *args, **kwargs: FramesEnv())
    cfg = EasyDict(env_id='PongNoFrameskip-v4', frame_skip=4, max_episode_steps=2,
                   warp_frame=True, observation_shape=(1, 84, 84), gray_scale=True,
                   scale=True, transform2string=False, game_wrapper=True)
    env = wrappers.wrap_lightzero(cfg, episode_life=episode_life, clip_rewards=True)
    try:
        for _ in range(2):
            assert env.reset().shape == (84, 84, 1)
            assert env.legal_actions() == [0, 1, 2]
            assert env.step(0)[2] is False
            obs, reward, done, info = env.step(0)
            assert done and info['TimeLimit.truncated']
            assert reward == 1.
            assert obs.dtype == np.float32
    finally:
        env.close()


def test_optional_jericho_spaces(monkeypatch):
    pytest.importorskip('jericho')
    from zoo.jericho.envs import jericho_env as module

    # Space construction must not download a tokenizer or require a game ROM.
    monkeypatch.setattr(module.JerichoEnv, 'tokenizer', object())
    monkeypatch.setattr(module, 'FrotzEnv', lambda *args: None)
    env = module.JerichoEnv({'game_path': 'unused.z5', 'max_action_num': 7})
    assert isinstance(env.observation_space, gym.spaces.Dict)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 7
    assert isinstance(env.reward_space, gym.spaces.Box)


def test_optional_pooltool_spaces():
    pytest.importorskip('pooltool')
    from zoo.pooltool.datatypes import Bounds, Spaces
    from zoo.pooltool.image_representation import RenderConfig
    from zoo.pooltool.sum_to_three.envs.sum_to_three_env import get_action_space
    from zoo.pooltool.sum_to_three.envs.utils import get_image_obs_space, get_reward_space

    spaces = Spaces(
        get_image_obs_space(RenderConfig.default()),
        get_action_space(Bounds(0., 2.), Bounds(0., 90.)),
        get_reward_space('binary'),
    )
    for space in (spaces.observation, spaces.action, spaces.reward):
        assert isinstance(space, gym.spaces.Box)
        assert space.contains(space.sample())
