# Borrow a lot from openai baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

from typing import Tuple

import cv2
import gym
import numpy as np
from PIL import Image
from ding.envs import NoopResetWrapper, MaxAndSkipWrapper, EpisodicLifeWrapper, FireResetWrapper, WarpFrameWrapper, \
    ScaledFloatFrameWrapper, \
    ClipRewardWrapper, FrameStackWrapper
from ding.utils.compression_helper import jpeg_data_compressor
from gym.wrappers import RecordVideo


def wrap_deepmind(env_id, episode_life=True, clip_rewards=True, frame_stack=4, scale=True, warp_frame=True):
    """Configure environment for DeepMind-style Atari. The observation is
    channel-first: (c, h, w) instead of (h, w, c).

    :param str env_id: the atari environment id.
    :param bool episode_life: wrap the episode life wrapper.
    :param bool clip_rewards: wrap the reward clipping wrapper.
    :param int frame_stack: wrap the frame stacking wrapper.
    :param bool scale: wrap the scaling observation wrapper.
    :param bool warp_frame: wrap the grayscale + resize observation wrapper.
    :return: the wrapped atari environment.
    """
    assert 'NoFrameskip' in env_id
    env = gym.make(env_id)
    env = NoopResetWrapper(env, noop_max=30)
    env = MaxAndSkipWrapper(env, skip=4)
    if episode_life:
        env = EpisodicLifeWrapper(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetWrapper(env)
    if warp_frame:
        env = WarpFrameWrapper(env)
    if scale:
        env = ScaledFloatFrameWrapper(env)
    if clip_rewards:
        env = ClipRewardWrapper(env)
    if frame_stack:
        env = FrameStackWrapper(env, frame_stack)
    return env


def wrap_deepmind_mr(env_id, episode_life=True, clip_rewards=True, frame_stack=4, scale=True, warp_frame=True):
    """Configure environment for DeepMind-style Atari. The observation is
    channel-first: (c, h, w) instead of (h, w, c).

    :param str env_id: the atari environment id.
    :param bool episode_life: wrap the episode life wrapper.
    :param bool clip_rewards: wrap the reward clipping wrapper.
    :param int frame_stack: wrap the frame stacking wrapper.
    :param bool scale: wrap the scaling observation wrapper.
    :param bool warp_frame: wrap the grayscale + resize observation wrapper.
    :return: the wrapped atari environment.
    """
    assert 'MontezumaRevenge' in env_id
    env = gym.make(env_id)
    env = NoopResetWrapper(env, noop_max=30)
    env = MaxAndSkipWrapper(env, skip=4)
    if episode_life:
        env = EpisodicLifeWrapper(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetWrapper(env)
    if warp_frame:
        env = WarpFrameWrapper(env)
    if scale:
        env = ScaledFloatFrameWrapper(env)
    if clip_rewards:
        env = ClipRewardWrapper(env)
    if frame_stack:
        env = FrameStackWrapper(env, frame_stack)
    return env


def wrap_lightzero(config, episode_life, clip_rewards):
    """
    Overview:
        Configure environment for MuZero-style Atari. The observation is
        channel-first: (c, h, w) instead of (h, w, c).
    Arguments:
        - config (:obj:`Dict`): Dict containing configuration.
        - wrap_frame (:obj:`bool`):
        - save_video (:obj:`bool`):
        - save_path (:obj:`bool`):
    Return:
        - the wrapped atari environment.
    """
    # TODO: full_action_space=True
    full_action_space = False

    if config.render_mode_human:
        env = gym.make(config.env_id, render_mode='human', full_action_space=full_action_space)
    else:
        env = gym.make(config.env_id, full_action_space=full_action_space)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetWrapper(env, noop_max=30)
    env = MaxAndSkipWrapper(env, skip=config.frame_skip)
    if episode_life:
        env = EpisodicLifeWrapper(env)
    env = TimeLimit(env, max_episode_steps=config.max_episode_steps)
    if config.warp_frame:
        env = WarpFrame(env, width=config.observation_shape[1], height=config.observation_shape[2], grayscale=config.gray_scale)

    if config.scale:
        env = ScaledFloatFrameWrapper(env)
    if clip_rewards:
        env = ClipRewardWrapper(env)
    if config.save_video:
        import random, string
        env = RecordVideo(
            env,
            video_folder=config.save_path,
            episode_trigger=lambda episode_id: True,
            name_prefix='rl-video-{}'.format(''.join(random.choice(string.ascii_lowercase) for i in range(5))),
        )

    env = JpegWrapper(env, transform2string=config.transform2string)
    if config.game_wrapper:
        env = GameWrapper(env)

    return env


class TimeLimit(gym.Wrapper):

    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class WarpFrame(gym.ObservationWrapper):

    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class JpegWrapper(gym.Wrapper):

    def __init__(self, env, transform2string=True):
        """
        Overview: convert the observation into string to save memory
        """
        super().__init__(env)
        self.transform2string = transform2string

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        if self.transform2string:
            observation = jpeg_data_compressor(observation)

        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)

        if self.transform2string:
            observation = jpeg_data_compressor(observation)

        return observation


class GameWrapper(gym.Wrapper):

    def __init__(self, env):
        """
        Overview: warp env to adapt the game interface
        """
        super().__init__(env)

    def legal_actions(self):
        return [_ for _ in range(self.env.action_space.n)]