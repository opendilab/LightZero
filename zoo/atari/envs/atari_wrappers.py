# Adapted from openai baselines: https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
from datetime import datetime
from typing import Optional

import cv2
import gymnasium 
import gym
import numpy as np
from ding.envs import NoopResetWrapper, MaxAndSkipWrapper, EpisodicLifeWrapper, FireResetWrapper, WarpFrameWrapper, \
    ScaledFloatFrameWrapper, \
    ClipRewardWrapper, FrameStackWrapper
from ding.utils.compression_helper import jpeg_data_compressor
from gym.wrappers import RecordVideo
from typing import Tuple
import gym
import numpy as np
from PIL import Image

# only for reference now
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


# only for reference now
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


def wrap_lightzero(config: EasyDict, episode_life: bool, clip_rewards: bool) -> gym.Env:
    """
    Overview:
        Configure environment for MuZero-style Atari. The observation is
        channel-first: (c, h, w) instead of (h, w, c).
    Arguments:
        - config (:obj:`Dict`): Dict containing configuration parameters for the environment.
        - episode_life (:obj:`bool`): If True, the agent starts with a set number of lives and loses them during the game.
        - clip_rewards (:obj:`bool`): If True, the rewards are clipped to a certain range.
    Return:
        - env (:obj:`gym.Env`): The wrapped Atari environment with the given configurations.
    """
    # TODO: full_action_space=True
    # full_action_space=True
    full_action_space=False

    if config.render_mode_human:
        env = gym.make(config.env_name, render_mode='human', full_action_space=full_action_space)
    else:
        env = gym.make(config.env_name, full_action_space=full_action_space)
    assert 'NoFrameskip' in env.spec.id
    if hasattr(config, 'save_replay') and config.save_replay \
            and hasattr(config, 'replay_path') and config.replay_path is not None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        video_name = f'{env.spec.id}-video-{timestamp}'
        env = RecordVideo(
            env,
            video_folder=config.replay_path,
            episode_trigger=lambda episode_id: True,
            name_prefix=video_name
        )
    env = GymnasiumToGymWrapper(env)
    env = NoopResetWrapper(env, noop_max=30)
    env = MaxAndSkipWrapper(env, skip=config.frame_skip)
    if episode_life:
        env = EpisodicLifeWrapper(env)
    env = TimeLimit(env, max_episode_steps=config.max_episode_steps)
    # TODO: LightZero
    if config.warp_frame:
        # we must set WarpFrame before ScaledFloatFrameWrapper # (84,84,3) -> (64,64,1)?
        env = WarpFrame(env, width=config.observation_shape[1], height=config.observation_shape[2], grayscale=config.gray_scale)
    # TODO: IRIS
    # env = ResizeObsWrapper(env, (config.observation_shape[1], config.observation_shape[2]))

    if config.scale:
        env = ScaledFloatFrameWrapper(env)
    if clip_rewards:
        env = ClipRewardWrapper(env)

    env = JpegWrapper(env, transform2string=config.transform2string)
    if config.game_wrapper:
        env = GameWrapper(env)

    return env


class TimeLimit(gym.Wrapper):
    """
    Overview:
        A wrapper that limits the maximum number of steps in an episode.
    """

    def __init__(self, env: gym.Env, max_episode_steps: Optional[int] = None):
        """
        Arguments:
            - env (:obj:`gym.Env`): The environment to wrap.
            - max_episode_steps (:obj:`Optional[int]`): Maximum number of steps per episode. If None, no limit is applied.
        """
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
    """
    Overview:
        A wrapper that warps frames to 84x84 as done in the Nature paper and later work.
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84, grayscale: bool = True,
                 dict_space_key: Optional[str] = None):
        """
        Arguments:
            - env (:obj:`gym.Env`): The environment to wrap.
            - width (:obj:`int`): The width to which the frames are resized.
            - height (:obj:`int`): The height to which the frames are resized.
            - grayscale (:obj:`bool`): If True, convert frames to grayscale.
            - dict_space_key (:obj:`Optional[str]`): If specified, indicates which observation should be warped.
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

# same as in IRIS
class ResizeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, size: Tuple[int, int]) -> None:
        gym.ObservationWrapper.__init__(self, env)
        self.size = tuple(size)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(size[0], size[1], 3), dtype=np.uint8)
        self.unwrapped.original_obs = None

    def resize(self, obs: np.ndarray):
        img = Image.fromarray(obs)
        img = img.resize(self.size, Image.BILINEAR)
        return np.array(img)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.unwrapped.original_obs = observation
        return self.resize(observation)


class JpegWrapper(gym.Wrapper):
    """
    Overview:
        A wrapper that converts the observation into a string to save memory.
    """

    def __init__(self, env: gym.Env, transform2string: bool = True):
        """
        Arguments:
            - env (:obj:`gym.Env`): The environment to wrap.
            - transform2string (:obj:`bool`): If True, transform the observations to string.
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
    """
    Overview:
        A wrapper to adapt the environment to the game interface.
    """

    def __init__(self, env: gym.Env):
        """
        Arguments:
            - env (:obj:`gym.Env`): The environment to wrap.
        """
        super().__init__(env)

    def legal_actions(self):
        return [_ for _ in range(self.env.action_space.n)]

class GymnasiumToGymWrapper(gym.Wrapper):
    """
    Overview:
        A wrapper class that adapts a Gymnasium environment to the Gym interface.
    Interface:
        ``__init__``, ``reset``, ``seed``
    Properties:
        - _seed (:obj:`int` or None): The seed value for the environment.
    """

    def __init__(self, env):
        """
        Overview:
            Initializes the GymnasiumToGymWrapper.
        Arguments:
            - env (:obj:`gymnasium.Env`): The Gymnasium environment to be wrapped.
        """

        assert isinstance(env, gymnasium.Env), type(env)
        super().__init__(env)
        self._seed = None

    def seed(self, seed):
        """
        Overview:
            Sets the seed value for the environment.
        Arguments:
            - seed (:obj:`int`): The seed value to use for random number generation.
        """
        self._seed = seed

    def reset(self):
        """
        Overview:
            Resets the environment and returns the initial observation.
        Returns:
            - observation (:obj:`Any`): The initial observation of the environment.
        """
        if self._seed is not None:
            obs, _ = self.env.reset(seed=self._seed)
            return obs
        else:
            obs, _ = self.env.reset()
            return obs