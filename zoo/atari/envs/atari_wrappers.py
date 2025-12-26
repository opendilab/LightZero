# Adapted from openai baselines: https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
from datetime import datetime
from typing import Optional

import cv2
import gym  # For legacy API wrapper base class
import gymnasium  # For creating environments
import ale_py
import numpy as np
from ding.envs import NoopResetWrapper, MaxAndSkipWrapper, EpisodicLifeWrapper, FireResetWrapper, WarpFrameWrapper, \
    ScaledFloatFrameWrapper, \
    ClipRewardWrapper, FrameStackWrapper, TimeLimitWrapper
from ding.utils.compression_helper import jpeg_data_compressor
from easydict import EasyDict
from gymnasium.wrappers import RecordVideo


# only for reference now
# Note: If these functions are to be used with new environments, they also need similar gym/gymnasium compatibility modifications.
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
    # assert 'NoFrameskip' in env_id
    env = gymnasium.make(env_id)
    env = GymnasiumToGymWrapper(env)  # Add compatibility layer
    env = NoopResetWrapper(env, noop_max=30)
    env = MaxAndSkipWrapper(env, skip=1)
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
    # assert 'MontezumaRevenge' in env_id
    env = gymnasium.make(env_id)
    env = GymnasiumToGymWrapper(env)  # Add compatibility layer
    env = NoopResetWrapper(env, noop_max=30)
    env = MaxAndSkipWrapper(env, skip=1)
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


# This TimeLimit class can be replaced by ding.envs.TimeLimitWrapper for better consistency.
# However, if it needs to be retained, it now works correctly because it wraps the output of GymnasiumToGymWrapper.
class TimeLimit(gym.Wrapper):
    """
    Overview:
        A wrapper that limits the maximum number of steps in an episode.
    """
    def __init__(self, env: gym.Env, max_episode_steps: Optional[int] = None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps is not None and self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

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
    # Step 1: Create base environment using gymnasium
    if config.render_mode_human:
        env = gymnasium.make(config.env_id, render_mode='human', full_action_space=config.full_action_space)
    else:
        env = gymnasium.make(config.env_id, render_mode='rgb_array', full_action_space=config.full_action_space)

    # (Optional) Apply gymnasium native wrappers
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

    # Step 2: Add compatibility layer to convert gymnasium environment to gym interface
    env = GymnasiumToGymWrapper(env)

    # Step 3: Now safely apply all ding and legacy gym-style wrappers
    env = NoopResetWrapper(env, noop_max=30)
    env = MaxAndSkipWrapper(env, skip=config.frame_skip)
    if episode_life:
        env = EpisodicLifeWrapper(env)

    env = TimeLimit(env, max_episode_steps=config.max_episode_steps)

    if config.warp_frame:
        # we must set WarpFrame before ScaledFloatFrameWrapper
        env = WarpFrame(env, width=config.observation_shape[1], height=config.observation_shape[2], grayscale=config.gray_scale)
    if config.scale:
        env = ScaledFloatFrameWrapper(env)
    if clip_rewards:
        env = ClipRewardWrapper(env)

    env = JpegWrapper(env, transform2string=config.transform2string)
    if config.game_wrapper:
        env = GameWrapper(env)

    return env


# This wrapper inherits from gym.ObservationWrapper and now works correctly
class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, width: int = 84, height: int = 84, grayscale: bool = True,
                 dict_space_key: Optional[str] = None):
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
    def __init__(self, env: gym.Env, transform2string: bool = True):
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
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def legal_actions(self):
        return [_ for _ in range(self.env.action_space.n)]


# This is the key compatibility wrapper
class GymnasiumToGymWrapper(gym.Wrapper):
    """
    Overview:
        A wrapper class that adapts a Gymnasium environment to the Gym interface.
    """
    def __init__(self, env):
        # Ensure the input is a gymnasium environment
        assert isinstance(env, gymnasium.Env), f"Expected env to be a `gymnasium.Env` but got {type(env)}"
        super().__init__(env)
        self._seed = None

    def seed(self, seed):
        self._seed = seed
        # Call gymnasium's new seeder
        self.env.reset(seed=seed)

    def reset(self, **kwargs):
        # If seed is in kwargs, use it with priority
        if self._seed is not None:
            kwargs['seed'] = self._seed
            self._seed = None  # Seed only takes effect on first reset

        # Call gymnasium's reset, which returns (obs, info)
        result = self.env.reset(**kwargs)
        obs, info = result
        # Only return obs to match legacy gym API
        return obs

    def step(self, action):
        # Call gymnasium's step, which returns (obs, rew, terminated, truncated, info)
        obs, rew, terminated, truncated, info = self.env.step(action)
        # Merge terminated and truncated into done
        done = terminated or truncated
        # Return 4 values to match legacy gym API
        return obs, rew, done, info
