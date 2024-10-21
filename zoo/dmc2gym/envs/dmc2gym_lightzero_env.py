import copy
from datetime import datetime
from typing import Optional, Callable, Union, Dict

import dmc2gym
import gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs import WarpFrameWrapper, ScaledFloatFrameWrapper, ClipRewardWrapper, ActionRepeatWrapper, \
    FrameStackWrapper
from ding.envs.common.common_function import affine_transform
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from gym.spaces import Box


def dmc2gym_observation_space(dim, minimum=-np.inf, maximum=np.inf, dtype=np.float32) -> Callable:
    def observation_space(from_pixels=True, height=84, width=84, channels_first=True) -> Box:
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            return Box(low=0, high=255, shape=shape, dtype=np.uint8)
        else:
            return Box(np.repeat(minimum, dim).astype(dtype), np.repeat(maximum, dim).astype(dtype), dtype=dtype)

    return observation_space


def dmc2gym_state_space(dim, minimum=-np.inf, maximum=np.inf, dtype=np.float32) -> Box:
    return Box(np.repeat(minimum, dim).astype(dtype), np.repeat(maximum, dim).astype(dtype), dtype=dtype)


def dmc2gym_action_space(dim, minimum=-1, maximum=1, dtype=np.float32) -> Box:
    return Box(np.repeat(minimum, dim).astype(dtype), np.repeat(maximum, dim).astype(dtype), dtype=dtype)


def dmc2gym_reward_space(minimum=0, maximum=1, dtype=np.float32) -> Callable:
    def reward_space(frame_skip=1) -> Box:
        return Box(
            np.repeat(minimum * frame_skip, 1).astype(dtype),
            np.repeat(maximum * frame_skip, 1).astype(dtype),
            dtype=dtype
        )

    return reward_space


dmc2gym_env_info = {
    "ball_in_cup": {
        "catch": {
            "observation_space": dmc2gym_observation_space(8),
            "state_space": dmc2gym_state_space(8),
            "action_space": dmc2gym_action_space(2),
            "reward_space": dmc2gym_reward_space()
        }
    },
    "cartpole": {
        "balance": {
            "observation_space": dmc2gym_observation_space(5),
            "state_space": dmc2gym_state_space(4),
            "action_space": dmc2gym_action_space(1),
            "reward_space": dmc2gym_reward_space()
        },
        "swingup": {
            "observation_space": dmc2gym_observation_space(5),
            "state_space": dmc2gym_state_space(4),
            "action_space": dmc2gym_action_space(1),
            "reward_space": dmc2gym_reward_space()
        }
    },
    "cheetah": {
        "run": {
            "observation_space": dmc2gym_observation_space(17),
            "state_space": dmc2gym_state_space(18),
            "action_space": dmc2gym_action_space(6),
            "reward_space": dmc2gym_reward_space()
        }
    },
    "finger": {
        "spin": {
            "observation_space": dmc2gym_observation_space(9),
            "state_space": dmc2gym_state_space(9),
            "action_space": dmc2gym_action_space(1),
            "reward_space": dmc2gym_reward_space()
        }
    },
    "reacher": {
        "easy": {
            "observation_space": dmc2gym_observation_space(6),
            "state_space": dmc2gym_state_space(6),
            "action_space": dmc2gym_action_space(2),
            "reward_space": dmc2gym_reward_space()
        }
    },
    "walker": {
        "walk": {
            "observation_space": dmc2gym_observation_space(24),
            "state_space": dmc2gym_state_space(24),
            "action_space": dmc2gym_action_space(6),
            "reward_space": dmc2gym_reward_space()
        }
    },
    "hopper": {
        "hop": {
            "observation_space": dmc2gym_observation_space(15),
            "state_space": dmc2gym_state_space(14),
            "action_space": dmc2gym_action_space(4),
            "reward_space": dmc2gym_reward_space()
        }
    },
    "humanoid": {
        "run": {
            "observation_space": dmc2gym_observation_space(67),
            "state_space": dmc2gym_state_space(54),
            "action_space": dmc2gym_action_space(21),
            "reward_space": dmc2gym_reward_space()
        }
    }
}


@ENV_REGISTRY.register('dmc2gym_lightzero')
class DMC2GymEnv(BaseEnv):
    """
    LightZero version of the DeepMind Control Suite to gym environment. This class includes methods for resetting, 
    closing, and stepping through the environment, as well as seeding for reproducibility, saving replay videos, 
    and generating random actions. It also includes properties for accessing the observation space, action space, 
    and reward space of the environment.
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = dict(
        domain_name=None,
        task_name=None,
        frame_skip=4,
        warp_frame=False,
        scale=False,
        clip_rewards=False,
        action_repeat=1,
        frame_stack=1,
        from_pixels=True,
        visualize_reward=False,
        height=84,
        width=84,
        channels_first=True,
        resize=84,
        replay_path=None,
    )

    def __init__(self, cfg: dict = {}) -> None:
        """
        Initialize the environment with a configuration dictionary. Sets up spaces for observations, actions, and rewards.
        """
        default_config = self.default_config()
        default_config.update(cfg)
        self._cfg = default_config

        assert self._cfg.domain_name in dmc2gym_env_info, '{}/{}'.format(self._cfg.domain_name, dmc2gym_env_info.keys())
        assert self._cfg.task_name in dmc2gym_env_info[
            self._cfg.domain_name], '{}/{}'.format(self._cfg.task_name, dmc2gym_env_info[self._cfg.domain_name].keys())

        self._init_flag = False
        self._replay_path = self._cfg.replay_path

        self._observation_space = dmc2gym_env_info[self._cfg.domain_name][self._cfg.task_name]["observation_space"](
            from_pixels=self._cfg["from_pixels"],
            height=self._cfg["height"],
            width=self._cfg["width"],
            channels_first=self._cfg["channels_first"]
        )
        self._action_space = dmc2gym_env_info[self._cfg.domain_name][self._cfg.task_name]["action_space"]
        self._reward_space = dmc2gym_env_info[self._cfg.domain_name][self._cfg.task_name]["reward_space"](
            self._cfg["frame_skip"])

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the environment. If it hasn't been initialized yet, this method also handles that. It also handles seeding
        if necessary. Returns the first observation.
        """
        if not self._init_flag:
            self._env = dmc2gym.make(
                domain_name=self._cfg["domain_name"],
                task_name=self._cfg["task_name"],
                seed=1,
                visualize_reward=self._cfg["visualize_reward"],
                from_pixels=self._cfg["from_pixels"],
                height=self._cfg["height"],
                width=self._cfg["width"],
                frame_skip=self._cfg["frame_skip"],
                channels_first=self._cfg["channels_first"],
            )

            # optional env wrapper
            if self._cfg['warp_frame']:
                self._env = WarpFrameWrapper(self._env, size=self._cfg['resize'])
            if self._cfg['scale']:
                self._env = ScaledFloatFrameWrapper(self._env)
            if self._cfg['clip_rewards']:
                self._env = ClipRewardWrapper(self._env)
            if self._cfg['action_repeat']:
                self._env = ActionRepeatWrapper(self._env, self._cfg['action_repeat'])
            if self._cfg['frame_stack'] > 1:
                self._env = FrameStackWrapper(self._env, self._cfg['frame_stack'])

            # set the obs, action space of wrapped env
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space

            if self._replay_path is not None:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                video_name = f'{self._env.spec.id}-video-{timestamp}'
                self._env = gym.wrappers.RecordVideo(
                    self._env,
                    video_folder=self._replay_path,
                    episode_trigger=lambda episode_id: True,
                    name_prefix=video_name
                )

            self._init_flag = True

        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)

        self._eval_episode_return = 0
        obs = self._env.reset()  # This line will cause errors when subprocess_env_manager is used
        obs = obs['state'] 
        obs = to_ndarray(obs).astype(np.float32)
        action_mask = None

        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

        return obs

    def close(self) -> None:
        """
        Close the environment if it was initialized.
        """
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """
        Set the seed for the environment.
        """
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: Union[int, np.ndarray]) -> BaseEnvTimestep:
        """
        Step the environment forward with the provided action. This method returns the next state of the environment
        (observation, reward, done flag, and info dictionary) encapsulated in a BaseEnvTimestep object.
        """
        action = action.astype('float32')
        action = affine_transform(action, min_val=self._env.action_space.low, max_val=self._env.action_space.high)
        obs, rew, done, info = self._env.step(action)
        self._eval_episode_return += rew
        obs = to_ndarray(obs).astype(np.float32)
        rew = to_ndarray([rew]).astype(np.float32)  # wrapped to be transferred to an array with shape (1,)

        if done:
            info['eval_episode_return'] = self._eval_episode_return

        action_mask = None
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

        return BaseEnvTimestep(obs, rew, done, info)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        """
        Enable saving replay videos to the specified path.
        """
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    def random_action(self) -> np.ndarray:
        """
        Generate a random action using the action space's sample method. Returns a numpy array containing the action.
        """
        random_action = self.action_space.sample().astype(np.float32)
        return random_action

    @property
    def observation_space(self) -> gym.spaces.Space:
        """
        Get the observation space of the environment.
        """
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        """
        Get the action space of the environment.
        """
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        """
        Get the reward space of the environment.
        """
        return self._reward_space

    def __repr__(self) -> str:
        """
        String representation of the environment.
        """
        return "LightZero DMC2Gym Env({}:{})".format(self._cfg["domain_name"], self._cfg["task_name"])
