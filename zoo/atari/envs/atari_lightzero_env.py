import copy
import sys
from typing import List, Any

import gymnasium as gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

from zoo.atari.envs.atari_wrappers import wrap_lightzero


@ENV_REGISTRY.register('atari_lightzero')
class AtariLightZeroEnv(BaseEnv):
    """
    Overview:
        AtariLightZeroEnv is a derived class from BaseEnv and represents the environment for the Atari LightZero game.
        This class provides the necessary interfaces to interact with the environment, including reset, step, seed,
        close, etc. and manages the environment's properties such as observation_space, action_space, and reward_space.
    Properties:
        cfg, _init_flag, channel_last, clip_rewards, episode_life, _env, _observation_space, _action_space,
        _reward_space, obs, _eval_episode_return, has_reset, _seed, _dynamic_seed
    """
    config = dict(
        # (int) The number of environment instances used for data collection.
        collector_env_num=8,
        # (int) The number of environment instances used for evaluator.
        evaluator_env_num=3,
        # (int) The number of episodes to evaluate during each evaluation period.
        n_evaluator_episode=3,
        # (str) The name of the Atari game environment.
        # env_id='PongNoFrameskip-v4',
        # (str) The type of the environment, here it's Atari.
        env_type='Atari',
        observation_shape=(4, 96, 96),
        collect_max_episode_steps=int(1.08e5),
        # (int) The maximum number of steps in each episode during evaluation.
        eval_max_episode_steps=int(1.08e5),
        # (bool) If True, the game is rendered in real-time.
        render_mode_human=False,
        # (bool) If True, a video of the game play is saved.
        save_replay=False,
        # replay_path (str or None): The path to save the replay video. If None, the replay will not be saved.
        # Only effective when env_manager.type is 'base'.
        replay_path=None,
        # (bool) If set to True, the game screen is converted to grayscale, reducing the complexity of the observation space.
        gray_scale=True,
        # (int) The number of frames to skip between each action. Higher values result in faster simulation.
        frame_skip=4,
        # (bool) If True, the game ends when the agent loses a life, otherwise, the game only ends when all lives are lost.
        episode_life=True,
        # (bool) If True, the rewards are clipped to a certain range, usually between -1 and 1, to reduce variance.
        clip_rewards=True,
        # (bool) If True, the channels of the observation images are placed last (e.g., height, width, channels).
        # Default is False, which means the channels are placed first (e.g., channels, height, width).
        channel_last=False,
        # (bool) If True, the pixel values of the game frames are scaled down to the range [0, 1].
        scale=True,
        # (bool) If True, the game frames are preprocessed by cropping irrelevant parts and resizing to a smaller resolution.
        warp_frame=True,
        # (bool) If True, the game state is transformed into a string before being returned by the environment.
        transform2string=False,
        # (bool) If True, additional wrappers for the game environment are used.
        game_wrapper=True,
        # (dict) The configuration for the environment manager. If shared_memory is set to False, each environment instance
        # runs in the same process as the trainer, otherwise, they run in separate processes.
        manager=dict(shared_memory=False, ),
        # (int) The value of the cumulative reward at which the training stops.
        stop_value=int(1e6),
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        """
        Overview:
            Return the default configuration for the Atari LightZero environment.
        Arguments:
            - cls (:obj:`type`): The class AtariLightZeroEnv.
        Returns:
            - cfg (:obj:`EasyDict`): The default configuration dictionary.
        """
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: EasyDict) -> None:
        """
        Overview:
            Initialize the Atari LightZero environment with the given configuration.
        Arguments:
            - cfg (:obj:`EasyDict`): The configuration dictionary.
        """
        self.cfg = cfg
        self._init_flag = False
        self.channel_last = cfg.channel_last
        self.clip_rewards = cfg.clip_rewards
        self.episode_life = cfg.episode_life

    def reset(self) -> dict:
        """
        Overview:
            Reset the environment and return the initial observation.
        Returns:
            - obs (:obj:`dict`): The initial observation after reset.
        """
        if not self._init_flag:
            # Create and return the wrapped environment for Atari LightZero.
            self._env = wrap_lightzero(self.cfg, episode_life=self.cfg.episode_life, clip_rewards=self.cfg.clip_rewards)
            self._observation_space = self._env.env.observation_space
            self._action_space = self._env.env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.env.reward_range[0], high=self._env.env.reward_range[1], shape=(1,), dtype=np.float32
            )

            self._init_flag = True

        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.env.seed(self._seed)

        obs = self._env.reset()
        self.obs = to_ndarray(obs)
        self._eval_episode_return = 0.
        obs = self.observe()
        return obs

    def step(self, action: int) -> BaseEnvTimestep:
        """
        Overview:
            Execute the given action and return the resulting environment timestep.
        Arguments:
            - action (:obj:`int`): The action to be executed.
        Returns:
            - timestep (:obj:`BaseEnvTimestep`): The environment timestep after executing the action.
        """
        observation = self.obs

        if not self.channel_last:
            # move the channel dim to the fist axis
            # (96, 96, 3) -> (3, 96, 96)
            observation = np.transpose(observation, (2, 0, 1))

        action_mask = np.ones(self._action_space.n, 'int8')
        # action_mask = np.ones(18, 'int8')  # TODO: full action space

        return {'observation': observation, 'action_mask': action_mask, 'to_play': -1}

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self.obs = to_ndarray(obs)
        self.reward = np.array(reward).astype(np.float32)
        self._eval_episode_return += self.reward
        observation = self.observe()
        if done:
            print('one episode done!')
            info['eval_episode_return'] = self._eval_episode_return

        return BaseEnvTimestep(observation, self.reward, done, info)

    def observe(self) -> dict:
        """
        Overview:
            Return the current observation along with the action mask and to_play flag.
        Returns:
            - observation (:obj:`dict`): The dictionary containing current observation, action mask, and to_play flag.
        """
        observation = self.obs

        if not self.channel_last:
            # move the channel dim to the fist axis
            # (96, 96, 3) -> (3, 96, 96)
            observation = np.transpose(observation, (2, 0, 1))

        action_mask = np.ones(self._action_space.n, 'int8')
        return {'observation': observation, 'action_mask': action_mask, 'to_play': -1}

    @property
    def legal_actions(self):
        return np.arange(self._action_space.n)

    def random_action(self):
        action_list = self.legal_actions
        return np.random.choice(action_list)

    def close(self) -> None:
        """
        Close the environment, and set the initialization flag to False.
        """
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """
        Set the seed for the environment's random number generator. Can handle both static and dynamic seeding.
        """
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    @property
    def observation_space(self) -> gym.spaces.Space:
        """
        Property to access the observation space of the environment.
        """
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        """
        Property to access the action space of the environment.
        """
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        """
        Property to access the reward space of the environment.
        """
        return self._reward_space

    def __repr__(self) -> str:
        return "LightZero Atari Env({})".format(self.cfg.env_id)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.max_episode_steps = cfg.collect_max_episode_steps
        cfg.episode_life = True
        cfg.clip_rewards = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.max_episode_steps = cfg.eval_max_episode_steps
        cfg.episode_life = False
        # cfg.episode_life = True
        cfg.clip_rewards = False
        return [cfg for _ in range(evaluator_env_num)]
