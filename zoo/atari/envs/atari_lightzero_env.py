import copy
from ditk import logging
from typing import List

import gym
import ale_py
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

from zoo.atari.envs.atari_wrappers import wrap_lightzero


def _prepare_reset_seed(base_seed: int, dynamic_seed: bool, rng=None) -> int:
    """Choose an ALE seed without mutating NumPy's process-global RNG.

    ``AtariEnvLightZero`` owns a per-environment ``RandomState`` for dynamic seed generation.
    The legacy ``NoopResetWrapper`` is seeded only inside ``_reset_with_numpy_seed`` so a base
    environment manager cannot accidentally perturb replay sampling in the main process.
    """
    random_source = np.random if rng is None else rng
    reset_seed = base_seed + 100 * random_source.randint(1, 1000) if dynamic_seed else base_seed
    return int(reset_seed)


def _reset_with_numpy_seed(env, reset_seed: int):
    """Run legacy reset-time NumPy sampling reproducibly and restore caller RNG state."""
    process_rng_state = np.random.get_state()
    try:
        np.random.seed(reset_seed)
        return env.reset()
    finally:
        np.random.set_state(process_rng_state)


@ENV_REGISTRY.register('atari_lightzero')
class AtariEnvLightZero(BaseEnv):
    """
    Overview:
        AtariEnvLightZero is a derived class from BaseEnv and represents the environment for the Atari LightZero game.
        This class provides the necessary interfaces to interact with the environment, including reset, step, seed,
        close, etc. and manages the environment's properties such as observation_space, action_space, and reward_space.
    Properties:
        cfg, _init_flag, channel_last, clip_rewards, episode_life, _env, _observation_space, _action_space,
        _reward_space, obs, _eval_episode_return, has_reset, _seed, _dynamic_seed
    """
    config = dict(
        # (bool) Whether to use the full action space of the environment. Default is False. If set to True, the action space size is 18 for Atari.
        full_action_space=False,
        # (int) The number of environment instances used for data collection.
        collector_env_num=8,
        # (int) The number of environment instances used for evaluator.
        evaluator_env_num=3,
        # (int) The number of episodes to evaluate during each evaluation period.
        n_evaluator_episode=3,
        # (str) The type of the environment, here it's Atari.
        env_type='Atari',
        # (tuple) The shape of the observation space, which is a stacked frame of 4 images each of 96x96 pixels.
        observation_shape=(4, 96, 96),
        # (int) The maximum number of steps in each episode during data collection.
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
        # (int) Specifies the number of consecutive frames to stack after collecting environment data.
        # The stacking process is applied within the collector and evaluator modules.
        frame_stack_num=1,
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
            - cls (:obj:`type`): The class AtariEnvLightZero.
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
        # Direct users and older tests often pass a partial environment config,
        # while DI-engine's compiled path supplies every default explicitly.
        # Normalize both paths once so wrappers and space construction observe
        # the same fields. ``obs_shape`` is the historical alias.
        normalized_cfg = self.default_config()
        normalized_cfg.update(cfg)
        if 'obs_shape' in cfg and 'observation_shape' not in cfg:
            normalized_cfg.observation_shape = tuple(cfg.obs_shape)
        self.cfg = normalized_cfg
        self._init_flag = False
        self.channel_last = self.cfg.channel_last
        self.clip_rewards = self.cfg.clip_rewards
        self.episode_life = self.cfg.episode_life
        self._timestep = 0

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

            observation_space_before_stack = (
                int(self.cfg.observation_shape[0] / self.cfg.frame_stack_num),
                self.cfg.observation_shape[1],
                self.cfg.observation_shape[2]
            )

            self._action_space = self._env.action_space

            self._observation_space = gym.spaces.Dict({
                'observation': gym.spaces.Box(
                    low=0, high=1, shape=observation_space_before_stack, dtype=np.float32
                ),
                'action_mask': gym.spaces.Box(
                    low=0, high=1, shape=(self._action_space.n,), dtype=np.int8
                ),
                'to_play': gym.spaces.Box(
                    low=-1, high=2, shape=(), dtype=np.int8
                ),
                'timestep': gym.spaces.Box(
                    low=0, high=self.cfg.collect_max_episode_steps, shape=(), dtype=np.int32
                ),
            })

            # self._reward_space = gym.spaces.Box(
            #     low=self._env.env.reward_range[0], high=self._env.env.reward_range[1], shape=(1,), dtype=np.float32
            # )
            self._reward_space = gym.spaces.Box(
                low=-9999, high=9999, shape=(1,), dtype=np.float32
            )

            self._init_flag = True

        if hasattr(self, '_seed'):
            reset_seed = _prepare_reset_seed(
                self._seed,
                getattr(self, '_dynamic_seed', True),
                rng=self._seed_rng,
            )
            self._env.seed(reset_seed)
            result = _reset_with_numpy_seed(self._env, reset_seed)
        else:
            result = self._env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result

        self.obs = to_ndarray(obs)
        self._eval_episode_return = 0.
        self._timestep = 0
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
        obs, reward, done, info = self._env.step(action)
        self.obs = to_ndarray(obs)
        self.reward = np.array(reward).astype(np.float32)
        self._eval_episode_return += self.reward
        self._timestep += 1
        # if self._timestep % 200 == 0:
        #     logging.info(f'self._timestep: {self._timestep}')
        observation = self.observe()
        if done:
            logging.info(f'one episode done! total episode length is: {self._timestep}')
            info['eval_episode_return'] = self._eval_episode_return
            logging.debug(f'one episode of {self.cfg.env_id} done')

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

        return {'observation': observation, 'action_mask': action_mask, 'to_play': np.array(-1), 'timestep': np.array(self._timestep)}

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
        self._seed_rng = np.random.RandomState(self._seed)

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
        cfg.clip_rewards = False
        return [cfg for _ in range(evaluator_env_num)]
