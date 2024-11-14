import copy
import random
from datetime import datetime
from typing import Union, Optional, Dict

import gymnasium as gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs import ObsPlusPrevActRewWrapper
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
import matplotlib.pyplot as plt
from matplotlib import animation


@ENV_REGISTRY.register('cartpole_lightzero')
class CartPoleEnv(BaseEnv):
    """
    LightZero version of the classic CartPole environment. This class includes methods for resetting, closing, and
    stepping through the environment, as well as seeding for reproducibility, saving replay videos, and generating random
    actions. It also includes properties for accessing the observation space, action space, and reward space of the
    environment.
    """

    config = dict(
        # env_id (str): The name of the CartPole environment.
        env_id="CartPole-v0",
        # enable_chance (bool): Whether to enable chance in observation.
        # If enabled, one of the first 3 values of observation will be multiplied by 2.
        # used for testing chance encoder in stochastic_muzero.
        # chance space is 3.
        enable_chance=False,
        # save_replay_gif (bool): If True, saves the replay as a gif.
        save_replay_gif=False,
        # replay_path_gif (str or None): The path to save the gif replay. If None, gif will not be saved.
        replay_path_gif=None,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict = {}) -> None:
        """
        Initializes the CartPole environment with the given configuration.

        Args:
            cfg (dict): Configuration dict that includes `env_id`, `save_replay_gif`, and `replay_path_gif`.
        """
        self._cfg = cfg
        self._enable_chance = self._cfg.get('enable_chance', False)
        self._init_flag = False
        self._replay_path_gif = cfg.get('replay_path_gif', None)
        self._save_replay_gif = cfg.get('save_replay_gif', False)
        self._save_replay_count = 0

        # Define observation, action, and reward spaces.
        self._observation_space = gym.spaces.Box(
            low=np.array([-4.8, float("-inf"), -0.42, float("-inf")]),
            high=np.array([4.8, float("inf"), 0.42, float("inf")]),
            shape=(4,),
            dtype=np.float32
        )
        self._action_space = gym.spaces.Discrete(2)
        self._reward_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the environment and return the initial observation.

        Returns:
            Dict[str, np.ndarray]: The initial observation from the environment.
        """
        if not self._init_flag:
            self._env = gym.make(self._cfg['env_id'], render_mode="rgb_array")
            # If replay saving as GIF is enabled, prepare for recording.
            if self._save_replay_gif:
                self._frames = []
            if hasattr(self._cfg, 'obs_plus_prev_action_reward') and self._cfg.obs_plus_prev_action_reward:
                self._env = ObsPlusPrevActRewWrapper(self._env)
            self._init_flag = True

        obs, _ = self._env.reset()
        self._eval_episode_return = 0
        obs = to_ndarray(obs)

        # Initialize the action mask and return the observation.
        action_mask = np.ones(self.action_space.n, 'int8')
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

        # this is to artificially introduce randomness in order to evaluate the performance of
        # stochastic_muzero on state input
        if self._enable_chance:
            chance_value = random.randint(0, 2)
            obs['observation'][chance_value] *= 2
            obs['chance'] = chance_value

        return obs

    def step(self, action: Union[int, np.ndarray]) -> BaseEnvTimestep:
        """
        Overview:
            Perform a step in the environment using the provided action, and return the next state of the environment.
            The next state is encapsulated in a BaseEnvTimestep object, which includes the new observation, reward,
            done flag, and info dictionary.
        Arguments:
            - action (:obj:`Union[int, np.ndarray]`): The action to be performed in the environment. If the action is
              a 1-dimensional numpy array, it is squeezed to a 0-dimension array.
        Returns:
            - timestep (:obj:`BaseEnvTimestep`): An object containing the new observation, reward, done flag,
              and info dictionary.
        .. note::
            - The cumulative reward (`_eval_episode_return`) is updated with the reward obtained in this step.
            - If the episode ends (done is True), the total reward for the episode is stored in the info dictionary
              under the key 'eval_episode_return'.
            - An action mask is created with ones, which represents the availability of each action in the action space.
            - Observations are returned in a dictionary format containing 'observation', 'action_mask', and 'to_play'.
        """
        if isinstance(action, np.ndarray) and action.shape == (1,):
            action = action.squeeze()  # Handle 0-dim array

        obs, rew, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated

        # Record the frame if replay saving as GIF is enabled.
        if self._save_replay_gif:
            self._frames.append(self._env.render())

        # Update rewards and check if the episode is done.
        self._eval_episode_return += rew
        if done:
            info['eval_episode_return'] = self._eval_episode_return
            if self._save_replay_gif:
                self.save_gif_replay()

        action_mask = np.ones(self.action_space.n, 'int8')
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

        # this is to artificially introduce randomness in order to evaluate the performance of
        # stochastic_muzero on state input
        if self._enable_chance:
            chance_value = random.randint(0, 2)
            obs['observation'][chance_value] *= 2
            obs['chance'] = chance_value

        return BaseEnvTimestep(obs, rew, done, info)

    def save_gif_replay(self) -> None:
        """
        Save the recorded frames as a GIF replay.
        """
        if not os.path.exists(self._replay_path_gif):
            os.makedirs(self._replay_path_gif)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        gif_filename = f'{self._cfg["env_id"]}_episode_{self._save_replay_count}_{timestamp}.gif'
        gif_path = os.path.join(self._replay_path_gif, gif_filename)

        # Create the GIF using the recorded frames.
        self.display_frames_as_gif(self._frames, gif_path)
        print(f"Replay saved as {gif_path}")
        self._save_replay_count += 1

    @staticmethod
    def display_frames_as_gif(frames: list, path: str) -> None:
        """
        Convert a list of frames into a GIF and save it.

        Args:
            frames (list): List of frames to be saved as a GIF.
            path (str): Path where the GIF will be saved.
        """
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
        anim.save(path, writer='imagemagick', fps=20)

    def close(self) -> None:
        """
        Close the environment and reset the initialization flag.
        """
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """
        Set the random seed for the environment.

        Args:
            seed (int): The seed value.
            dynamic_seed (bool): Whether to use dynamic seed generation.
        """
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def random_action(self) -> np.ndarray:
        """
        Generate a random action from the action space.

        Returns:
            np.ndarray: A random action.
        """
        random_action = self.action_space.sample()
        random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

    @property
    def observation_space(self) -> gym.spaces.Space:
        """
        Returns the observation space of the environment.
        """
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        """
        Returns the action space of the environment.
        """
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        """
        Returns the reward space of the environment.
        """
        return self._reward_space

    def __repr__(self) -> str:
        """
        String representation of the environment.
        """
        return f"LightZero CartPole Env({self._cfg['env_id']})"