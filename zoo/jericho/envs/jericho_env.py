import copy
from typing import Any, Dict, List, Optional, Union

import gym
import numpy as np
import torch
from transformers import AutoTokenizer

from ding.utils import ENV_REGISTRY, set_pkg_seed, get_rank, get_world_size
from ding.envs import BaseEnv, BaseEnvTimestep
from jericho import FrotzEnv
import os
from datetime import datetime


@ENV_REGISTRY.register('jericho')
class JerichoEnv(BaseEnv):
    """
    JerichoEnv encapsulates a text game environment using Jericho and FrotzEnv.

    Overview:
        JerichoEnv represents a text game environment to train agents in text-based interactive games.

    Arguments:
        - max_steps (:obj:`int`): Maximum number of steps per episode.
        - game_path (:obj:`str`): The file path to the game file.
        - max_action_num (:obj:`int`): The maximum number of actions.
        - tokenizer_path (:obj:`str`): The name or path of the pretrained tokenizer.
        - max_seq_len (:obj:`int`): Maximum sequence length for tokenization of observations.
        - remove_stuck_actions (:obj:`bool`): Whether to remove actions that do not change the observation.
        - add_location_and_inventory (:obj:`bool`): Whether to include player location and inventory in the observation.
        - for_unizero (:obj:`bool`): If True, specify additional keys for unizero compatibility.
        - save_replay (:obj:`bool`): If True, the interaction log of the entire episode will be saved. 
        - save_replay_path (:obj:`str`): Path where interaction logs are saved.
        - collect_policy_mode (:obj:`str`): Data collection strategies, including "human" and "random".
        - n_collector_episodes (:obj:`int`): The total number of episodes of data collected by the data collectors.
        - env_type (:obj:`str`): Type of environment.
    Attributes:
        - tokenizer (Optional[AutoTokenizer]): The tokenizer loaded from the pretrained model.
    """
    tokenizer: Optional[AutoTokenizer] = None

    # Default configuration values can be set here as reference.
    DEFAULT_CONFIG: Dict[str, Any] = {
        'max_steps': 400,
        'max_action_num': 10,
        'tokenizer_path': "google-bert/bert-base-uncased",
        'max_seq_len': 512,
        'remove_stuck_actions': False,
        'add_location_and_inventory': False,
        'for_unizero': False,
        'save_replay': False,
        'save_replay_path': None,
        'env_type': "zork1",
        'collect_policy_mode': "random",
        'n_collector_episodes': 1
    }

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """
        Overview:
            Initialize the Jericho environment.

        Arguments:
            - cfg (:obj:`Dict[str, Any]`): Configuration dictionary containing keys like max_steps, game_path, etc.
        """
        merged_cfg = copy.deepcopy(self.DEFAULT_CONFIG)
        merged_cfg.update(cfg)
        self.cfg = merged_cfg

        self.max_steps: int = self.cfg['max_steps']
        self.game_path: str = self.cfg['game_path']
        self.max_action_num: int = self.cfg['max_action_num']
        self.max_seq_len: int = self.cfg['max_seq_len']
        self.save_replay: bool = self.cfg['save_replay']
        self.save_replay_path: str = self.cfg['save_replay_path']
        self.collect_policy_mode: str = self.cfg['collect_policy_mode']
        self.n_collector_episodes: int = self.cfg['n_collector_episodes']
        self.env_type: str = self.cfg['env_type']

        # Record the last observation and action for detecting stuck actions.
        self.last_observation: Optional[str] = None
        self.last_action: Optional[str] = None
        self.blocked_actions: set = set()

        # Get current world size and rank for distributed setups.
        self.world_size: int = get_world_size()
        self.rank: int = get_rank()

        # Read configuration values.
        self.remove_stuck_actions: bool = self.cfg['remove_stuck_actions']
        self.add_location_and_inventory: bool = self.cfg['add_location_and_inventory']
        self.for_unizero: bool = self.cfg['for_unizero']
        # Initialize the tokenizer once (only in rank 0 process if distributed)
        if JerichoEnv.tokenizer is None:
            if self.rank == 0:
                JerichoEnv.tokenizer = AutoTokenizer.from_pretrained(self.cfg['tokenizer_path'])
            if self.world_size > 1:
                # Wait until rank 0 finishes loading the tokenizer
                torch.distributed.barrier()
            if self.rank != 0:
                JerichoEnv.tokenizer = AutoTokenizer.from_pretrained(self.cfg['tokenizer_path'])

        # Initialize FrotzEnv with the given game.
        self._env: FrotzEnv = FrotzEnv(self.game_path, 0)
        self._action_list: Optional[List[str]] = None
        self.finished: bool = False
        self._init_flag: bool = False
        self.episode_return: float = 0.0
        self.env_step: int = 0
        self.timestep: int = 0
        self.episode_history: List[Dict[str, Any]] = []
        self.collected_episodes_experiences: List[List[Dict[str, Any]]] = []


        # Define observation, action, and reward spaces.
        self.observation_space: gym.spaces.Dict = gym.spaces.Dict()
        self.action_space: gym.spaces.Discrete = gym.spaces.Discrete(self.max_action_num)
        self.reward_space: gym.spaces.Box = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def prepare_obs(self, obs: str, return_str: bool = False) -> Dict[str, Any]:
        """
        Overview:
            Prepare the observation for the agent, including tokenization and the creation of an action mask.

        Arguments:
            - obs (:obj:`str`): The raw observation text.
            - return_str (:obj:`bool`, optional): If True, the observation is returned as a raw string (defaults to False).

        Returns:
            - (:obj:`Dict[str, Any]`): A dictionary containing the observation, attention mask (if applicable),
              and action mask. For unizero, an additional "to_play" key is provided.
        """
        if self._action_list is None:
            self._action_list = self._env.get_valid_actions()

        # Filter available actions based on whether stuck actions are removed.
        if self.remove_stuck_actions:
            available_actions: List[str] = [a for a in self._action_list if a not in self.blocked_actions]
            if len(available_actions) < 1 and len(self._action_list) > 0:
                # Fallback to the first action if all actions are blocked.
                available_actions = [self._action_list[0]]
            self._action_list = available_actions
        else:
            available_actions = self._action_list

        # Include player location and inventory in the observation if enabled.
        if self.add_location_and_inventory:
            player_location = self._env.get_player_location()
            inventory = self._env.get_inventory()
            full_obs: str = f"Location: {player_location}\nInventory: {inventory}{obs}\nValid actions: {available_actions}"
        else:
            full_obs = f"{obs}\nValid actions: {available_actions}"

        # Tokenize observation if required.
        if not return_str:
            tokenized_output = JerichoEnv.tokenizer(
                [full_obs], truncation=True, padding="max_length", max_length=self.max_seq_len)
            obs_attn_mask = tokenized_output['attention_mask']
            full_obs = np.array(tokenized_output['input_ids'][0], dtype=np.int32)
        # Create action mask based on the number of available actions.
        if len(available_actions) == 0:
            # Avoid an all-zero action mask that can cause segmentation faults.
            action_mask = [1] + [0] * (self.max_action_num - 1)
        elif 0 < len(available_actions) <= self.max_action_num:
            action_mask = [1] * len(available_actions) + [0] * (self.max_action_num - len(available_actions))
        elif len(available_actions) == self.max_action_num:
            action_mask = [1] * len(available_actions)
        else:
            action_mask = [1] * self.max_action_num

        action_mask = np.array(action_mask, dtype=np.int8)

        if return_str:
            if self.for_unizero:
                return {'observation': full_obs, 'action_mask': action_mask, 'to_play': -1}
            else:
                return {'observation': full_obs, 'action_mask': action_mask}
        else:
            if self.for_unizero:
                return {'observation': full_obs, 'obs_attn_mask': obs_attn_mask, 'action_mask': action_mask, 'to_play': -1}
            else:
                return {'observation': full_obs, 'obs_attn_mask': obs_attn_mask, 'action_mask': action_mask}

    def reset(self, return_str: bool = False) -> Dict[str, Any]:
        """
        Overview:
            Reset the environment for a new episode.

        Arguments:
            - return_str (:obj:`bool`, optional): If True, returns the observation as a raw string (defaults to False).

        Returns:
            - (:obj:`Dict[str, Any]`): The processed observation from the environment reset.
        """
        initial_observation, info = self._env.reset()
        self.finished = False
        self._init_flag = True
        self._action_list = None
        self.episode_return = 0.0
        self.env_step = 0
        self.timestep = 0
        self.episode_history = []

        if self.remove_stuck_actions:
            self.last_observation = initial_observation
        else:
            self.last_observation = None

        self.world_size = get_world_size()
        self.rank = get_rank()

        processed_obs = self.prepare_obs(initial_observation, return_str)
        self.episode_history.append({'observation': processed_obs['observation'], 'info': info, 'done': False})

        return processed_obs

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """
        Overview:
            Set the seed for the environment.

        Arguments:
            - seed (:obj:`int`): The seed value.
            - dynamic_seed (:obj:`bool`, optional): Whether to use a dynamic seed for randomness (defaults to True).
        """
        self._seed = seed
        self._env.seed(seed)

    def close(self) -> None:
        """
        Overview:
            Close the environment and release any resources.
        """
        self._init_flag = False

    def __repr__(self) -> str:
        """
        Overview:
            Return a string representation of the environment.

        Returns:
            - (:obj:`str`): String representation of the environment.
        """
        return "LightZero Jericho Env"

    def step(self, action: Union[int, np.ndarray, str], return_str: bool = False) -> BaseEnvTimestep:
        """
        Overview:
            Execute a single step in the environment using the provided action.

        Arguments:
            - action (:obj:`Union[int, np.ndarray, str]`): The action to execute. It can be an index to the valid actions list or a direct action string.
            - return_str (:obj:`bool`, optional): If True, returns the observation as a raw string (defaults to False).

        Returns:
            - (:obj:`BaseEnvTimestep`): A named tuple containing the observation, reward, done flag, and info.
        """
        # Clear previously blocked actions.
        self.blocked_actions = set()

        # Convert numerical action to string if necessary.
        if isinstance(action, str):
            action_str: str = action
        else:
            if isinstance(action, np.ndarray):
                action = int(action)
            try:
                action_str = self._action_list[action]
            except Exception as e:
                # Log error when illegal action is encountered.
                print('=' * 20)
                print(
                    e,
                    f'rank {self.rank}, action {action} is illegal. Randomly choosing a legal action from {self._action_list}!'
                )
                if self._action_list and len(self._action_list) > 0:
                    action = int(np.random.choice(len(self._action_list)))
                    action_str = self._action_list[action]
                else:
                    action_str = 'go'
                    print(
                        f"rank {self.rank}, available actions list empty. Using default action 'go'."
                    )

        previous_obs: Optional[str] = self.last_observation if (self.remove_stuck_actions and self.last_observation is not None) else None

        observation, reward, done, info = self._env.step(action_str)

        self.timestep += 1
        if not self.for_unizero:
            reward = np.array([float(reward)])
        self.env_step += 1
        self.episode_return += reward
        self._action_list = None

        # Detect and block ineffective (stuck) actions.
        if self.remove_stuck_actions and previous_obs is not None:
            if observation == previous_obs:
                self.blocked_actions.add(action_str)
                print(f'[Removing action] "{action_str}" as it did not change the observation.')

        if self.remove_stuck_actions:
            self.last_observation = observation

        processed_obs = self.prepare_obs(observation, return_str)

        if self.env_step >= self.max_steps:
            done = True

        self.episode_history.append({
            "step": self.env_step,
            "observation": processed_obs['observation'],
            "action": action_str,
            "reward": reward.item() if isinstance(reward, np.ndarray) else reward,
            "done": done,
            "info": info
        })

        if done:
            print('=' * 20)
            print(f'rank {self.rank} one episode done!')
            self.finished = True
            info['eval_episode_return'] = self.episode_return
            if self.save_replay:
                self.save_episode_data()

        return BaseEnvTimestep(processed_obs, reward, done, info)

    @staticmethod
    def create_collector_env_cfg(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Overview:
            Create a list of environment configuration dictionaries for the collector phase.

        Arguments:
            - cfg (:obj:`Dict[str, Any]`): The original environment configuration.

        Returns:
            - (:obj:`List[Dict[str, Any]]`): A list of configuration dictionaries for collector environments.
        """
        collector_env_num: int = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg['is_collect'] = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Overview:
            Create a list of environment configuration dictionaries for the evaluator phase.

        Arguments:
            - cfg (:obj:`Dict[str, Any]`): The original environment configuration.

        Returns:
            - (:obj:`List[Dict[str, Any]]`): A list of configuration dictionaries for evaluator environments.
        """
        evaluator_env_num: int = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg['reward_normalize'] = False
        cfg['is_collect'] = False
        return [cfg for _ in range(evaluator_env_num)]

    def save_episode_data(self):
        """
        Save the full episode history to a JSON file.
        """
        if self.save_replay_path is None:
            self.save_replay_path = './log'  
        os.makedirs(self.save_replay_path, exist_ok=True)

        timestamp = datetime.now().strftime("%m%d_%H%M")
        filename = os.path.join(self.save_replay_path, f"episode_record_{self.env_type}_{timestamp}.txt")
        try:
            with open(filename, mode="w", encoding="utf-8") as f:
                for i, item in enumerate(self.episode_history):
                    item['observation'] = item['observation'].strip('\n')
                    if i == 0:  # Starting Status
                        f.write(f"[ENV]\tmoves: {item['info']['moves']}\t\tscore: {item['info']['score']}\n{item['observation']}\n\n")
                    else:
                        f.write(f"[PLAYER]\tstep: {item['step']}\naction: {item['action']}\nreward: {item['reward']}\n\n")
                        f.write(f"[ENV]\tmoves: {item['info']['moves']}\t\tscore: {item['info']['score']}\n{item['observation']}\n\n")
                    if item['done']:
                        f.write(f"[SYSTEM]\tThe game is over, and the reward for this game is {self.episode_return}.\n")
                        
            print(f"[INFO] Episode saved to {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save episode: {e}")

    def human_step(self, observation:str) -> str:
        """
        Get action input from human player.
        """
        print(f"[OBS]\n{observation}")
        while True:
            try:
                action_id = int(input('Please input the action id (the id starts from zero): '))
                return action_id
            except ValueError:  
                print("Invalid input. Please enter an integer action id.")
    
    def random_step(self) -> str:
        """
        Get a random valid action.
        """
        if self._action_list is not None and len(self._action_list)>0:
            return np.random.choice(self._action_list)
        else:
            print(
                f"rank {self.rank}, available actions list empty. Using default action 'go'."
            )
            return 'go'

    def collect_episode_data(self):
        """
        Collects episode data for the specified number of episodes.
        """

        for _ in range(self.n_collector_episodes):
            obs = self.reset(return_str=True)
                
            done = False
            state = obs['observation']
            episode_experiences = []

            while not done:
                if self.collect_policy_mode == 'human':
                    action = self.human_step(state)
                elif self.collect_policy_mode == 'random':
                    action = self.random_step()
                else:
                    raise ValueError(f"Invalid collect_policy_mode: {self.collect_policy_mode}")

                next_obs, reward, done, info = self.step(action, return_str=True)
                next_state = next_obs['observation']

                experience = {
                    'state': state,
                    'action': action,
                    'reward': reward.item() if isinstance(reward, np.ndarray) else reward,
                    'next_state': next_state,
                    'done': done
                }
                episode_experiences.append(experience)

                state = next_state
                
                if done:
                    self.collected_episodes_experiences.append(episode_experiences)
                    print(f'The game is over, and the reward for this game is {self.episode_return}.\n')
                    break
                    
        # print(f'collected_episodes_experiences={self.collected_episodes_experiences}')

if __name__ == '__main__':
    from easydict import EasyDict

    # Configuration dictionary for the environment.
    env_cfg = EasyDict(
        dict(
            max_steps=400,
            game_path="./zoo/jericho/envs/z-machine-games-master/jericho-game-suite/" + "zork1.z5",
            max_action_num=10,
            tokenizer_path="google-bert/bert-base-uncased",
            max_seq_len=512,
            remove_stuck_actions=False,
            add_location_and_inventory=False,
            for_unizero=False,
            collector_env_num=1,
            evaluator_env_num=1,
            save_replay=True,
            save_replay_path=None,
            env_type='zork1',        # zork1, acorncourt, detective, omniquest
            collect_policy_mode='random',
            n_collector_episodes=1
        )
    )
    env = JerichoEnv(env_cfg)
    env.collect_episode_data()  

    obs = env.reset(return_str=True)
    print(f'[OBS]:\n{obs["observation"]}')
    while True:
        try:
            action_id = int(input('Please input the action id (the id starts from zero): '))
        except ValueError:
            print("Invalid input. Please enter an integer action id.")
            continue
        obs, reward, done, info = env.step(action_id, return_str=True)
        print(f'[OBS]:\n{obs["observation"]}')
        if done:
            user_choice = input('Would you like to RESTART, RESTORE a saved game, give the FULL score for that game or QUIT? ')
            del env  
            break