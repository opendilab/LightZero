import logging
import copy
import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import gym
import numpy as np
import torch
from transformers import AutoTokenizer

from ding.utils import ENV_REGISTRY, set_pkg_seed, get_rank, get_world_size
from ding.envs import BaseEnv, BaseEnvTimestep
from jericho import FrotzEnv


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
        - collect_policy_mode (:obj:`str`): The strategy pattern used in data collection in the collect_episode_data method, including "human", "random" and "expert".
        - env_type (:obj:`str`): Type of environment.
    Attributes:
        - tokenizer (Optional[AutoTokenizer]): The tokenizer loaded from the pretrained model.
    """
    tokenizer: Optional[AutoTokenizer] = None

    # Default configuration values can be set here as reference.
    DEFAULT_CONFIG: Dict[str, Any] = {
        'max_steps': 400,
        'max_action_num': 10,
        'tokenizer_path': "BAAI/bge-base-en-v1.5",
        'max_seq_len': 512,
        'remove_stuck_actions': False,
        'add_location_and_inventory': False,
        # 'for_unizero': False,
        'for_unizero': True,
        'save_replay': False,
        'save_replay_path': None,
        'env_type': "zork1",
        'collect_policy_mode': "agent"
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
        self.env_type: str = self.cfg['env_type']

        self.max_action_num: int = self.cfg['max_action_num']
        self.max_seq_len: int = self.cfg['max_seq_len']
        self.save_replay: bool = self.cfg['save_replay']
        self.save_replay_path: str = self.cfg['save_replay_path']
        self.collect_policy_mode: str = self.cfg['collect_policy_mode']

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
        self._timestep: int = 0
        self.episode_history: Optional[List[Dict[str, Any]]] = None
        self.walkthrough_actions: Optional[List[str]] = None


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
        # [PRIORZERO-NEW] Store raw observation text before processing
        raw_obs_text = obs  # Save original text BEFORE any modification

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
        
        full_obs_str = copy.deepcopy(full_obs)
        
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
                return {
                    'observation': full_obs,
                    'action_mask': action_mask,
                    'to_play': -1,
                    'timestep': self._timestep,
                    'valid_actions': available_actions,  # [PRIORZERO] Add valid actions list
                    'raw_obs_text': raw_obs_text  # [PRIORZERO-NEW] Add raw text
                }

            else:
                return {
                    'observation': full_obs,
                    'action_mask': action_mask,
                    'valid_actions': available_actions,  # [PRIORZERO] Add valid actions list
                    'raw_obs_text': raw_obs_text  # [PRIORZERO-NEW] Add raw text
                }
        else:
            if self.for_unizero:
                if self.save_replay:
                    return {
                        'observation': full_obs,
                        'observation_str': full_obs_str,
                        'obs_attn_mask': obs_attn_mask,
                        'action_mask': action_mask,
                        'to_play': -1,
                        'timestep': self._timestep,
                        'valid_actions': available_actions,  # [PRIORZERO] Add valid actions list
                        'raw_obs_text': raw_obs_text  # [PRIORZERO-NEW] Add raw text
                    }
                else:
                    return {
                        'observation': full_obs,
                        'obs_attn_mask': obs_attn_mask,
                        'action_mask': action_mask,
                        'to_play': -1,
                        'timestep': self._timestep,
                        'valid_actions': available_actions,  # [PRIORZERO] Add valid actions list
                        'raw_obs_text': raw_obs_text  # [PRIORZERO-NEW] Add raw text
                    }
            else:
                return {
                    'observation': full_obs,
                    'obs_attn_mask': obs_attn_mask,
                    'action_mask': action_mask,
                    'valid_actions': available_actions,  # [PRIORZERO] Add valid actions list
                    'raw_obs_text': raw_obs_text  # [PRIORZERO-NEW] Add raw text
                }

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
        self._timestep = 0
        self.episode_history = []
        if self.collect_policy_mode == 'expert':
            self.walkthrough_actions = self._env.get_walkthrough()

        if self.remove_stuck_actions:
            self.last_observation = initial_observation
        else:
            self.last_observation = None

        self.world_size = get_world_size()
        self.rank = get_rank()

        processed_obs = self.prepare_obs(initial_observation, return_str)

        if self.save_replay:
            self.episode_history.append({
                'timestep': 0,
                'obs': processed_obs['observation'] if return_str else processed_obs['observation_str'] ,
                'act': None,
                'done': False,
                'info': info
            })

       
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
        info['action_str'] = action_str

        self._timestep += 1
        if not self.for_unizero:
            reward = np.array([float(reward)])
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

        if self._timestep >= self.max_steps:
            done = True

        if self.save_replay:
            self.episode_history.append({
                'timestep': self._timestep,
                'obs': processed_obs['observation'] if return_str else processed_obs['observation_str'],
                'act': action_str,
                'reward': reward.item() if isinstance(reward, np.ndarray) else reward,
                'done': done,
                'info': info
            })

        if done:
            print('=' * 20)
            print(f'rank {self.rank} one episode done! episode_return:{self.episode_return}')
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
        Overview:
            Save the full episode interaction history (self.episode_history) to a JSON file.
        """
        if self.save_replay_path is None:
            self.save_replay_path = './log'  
        os.makedirs(self.save_replay_path, exist_ok=True)

        timestamp = datetime.now().strftime("%m%d_%H%M")
        filename = os.path.join(self.save_replay_path, f"episode_record_{self.env_type}_{self.collect_policy_mode}_{timestamp}.json")
        
        info = self.episode_history[-1]['info']
        if 'eval_episode_return' in info and isinstance(info['eval_episode_return'], np.ndarray):
            info['eval_episode_return'] = info['eval_episode_return'].item()

        with open(filename, mode="w", encoding="utf-8") as f:
            json.dump(self.episode_history, f, ensure_ascii=False)
            logging.info(
                f"Episode data successfully saved to '{filename}'. "
                f"Episode length: {len(self.episode_history)} interactions, "
                f"Environment type: {self.env_type}, Policy mode: {self.collect_policy_mode}."
            )
         
    def human_step(self, observation:str) -> str:
        """
        Overview:
            Interactively receive an action from a human player via command line input.

        Arguments:
            - observation (:obj:`str`): The current observation shown to the human.

        Returns:
            - (:obj:`int`): The action index input by the user, converted to int.
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
        Overview:
            Randomly select a valid action from the current valid action list.

        Returns:
            - (:obj:`str`): A randomly selected action string from the available actions. If no actions are available, returns 'go' as a fallback.
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
        Overview:
            Run a single episode using the specified policy mode, and store the trajectory in self.episode_history.
        """

        obs = self.reset(return_str=True)

        done = False
        expert_step_count = 0

        while not done:
            if self.collect_policy_mode == 'human':
                action = self.human_step(obs['observation'])
            elif self.collect_policy_mode == 'random':
                action = self.random_step()
            elif self.collect_policy_mode == 'expert':
                action = self.walkthrough_actions[expert_step_count]
                expert_step_count += 1
            else:
                raise ValueError(f"Invalid collect_policy_mode: {self.collect_policy_mode}")

            obs, reward, done, info = self.step(action, return_str=True)

            if self.collect_policy_mode == 'expert' and expert_step_count >= len(self.walkthrough_actions):
                done = True

            if done:
                info['eval_episode_return'] = self.episode_return
                break
     
if __name__ == '__main__':
    from easydict import EasyDict

    env_type='detective' # zork1, acorncourt, detective, omniquest
    # Configuration dictionary for the environment.
    env_cfg = EasyDict(
        dict(
            max_steps=400,
            game_path="./zoo/jericho/envs/z-machine-games-master/jericho-game-suite/" + f"{env_type}.z5",
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
            env_type=env_type,
            collect_policy_mode='expert'    # random, human, expert
        )
    )
    env = JerichoEnv(env_cfg)
    # Collect data for an episode according to collect_policy_mode
    env.collect_episode_data()  
    del env  