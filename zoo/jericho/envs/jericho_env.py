import logging
import copy
import os
import json
import time
import signal
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import gym
import numpy as np
import torch
from transformers import AutoTokenizer

from ding.utils import ENV_REGISTRY, set_pkg_seed, get_rank, get_world_size
from ding.envs import BaseEnv, BaseEnvTimestep
from jericho import FrotzEnv


class TimeoutException(Exception):
    """Exception raised when an operation times out."""
    pass


@contextmanager
def timeout(seconds: float, error_message: str = 'Operation timed out'):
    """
    Context manager for timeout operations.

    Args:
        seconds: Timeout duration in seconds
        error_message: Error message to raise on timeout

    Usage:
        with timeout(5.0, "Step operation timed out"):
            result = some_operation()
    """
    def timeout_handler(signum, frame):
        raise TimeoutException(error_message)

    # Set up the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    # Schedule the alarm
    signal.setitimer(signal.ITIMER_REAL, seconds)

    try:
        yield
    finally:
        # Cancel the alarm and restore the old handler
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


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
        'for_unizero': False,
        'save_replay': False,
        'save_replay_path': None,
        'env_type': "zork1",
        'collect_policy_mode': "agent",
        # Robustness and debugging configurations
        'enable_timeout': True,
        'step_timeout': 30.0,  # Timeout for step operation in seconds
        'reset_timeout': 10.0,  # Timeout for reset operation in seconds
        'enable_debug_logging': False,  # Enable detailed debug logging
        'max_reset_retries': 3,  # Maximum number of reset retries on failure
        'max_step_retries': 2,   # Maximum number of step retries on failure
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

        # Robustness configurations
        self.enable_timeout: bool = self.cfg['enable_timeout']
        self.step_timeout: float = self.cfg['step_timeout']
        self.reset_timeout: float = self.cfg['reset_timeout']
        self.enable_debug_logging: bool = self.cfg['enable_debug_logging']
        self.max_reset_retries: int = self.cfg['max_reset_retries']
        self.max_step_retries: int = self.cfg['max_step_retries']

        # Performance monitoring
        self.step_count: int = 0
        self.total_step_time: float = 0.0
        self.total_reset_time: float = 0.0
        self.timeout_count: int = 0
        self.error_count: int = 0

        # Setup logger
        self.logger = logging.getLogger(f"JerichoEnv-{self.rank}")
        if self.enable_debug_logging:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        
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
                return {'observation': full_obs, 'action_mask': action_mask, 'to_play': -1, 'timestep': self._timestep}

            else:
                return {'observation': full_obs, 'action_mask': action_mask}
        else:
            if self.for_unizero:
                if self.save_replay:
                    return {'observation': full_obs, 'observation_str': full_obs_str,'obs_attn_mask': obs_attn_mask, 'action_mask': action_mask, 'to_play': -1, 'timestep': self._timestep}
                else:
                    return {'observation': full_obs, 'obs_attn_mask': obs_attn_mask, 'action_mask': action_mask, 'to_play': -1, 'timestep': self._timestep}
            else:
                return {'observation': full_obs, 'obs_attn_mask': obs_attn_mask, 'action_mask': action_mask}

    def reset(self, return_str: bool = False) -> Dict[str, Any]:
        """
        Overview:
            Reset the environment for a new episode with timeout protection and retry mechanism.

        Arguments:
            - return_str (:obj:`bool`, optional): If True, returns the observation as a raw string (defaults to False).

        Returns:
            - (:obj:`Dict[str, Any]`): The processed observation from the environment reset.
        """
        reset_start_time = time.time()

        for retry_idx in range(self.max_reset_retries):
            try:
                if self.enable_debug_logging:
                    self.logger.debug(f"[Rank {self.rank}] Reset attempt {retry_idx + 1}/{self.max_reset_retries}")

                if self.enable_timeout:
                    with timeout(self.reset_timeout, f"Reset operation timed out after {self.reset_timeout}s"):
                        initial_observation, info = self._env.reset()
                else:
                    initial_observation, info = self._env.reset()

                # Reset successful, initialize state
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
                        'obs': processed_obs['observation'] if return_str else processed_obs['observation_str'],
                        'act': None,
                        'done': False,
                        'info': info
                    })

                reset_duration = time.time() - reset_start_time
                self.total_reset_time += reset_duration

                if self.enable_debug_logging:
                    self.logger.debug(
                        f"[Rank {self.rank}] Reset successful in {reset_duration:.3f}s "
                        f"(attempt {retry_idx + 1})"
                    )

                return processed_obs

            except TimeoutException as e:
                self.timeout_count += 1
                self.logger.error(
                    f"[Rank {self.rank}] Reset timeout on attempt {retry_idx + 1}/{self.max_reset_retries}: {e}"
                )
                if retry_idx < self.max_reset_retries - 1:
                    # Recreate the environment for next retry
                    try:
                        self._env = FrotzEnv(self.game_path, 0)
                        time.sleep(0.5)  # Brief delay before retry
                    except Exception as recreate_error:
                        self.logger.error(f"[Rank {self.rank}] Failed to recreate environment: {recreate_error}")
                else:
                    raise

            except Exception as e:
                self.error_count += 1
                self.logger.error(
                    f"[Rank {self.rank}] Reset error on attempt {retry_idx + 1}/{self.max_reset_retries}: "
                    f"{type(e).__name__}: {e}"
                )
                if retry_idx < self.max_reset_retries - 1:
                    # Recreate the environment for next retry
                    try:
                        self._env = FrotzEnv(self.game_path, 0)
                        time.sleep(0.5)  # Brief delay before retry
                    except Exception as recreate_error:
                        self.logger.error(f"[Rank {self.rank}] Failed to recreate environment: {recreate_error}")
                else:
                    raise

        # If all retries failed, raise an exception
        raise RuntimeError(f"[Rank {self.rank}] Reset failed after {self.max_reset_retries} attempts")

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

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Overview:
            Get diagnostic information about the environment's performance and health.

        Returns:
            - (:obj:`Dict[str, Any]`): Dictionary containing diagnostic metrics.
        """
        avg_step_time = self.total_step_time / self.step_count if self.step_count > 0 else 0
        avg_reset_time = self.total_reset_time / max(1, self.step_count // self.max_steps) if self.step_count > 0 else 0

        return {
            'rank': self.rank,
            'total_steps': self.step_count,
            'total_step_time': self.total_step_time,
            'avg_step_time': avg_step_time,
            'total_reset_time': self.total_reset_time,
            'avg_reset_time': avg_reset_time,
            'timeout_count': self.timeout_count,
            'error_count': self.error_count,
            'current_timestep': self._timestep,
            'episode_return': self.episode_return,
        }

    def log_diagnostics(self) -> None:
        """
        Overview:
            Log current diagnostic information.
        """
        diagnostics = self.get_diagnostics()
        self.logger.info(
            f"[Rank {self.rank}] Diagnostics: "
            f"Steps={diagnostics['total_steps']}, "
            f"AvgStepTime={diagnostics['avg_step_time']:.3f}s, "
            f"Timeouts={diagnostics['timeout_count']}, "
            f"Errors={diagnostics['error_count']}"
        )

    def close(self) -> None:
        """
        Overview:
            Close the environment and release any resources.
        """
        # Log final diagnostics before closing
        if self.step_count > 0:
            self.log_diagnostics()

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
            Execute a single step in the environment using the provided action with timeout protection.

        Arguments:
            - action (:obj:`Union[int, np.ndarray, str]`): The action to execute. It can be an index to the valid actions list or a direct action string.
            - return_str (:obj:`bool`, optional): If True, returns the observation as a raw string (defaults to False).

        Returns:
            - (:obj:`BaseEnvTimestep`): A named tuple containing the observation, reward, done flag, and info.
        """
        step_start_time = time.time()

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
                self.logger.warning(
                    f'[Rank {self.rank}] Action {action} is illegal. Error: {e}. '
                    f'Randomly choosing from {len(self._action_list) if self._action_list else 0} available actions'
                )
                if self._action_list and len(self._action_list) > 0:
                    action = int(np.random.choice(len(self._action_list)))
                    action_str = self._action_list[action]
                else:
                    action_str = 'go'
                    self.logger.warning(f"[Rank {self.rank}] Available actions list empty. Using default action 'go'")

        previous_obs: Optional[str] = self.last_observation if (self.remove_stuck_actions and self.last_observation is not None) else None

        # Execute step with timeout and retry mechanism
        for retry_idx in range(self.max_step_retries):
            try:
                if self.enable_debug_logging:
                    self.logger.debug(
                        f"[Rank {self.rank}] Step {self._timestep} executing action '{action_str}' "
                        f"(attempt {retry_idx + 1}/{self.max_step_retries})"
                    )

                if self.enable_timeout:
                    with timeout(self.step_timeout, f"Step operation timed out after {self.step_timeout}s"):
                        observation, reward, done, info = self._env.step(action_str)
                else:
                    observation, reward, done, info = self._env.step(action_str)

                # Step successful
                self._timestep += 1
                self.step_count += 1

                if not self.for_unizero:
                    reward = np.array([float(reward)])
                self.episode_return += reward
                self._action_list = None

                # Detect and block ineffective (stuck) actions.
                if self.remove_stuck_actions and previous_obs is not None:
                    if observation == previous_obs:
                        self.blocked_actions.add(action_str)
                        self.logger.info(f'[Rank {self.rank}] Blocking action "{action_str}" - no observation change')

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

                step_duration = time.time() - step_start_time
                self.total_step_time += step_duration

                if self.enable_debug_logging:
                    self.logger.debug(
                        f"[Rank {self.rank}] Step {self._timestep} completed in {step_duration:.3f}s, "
                        f"reward={reward}, done={done}"
                    )

                if done:
                    avg_step_time = self.total_step_time / self.step_count if self.step_count > 0 else 0
                    self.logger.info(
                        f"[Rank {self.rank}] Episode done! Return: {self.episode_return:.2f}, "
                        f"Steps: {self._timestep}, Avg step time: {avg_step_time:.3f}s, "
                        f"Timeouts: {self.timeout_count}, Errors: {self.error_count}"
                    )
                    self.finished = True
                    info['eval_episode_return'] = self.episode_return

                    if self.save_replay:
                        self.save_episode_data()

                return BaseEnvTimestep(processed_obs, reward, done, info)

            except TimeoutException as e:
                self.timeout_count += 1
                self.logger.error(
                    f"[Rank {self.rank}] Step timeout on attempt {retry_idx + 1}/{self.max_step_retries} "
                    f"at timestep {self._timestep}, action '{action_str}': {e}"
                )
                if retry_idx < self.max_step_retries - 1:
                    time.sleep(0.2)  # Brief delay before retry
                else:
                    # On final retry failure, mark episode as done with abnormal flag
                    self.logger.error(
                        f"[Rank {self.rank}] Step failed after {self.max_step_retries} timeouts. "
                        f"Marking episode as done with abnormal flag."
                    )
                    info = {'abnormal': True, 'timeout': True, 'eval_episode_return': self.episode_return}
                    # Return a default observation with done=True
                    default_obs = self.prepare_obs("", return_str)
                    return BaseEnvTimestep(default_obs, 0.0, True, info)

            except Exception as e:
                self.error_count += 1
                self.logger.error(
                    f"[Rank {self.rank}] Step error on attempt {retry_idx + 1}/{self.max_step_retries} "
                    f"at timestep {self._timestep}, action '{action_str}': {type(e).__name__}: {e}"
                )
                if retry_idx < self.max_step_retries - 1:
                    time.sleep(0.2)  # Brief delay before retry
                else:
                    # On final retry failure, mark episode as done with abnormal flag
                    self.logger.error(
                        f"[Rank {self.rank}] Step failed after {self.max_step_retries} errors. "
                        f"Marking episode as done with abnormal flag."
                    )
                    info = {'abnormal': True, 'error': str(e), 'eval_episode_return': self.episode_return}
                    # Return a default observation with done=True
                    default_obs = self.prepare_obs("", return_str)
                    return BaseEnvTimestep(default_obs, 0.0, True, info)

        # Should not reach here, but just in case
        raise RuntimeError(f"[Rank {self.rank}] Step failed unexpectedly after all retries")

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