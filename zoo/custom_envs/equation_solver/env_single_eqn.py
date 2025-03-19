import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import numpy as np
from sympy import sympify, symbols
from gymnasium import spaces
from operator import add, sub, mul, truediv

# Import custom helper functions (assumed available)
from utils_custom_functions import *
from utils_env import *

# LightZero compatibility
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

logger = logging.getLogger(__name__)

@ENV_REGISTRY.register('singleEqn_env')
class singleEqn(BaseEnv):
    """Environment for solving simple algebraic equations using RL in a LightZero‐compatible format."""

    config = dict(
        env_id="singleEqn-v0",
        battle_mode='self_play_mode',
        battle_mode_in_simulation_env='self_play_mode',
        render_mode=None,
        replay_path=None,
        bot_action_type=None,
        agent_vs_human=False,
        prob_random_agent=0,
        prob_expert_agent=0,
        prob_random_action_in_bot=0.0,
        scale=False,
        stop_value=None    
        )

    metadata = {"render_modes": ["human"]}

    def __init__(self, env_fn=None, cfg=None, main_eqn='x+b', state_rep='integer_1d', normalize_rewards=True, cache=False, \
        verbose=False):
        #super().__init__(env_fn, cfg)
        self.cfg = EasyDict(cfg or self.config)
        self.battle_mode = self.cfg.get('battle_mode', 'self_play_mode')
        self.battle_mode_in_simulation_env = self.cfg.get('battle_mode_in_simulation_env', 'self_play_mode')
        
        # Parameters from configuration
        self.max_expr_length = 20
        self.max_steps = 10
        self.action_dim = 50
        self.observation_dim = 2 * self.max_expr_length + 1

        # Reward settings
        self.reward_solved = 100
        self.reward_invalid_equation = -100
        self.reward_illegal_action = -100
        self.episode_return = 0

        # Options
        self.normalize_rewards = normalize_rewards
        self.state_rep = state_rep
        self.verbose = verbose
        self.cache = cache
        if self.cache:
            self.action_cache = {}  # Cache for dynamic actions

        # Set up the equation and symbols
        self.main_eqn = sympify(main_eqn)
        self.lhs = self.main_eqn
        self.rhs = 0
        self.x = symbols('x')

        # Build feature dictionary and initial fixed actions
        self.setup()

        # Initial state vector
        obs, _ = self.to_vec(self.lhs, self.rhs)
        self.obs = np.array(obs, dtype=np.float32)  #to work with muzero
        self.current_steps = 0

        # Define action and observation spaces (LightZero expects these as properties)
        self._action_space = spaces.Discrete(self.action_dim)
        self._reward_space = spaces.Box(low=self.reward_invalid_equation, high=self.reward_solved, shape=(1,), dtype=np.float32)
        if state_rep == 'integer_1d':
            self._observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                                  shape=(self.observation_dim,), dtype=np.float32)
        elif state_rep == 'integer_2d':
            self._observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                                  shape=(self.observation_dim, 2), dtype=np.float32)
        elif state_rep in ['graph_integer_1d', 'graph_integer_2d']:
            self._observation_space = spaces.Dict({
                "node_features": spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(self.observation_dim, 2), dtype=np.float32),
                "edge_index": spaces.Box(low=0, high=self.observation_dim, 
                                         shape=(2, 2*self.observation_dim), dtype=np.int32),
                "node_mask": spaces.Box(low=0, high=1, 
                                        shape=(self.observation_dim,), dtype=np.int32),
                "edge_mask": spaces.Box(low=0, high=1, 
                                        shape=(2*self.observation_dim,), dtype=np.int32),
            })
        else:
            raise ValueError(f"Unsupported state representation: {state_rep}")

    def setup(self):
        # Build feature dictionary (e.g., {'add': -1, 'x': 1, 'a':2, ...})
        self.feature_dict = make_feature_dict(self.main_eqn, self.state_rep)

        # Define fixed actions (operations paired with a term)
        self.actions_fixed = [
            # (custom_expand, None),
            # (custom_factor, None),
            # (custom_collect, self.x), 
            # (custom_together, None),
            # (custom_ratsimp, None),
            # (custom_square, None),
            # (custom_sqrt, None),
            (sub, -1)
        ]
        # Generate dynamic actions based on the current equation
        if self.cache:
            self.actions, self.action_mask = make_actions_cache(self.lhs, self.rhs,
                                                                  self.actions_fixed,
                                                                  self.action_dim,
                                                                  self.action_cache)
        else:
            self.actions, self.action_mask = make_actions(self.lhs, self.rhs,
                                                           self.actions_fixed,
                                                           self.action_dim)

    def step(self, action_index: int) -> BaseEnvTimestep:
        # Recompute dynamic actions since they depend on the current state
        lhs_old, rhs_old = self.lhs, self.rhs

        # Double check mask
        if action_index not in self.legal_actions:
            print(f'\nIllegal action taken: action_index = {action_index}')
            #print(f'\nIllegal action taken: action_index = {self.action_mask}\n')
            legal_actions_temp = [i for i, valid in enumerate(self.action_mask) if valid]
            action_index = np.random.choice(legal_actions_temp)

        # Apply the selected action
        action = self.actions[action_index]
        operation, term = action
        lhs_new, rhs_new = operation(lhs_old, term), operation(rhs_old, term)
        obs_new, _ = self.to_vec(lhs_new, rhs_new)

        # Validate the new equation and check for a solved state
        is_valid_eqn, lhs_new, rhs_new = check_valid_eqn(lhs_new, rhs_new)
        is_solved = check_eqn_solved(lhs_new, rhs_new, self.main_eqn)



        # Compute reward based on equation complexity changes
        reward = self.find_reward(lhs_old, rhs_old, lhs_new, rhs_new, is_valid_eqn, is_solved)
        self.episode_return += reward

        # Termination conditions: solved, exceeded max steps, or invalid equation
        too_many_steps = self.current_steps >= self.max_steps
        done = bool(is_solved or too_many_steps or not is_valid_eqn)
        truncated = False

        # Update state and step counter
        self.lhs, self.rhs, self.obs = lhs_new, rhs_new, np.array(obs_new, dtype=np.float32)
        self.current_steps += 1

        # Update actions
        self.actions, self.action_mask  = make_actions(lhs_new, rhs_new, self.actions_fixed, self.action_dim)

        info = {
            'is_solved': is_solved,
            'is_valid_eqn': is_valid_eqn,
            'too_many_steps': too_many_steps,
            'action_mask': self.action_mask
        }

        if done:
            info['eval_episode_return'] = self.episode_return

        verbose = True
        if verbose:
            #print(f'{self.lhs} = {self.rhs}. (Operation, term): {operation_names.get(operation, operation)}, {term} | reward = {reward:.2f}')
            print(f'(Operation, term): {operation_names.get(operation, operation)}, {term}')

            if is_solved:
                print(f'\nSOLVED: {self.lhs} = {self.rhs}\n')

        lightzero_obs_dict = {
            'observation': np.array(obs_new, dtype=np.float32),
            'action_mask': self.action_mask,
            'board': np.array(obs_new, dtype=np.float32),  
            'current_player_index': 0,  #
            'to_play': -1
        }
        return BaseEnvTimestep(lightzero_obs_dict, reward, done, info)

    def reset(self, seed=0, options=None, **kwargs):
        # You can optionally capture start_player_index if needed:
        start_player_index = kwargs.get('start_player_index', 0)
        # Then proceed with the reset
        self.current_steps = 0
        self.lhs, self.rhs = self.main_eqn, 0
        self.actions, self.action_mask  = make_actions(self.lhs, self.rhs, self.actions_fixed, self.action_dim)
        obs, _ = self.to_vec(self.lhs, self.rhs)
        self.obs = np.array(obs, dtype=np.float32)
        self.episode_return = 0
        lightzero_obs_dict = {
            'observation': obs,
            'board': obs,  # for compatibility
            'action_mask': self.action_mask,
            'to_play': -1,
            'current_player_index': start_player_index 
        }
        return lightzero_obs_dict


    def render(self, mode: str = "human"):
        print(f'{self.lhs} = {self.rhs}')

    def to_vec(self, lhs, rhs):
        if self.state_rep == 'integer_1d':
            return integer_encoding_1d(lhs, rhs, self.feature_dict, self.max_expr_length)
        elif self.state_rep == 'integer_2d':
            return integer_encoding_2d(lhs, rhs, self.feature_dict, self.max_expr_length)
        elif self.state_rep in ['graph_integer_1d', 'graph_integer_2d']:
            return graph_encoding(lhs, rhs, self.feature_dict, self.max_expr_length)
        else:
            raise ValueError(f"Unknown state representation: {self.state_rep}")

    def find_reward(self, lhs_old, rhs_old, lhs_new, rhs_new, is_valid_eqn, is_solved):
        if not is_valid_eqn:
            reward = self.reward_invalid_equation
        elif is_solved:
            reward = self.reward_solved
        else:
            obs_old_complexity = get_complexity_expression(lhs_old) + get_complexity_expression(rhs_old)
            obs_new_complexity = get_complexity_expression(lhs_new) + get_complexity_expression(rhs_new)
            reward = obs_old_complexity - obs_new_complexity

        if self.normalize_rewards:
            max_reward, min_reward = self.reward_solved, self.reward_invalid_equation
            reward = 2 * (reward - min_reward) / (max_reward - min_reward) - 1
        return reward

    def get_valid_action_mask(self):
        return self.action_mask

    def current_state(self):
        raw_obs = self.obs.astype(np.float32)
        return raw_obs, raw_obs


    def get_done_winner(self):
        """
        Returns:
            A tuple (done, winner) where:
            - done (bool): True if the episode is over (e.g., solved, invalid, or max steps reached).
            - winner (int): In single-agent tasks, you can return 0 if solved, or -1 otherwise.
        """
        is_solved = check_eqn_solved(self.lhs, self.rhs, self.main_eqn)
        if is_solved:
            return True, 0
        else:
            return False, -1

    @property
    def current_player(self):
        return 0


    @property
    def legal_actions(self):
        # Return indices where the action mask is True.
        return [i for i, valid in enumerate(self.action_mask) if valid]


    def close(self):
        """
        Clean up resources.
        """
        self._init_flag = False

    def seed(self, seed: int = 0, dynamic_seed: bool = True):
        """
        Set the random seed for the environment.
        """
        np.random.seed(seed)

    def __repr__(self):
        """
        Return a string representation of the environment.
        """
        return f"singleEqn(main_eqn={self.main_eqn}"

    @property
    def observation_space(self):
        """
        Return the observation space.
        """
        return self._observation_space

    @property
    def action_space(self):
        """
        Return the action space.
        """
        return self._action_space

    @property
    def reward_space(self):
        """
        Return the reward space.
        """
        return self._reward_space
