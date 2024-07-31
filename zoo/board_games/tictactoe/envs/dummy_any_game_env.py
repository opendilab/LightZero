import copy
from typing import List

import gymnasium as gym
import numpy as np
from ding.envs.env.base_env import BaseEnvTimestep
from ding.utils.registry_factory import ENV_REGISTRY
from easydict import EasyDict

from zoo.board_games.tictactoe.envs.tictactoe_env import TicTacToeEnv


@ENV_REGISTRY.register('dummy_any_game')
class AnyGameEnv(TicTacToeEnv):
    """
    Overview:
        AnyGameEnv is a simplified test environment for both two-player non-zero-sum games and single-player games where
        each step has a reward. This environment is designed to test the functionality of MCTS algorithms.
    """

    config = dict(
        # (str): The name of the environment.
        env_id="DummyAnyGame",
        # (bool) If True, means that the game is not a zero-sum game.
        non_zero_sum=True,  # NOTE
        # (str): The mode of the battle. Choices are 'self_play_mode' or 'alpha_beta_pruning'.
        battle_mode='self_play_mode',
        # (str): The mode of Monte Carlo Tree Search. This is only used in AlphaZero.
        battle_mode_in_simulation_env='self_play_mode',
        # (str): The type of action the bot should take. Choices are 'v0' or 'alpha_beta_pruning'.
        bot_action_type='v0',
        # (str): The folder path where replay video saved, if None, will not save replay video.
        replay_path=None,
        # (bool): If True, the agent will play against a human.
        agent_vs_human=False,
        # (int): The probability of the random agent.
        prob_random_agent=0,
        # (int): The probability of the expert agent.
        prob_expert_agent=0,
        # (bool): If True, the channel will be the last dimension.
        channel_last=False,
        # (bool): If True, the pixel values will be scaled.
        scale=True,
        # (int): The value to stop the game.
        stop_value=1,
        # (bool): If True, the Monte Carlo Tree Search from AlphaZero is used.
        alphazero_mcts_ctree=False,
    )

    @classmethod
    def default_config(cls) -> EasyDict:
        """
        Overview:
            Return the default configuration for the environment.
        Returns:
            - cfg (:obj:`EasyDict`): The default configuration.
        """
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: EasyDict = None):
        """
        Overview:
            Initialize the AnyGameEnv with the given configuration.
        Arguments:
            - cfg (:obj:`EasyDict`): The configuration for the environment.
        """
        default_config = self.default_config()
        default_config.update(cfg)
        self._cfg = default_config
        self.channel_last = self._cfg.channel_last
        self.scale = self._cfg.scale
        self.non_zero_sum = self._cfg.non_zero_sum
        self.battle_mode = self._cfg.battle_mode
        assert self.battle_mode in ['self_play_mode', 'single_player_mode']
        if self.battle_mode == 'self_play_mode':
            self.battle_mode = self.battle_mode_in_simulation_env = 'self_play_mode'
            self.players = [1, 2]
        elif self.battle_mode == 'single_player_mode':
            self.battle_mode = self.battle_mode_in_simulation_env = 'single_player_mode'
            self.players = [1]

        self.board_size = 3
        self.total_num_actions = self.board_size ** 2
        self.prob_random_agent = self._cfg.prob_random_agent
        self.prob_expert_agent = self._cfg.prob_expert_agent
        assert (self.prob_random_agent >= 0 and self.prob_expert_agent == 0) or (
                self.prob_random_agent == 0 and self.prob_expert_agent >= 0), \
            f'self.prob_random_agent:{self.prob_random_agent}, self.prob_expert_agent:{self.prob_expert_agent}'
        self.agent_vs_human = self._cfg.agent_vs_human
        self.bot_action_type = self._cfg.bot_action_type
        self._replay_path = self._cfg.get('replay_path', None)
        self._save_replay_count = 0
        self.reset()

    @property
    def legal_actions(self) -> list:
        """
        Overview:
            Get the list of legal actions available in the current state.
        Returns:
            - legal_actions (:obj:`list`): A list of legal action indices.
        """
        return [i for i, x in enumerate(self.board.flatten()) if x == 0]

    def reset(self, start_player_index: int = 0, init_state: np.ndarray = None) -> dict:
        """
        Overview:
            Reset the environment to the initial state.
        Arguments:
            - start_player_index (:obj=`int`): The index of the player to start the game.
            - init_state (:obj=`np.ndarray`): The initial state of the board.
        Returns:
            - obs (:obj=`dict`): The initial observation.
        """
        self._current_player = self.players[start_player_index]
        self.board = np.zeros((self.board_size, self.board_size), dtype=int) if init_state is None else init_state

        action_mask = np.zeros(self.total_num_actions, 'int8')
        action_mask[self.legal_actions] = 1

        self.eval_episode_return_player_1 = 0
        self.eval_episode_return_player_2 = 0

        obs = {
            'observation': self.current_state()[1],
            'action_mask': action_mask,
            'board': copy.deepcopy(self.board),
            'current_player_index': start_player_index,
            'to_play': self._current_player
        }
        return obs

    def step(self, action: int) -> BaseEnvTimestep:
        """
        Overview:
            Execute the given action and return the next state, reward, done and info.
        Arguments:
            - action (:obj=`int`): The action to execute.
        Returns:
            - timestep (:obj=`BaseEnvTimestep`): The timestep containing the next state, reward, done, and info.
        """
        row, col = divmod(action, self.board_size)
        self.board[row, col] = self._current_player

        # done, winner = self.get_done_winner()
        done, winner, reward, eval_return_current_player = self.get_done_winner()

        info = {'next player to play': self.next_player}

        if self._current_player == 1:
            self.eval_episode_return_player_1 += reward
        elif self._current_player == 2:
            self.eval_episode_return_player_2 += reward

        # Exchange the player
        self.current_player = self.next_player

        action_mask = np.zeros(self.total_num_actions, 'int8')
        action_mask[self.legal_actions] = 1

        if done:
            if self._current_player == 1:
                info['eval_episode_return'] = self.eval_episode_return_player_1
            elif self._current_player == 2:
                info['eval_episode_return'] = self.eval_episode_return_player_2
            print('AnyGame one episode done! ', info)

        obs = {
            'observation': self.current_state()[1],
            'action_mask': action_mask,
            'board': copy.deepcopy(self.board),
            'current_player_index': self.players.index(self._current_player),
            'to_play': self._current_player
        }

        return BaseEnvTimestep(obs, reward, done, info)

    def get_done_winner(self):
        """
        Overview:
             Check if the game is over and who the winner is. Return 'done' and 'winner'.
        Returns:
            - outputs (:obj:`Tuple`): Tuple containing 'done' and 'winner',
                - if player 1 win,     'done' = True, 'winner' = 1
                - if player 2 win,     'done' = True, 'winner' = 2
                - if draw,             'done' = True, 'winner' = -1
                - if game is not over, 'done' = False, 'winner' = -1
        """
        done, winner = super().get_done_winner()
        reward = np.random.uniform(0, 0.01) if not done else (1 if winner == self._current_player else -1)
        print(f'AnyGame one step: self.board: {self.board}')
        print(f'current_player: {self._current_player}, done: {done},  winner: {winner},  reward: {reward}')
        if self._current_player == 1:
            self.eval_episode_return_player_1 += reward
            self.eval_return_current_player = self.eval_episode_return_player_1
        elif self._current_player == 2:
            self.eval_episode_return_player_2 += reward
            self.eval_return_current_player = self.eval_episode_return_player_2
        return done, winner, reward, self.eval_return_current_player
    @property
    def next_player(self) -> int:
        """
        Overview:
            Get the next player.
        Returns:
            - next_player (:obj=`int`): The next player.
        """
        if self.battle_mode == 'single_player_mode':
            return self.players[0]
        elif self.battle_mode == 'self_play_mode':
            return self.players[0] if self.current_player == self.players[1] else self.players[1]

    def current_state(self) -> tuple:
        """
        Overview:
            Get the current state of the board from the perspective of the current player.
        Returns:
            - raw_obs (:obj=`np.ndarray`): The raw observation.
            - scale_obs (:obj=`np.ndarray`): The scaled observation if scaling is enabled.
        """
        board_curr_player = np.where(self.board == self._current_player, 1, 0)
        board_opponent_player = np.where(self.board == self.next_player, 1, 0)
        board_to_play = np.full((self.board_size, self.board_size), self._current_player)

        assert board_curr_player.shape == board_opponent_player.shape == board_to_play.shape, \
            f"Arrays must have the same shape. Shapes are: board_curr_player: {board_curr_player.shape}, " \
            f"board_opponent_player: {board_opponent_player.shape}, board_to_play: {board_to_play.shape}"

        raw_obs = np.stack([board_curr_player, board_opponent_player, board_to_play], axis=0)
        scale_obs = raw_obs / 2 if self.scale else raw_obs

        if self.channel_last:
            return np.transpose(raw_obs, [1, 2, 0]), np.transpose(scale_obs, [1, 2, 0])
        else:
            return raw_obs, scale_obs

    def render(self, mode: str = "human"):
        """
        Overview:
            Render the current state of the board.
        Arguments:
            - mode (:obj=`str`): The mode to render with. Supports 'human' and 'rgb_array'.
        """
        if mode == "human":
            print(self.board)
        elif mode == "rgb_array":
            pass  # Implement rendering to an RGB array if needed

    @property
    def observation_space(self) -> gym.spaces.Box:
        """
        Overview:
            Get the observation space of the environment.
        Returns:
            - observation_space (:obj=`gym.spaces.Box`): The observation space.
        """
        if self.scale:
            return gym.spaces.Box(
                low=0, high=1, shape=(self.board_size, self.board_size, 3), dtype=np.float32
            )
        else:
            return gym.spaces.Box(
                low=0, high=2, shape=(self.board_size, self.board_size, 3), dtype=np.uint8
            )

    @property
    def action_space(self) -> gym.spaces.Discrete:
        """
        Overview:
            Get the action space of the environment.
        Returns:
            - action_space (:obj=`gym.spaces.Discrete`): The action space.
        """
        return gym.spaces.Discrete(self.board_size ** 2)

    @property
    def reward_space(self) -> gym.spaces.Box:
        """
        Overview:
            Get the reward space of the environment.
        Returns:
            - reward_space (:obj=`gym.spaces.Box`): The reward space.
        """
        return gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    @property
    def current_player(self) -> int:
        """
        Overview:
            Get the current player.
        Returns:
            - current_player (:obj=`int`): The current player.
        """
        return self._current_player

    @current_player.setter
    def current_player(self, value: int):
        """
        Overview:
            Set the current player.
        Arguments:
            - value (:obj=`int`): The player to set as the current player.
        """
        self._current_player = value

    def clone(self) -> 'AnyGameEnv':
        """
        Overview:
            Clone the environment.
        Returns:
            - clone (:obj=`AnyGameEnv`): A deep copy of the environment.
        """
        return copy.deepcopy(self)

    def seed(self, seed: int, dynamic_seed: bool = True):
        """
        Overview:
            Seed the environment's random number generator.
        Arguments:
            - seed (:obj=`int`): The seed value.
            - dynamic_seed (:obj=`bool`): Whether to dynamically change the seed.
        """
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def close(self):
        """
        Overview:
            Close the environment.
        """
        pass

    def __repr__(self) -> str:
        """
        Overview:
            Get a string representation of the environment.
        Returns:
            - repr (:obj=`str`): The string representation.
        """
        return "AnyGameEnv"

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(evaluator_env_num)]


# Configure environment
cfg = EasyDict(AnyGameEnv.default_config())
env = AnyGameEnv(cfg)

# Reset environment
obs = env.reset()

# Execute an action
timestep = env.step(action=0)
print(timestep)