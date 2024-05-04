"""
Overview:
    Implementation of a learning environment for the simple billiards game, sum-to-three.
    The game is played on a table with no pockets.
    There are 2 balls: a cue ball and an object ball
    The player must hit the object ball with the cue ball
    The player scores a point if the number of times a ball hits a cushion is 3
Mode:
    - ``self_play_mode``: In ``self_play_mode``, there is only one player, who takes 10 \
        shots. Their final score is the number of points they achieve.
    - ``play_with_bot_mode``: (**NOT YET IMPLEMENTED**) In ``play_with_bot_mode`` there are two players, \
        a learning agent and a bot. The game ends when either player achieves 5 points.
Bot:
    - MCTSBot: (**NOT YET IMPLEMENTED**) A bot which take action through a Monte Carlo Tree Search, which \
        has a high performance.
    - RuleBot: (**NOT YET IMPLEMENTED**) A bot which takes actions according to some simple heuristics, \
        which has a low performance.
Observation Space:
    The observation in this environment is a dictionary with three elements.
    - observation (:obj:`array`): A continuous 1D array holding the x- and y- coordinates of the cue ball \
        and the object ball. It has the following entries: ``[x_cue, y_cue, x_obj, y_obj]``. ``x_cue`` and \
        ``y_cue`` are the 2D coordinates of the cue ball, and ``x_obj`` and ``y_obj`` are 2D coordinates of \
        the object ball. x-coordinates can be between ``R`` and ``w-R``, where ``R`` is the ball radius and \
        ``w`` is the width of the table. Similarly, y-coordinates can be between ``R`` and ``l-R``, where \
        ``l`` is the length of the table.
    - action_mask (:obj:`None`): No actions are masked, so ``None`` is used here.
    - to_play (:obj:`None`): (**NOT YET IMPLEMENTED**) For ``self_play_mode``, this is
        set to -1. For ``play_with_bot_mode``, this indicates the player that needs to take an action in the \
        current state.
Action Space:
    A continuous length-2 array. The first element is ``V0``, the speed of the cue stick. The second element \
    is the ``cut_angle``, which is the angle that the cue ball hits the object ball with. A cut angle of 0 is \
    a head-on collision, a cut angle of -89 is a very slight graze on the left side of the object ball, and a \
    cut angle of 89 is a very slight graze on the right side of the object ball.
Reward Space:
    For the ``self_play_mode``, intermediate rewards of 1.0 are returned for each step where the player earns a point.
    For the ``play_with_bot_mode``, (**NOT YET IMPLEMENTED**)...
"""

from __future__ import annotations

import gc

from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
from ding.envs import BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from gym import spaces
from numpy.typing import NDArray
from zoo.pooltool.datatypes import (
    Bounds,
    ObservationDict,
    PoolToolEnv,
    PoolToolSimulator,
    Spaces,
    State,
)

import pooltool as pt
from zoo.pooltool.image_representation import PygameRenderer
from zoo.pooltool.sum_to_three.reward import get_reward_bounds, get_reward_function


def get_obs_space(system: pt.System) -> spaces.Box:
    """
    Overview:
        Given a billiards system, return the observation space.
    """
    table_length = system.table.l
    table_width = system.table.w
    ball_radius = system.balls["cue"].params.R

    xmin, ymin = ball_radius, ball_radius
    xmax, ymax = table_width - ball_radius, table_length - ball_radius

    return spaces.Box(
        low=np.array([xmin, ymin] * len(system.balls), dtype=np.float32),
        high=np.array([xmax, ymax] * len(system.balls), dtype=np.float32),
        shape=(BALL_DIM * len(system.balls),),
        dtype=np.float32,
    )


def get_action_space(V0: Bounds, angle: Bounds) -> spaces.Box:
    """
    Overview:
        Given the action bounds, return the action space.
    """
    return spaces.Box(
        low=np.array([V0.low, angle.low], dtype=np.float32),
        high=np.array([V0.high, angle.high], dtype=np.float32),
        shape=(2,),
        dtype=np.float32,
    )


def get_reward_space(algorithm: str) -> spaces.Box:
    """
    Overview:
        Determines the reward space based on the reward calculation algorithm.
    Arguments:
        - algorithm (:obj:`str`): The name of the algorithm used to calculate rewards.
    Returns:
        - space (:obj:`spaces.Box`): The reward space for the environment.
    """
    bounds = get_reward_bounds(algorithm)  # Assumes a function get_reward_bounds exists.
    return spaces.Box(
        low=np.array([bounds.low], dtype=np.float32),
        high=np.array([bounds.high], dtype=np.float32),
        shape=(1,),
        dtype=np.float32,
    )


def create_initial_state(random_pos: bool) -> State:
    """
    Overview:
        Creates a ready-to-play state.
    Arguments:
        - random_pos: If ``False``, initial ball positions are set to the starting \
            configuration of the game (with the cue ball on one side of the table and the \
            object ball on the other side). If ``True``, the ball positions are randomized.
    Returns:
        - state (:obj:`State`): The ready-to-play state. The game is setup to be single \
            player with perpetual play (no win condition). The cue stick parameters have not \
            yet been set.
    """
    gametype = pt.GameType.SUMTOTHREE
    players = [pt.Player("Player 1")]

    game = pt.get_ruleset(gametype)(
        players=players,
        win_condition=-1,  # type: ignore
    )

    system = pt.System(
        cue=pt.Cue.default(),
        table=(table := pt.Table.from_game_type(gametype)),
        balls=pt.get_rack(gametype, table),
    )

    system.cue.set_state(V0=0.0)

    if random_pos:
        get_pos = lambda table, ball: (
            (table.w - 2 * ball.params.R) * np.random.rand() + ball.params.R,
            (table.l - 2 * ball.params.R) * np.random.rand() + ball.params.R,
            ball.params.R,
        )
        system.balls["cue"].state.rvw[0] = get_pos(system.table, system.balls["cue"])
        system.balls["object"].state.rvw[0] = get_pos(
            system.table, system.balls["object"]
        )

    return State(system, game)


BALL_DIM = 2

def _slice(ball_idx: int) -> slice:
    return slice(ball_idx * BALL_DIM, (ball_idx + 1) * BALL_DIM)

def _null_obs(num_balls: int) -> NDArray[np.float32]:
    return np.empty(num_balls * BALL_DIM, dtype=np.float32)

def vector_coordinate_observation_array(state: State) -> NDArray[np.float32]:
    """
    Overview:
        Returns an observation array of the current state.
    Returns:
        - observation (:obj:`NDArray[np.float32]`): A continuous 1D array holding \
            the x- and y- coordinates of the cue ball and the object ball. It has \
            the following entries: ``[x_cue, y_cue, x_obj, y_obj]``. \
            ``x_cue`` and ``y_cue`` are the 2D coordinates of the cue ball, and \
            ``x_obj`` and ``y_obj`` are 2D coordinates of the object ball. The \
            coordinate system is defined by ``self.spaces.observation`` (see \
            :obj:`get_obs_space`).
    """
    obs = _null_obs(len(state.system.balls))
    for ball_idx, ball_id in enumerate(state.system.balls.keys()):
        coords = state.system.balls[ball_id].state.rvw[0, :BALL_DIM]
        obs[_slice(ball_idx)] = coords

    return obs


@dataclass
class SumToThreeSimulator(PoolToolSimulator):
    """
    Overview:
        Manages the simulation state for simulating actions and retrieving subsequent \
        observations.
    """
    renderer: Optional[PygameRenderer] = None

    def set_action(self, action: NDArray[np.float32]) -> None:
        """
        Overview:
            Sets the cue stick state for a 2-parameter action.
        Arguments:
            - action (:obj:`NDArray[np.float32]`): A length-2 array, where the first \
                parameter is the speed of the cue stick (in m/s), and the second is \
                the cut angle (i.e., the angle that the cue ball hits the object \
                ball with) (in degrees). Spin and strike elevation are set to 0.
        """
        self.state.system.cue.set_state(
            V0=action[0],
            phi=pt.aim.at_ball(self.state.system, "object", cut=action[1]),
            theta=0.0,
            a=0.0,
            b=0.0,
        )

    def observation_array(self) -> NDArray[np.float32]:
        """
        Overview:
            Returns an observation array of the current state.
        Returns:
            - observation (:obj:`NDArray[np.float32]`): The observation array. For
                details, see the docstrings of the delegate functions.
        """
        if True:
            return vector_coordinate_observation_array(self.state)

    def reset(self) -> None:
        if len(self.state.game.players) == 1:
            self.reset_single_player_env()
        else:
            raise NotImplementedError()

    def reset_single_player_env(self) -> None:
        """Return the passed environment, resetting things to an initial state"""
        del self.state.game
        self.state.game = pt.get_ruleset(pt.GameType.SUMTOTHREE)(
            players=[pt.Player("Player 1")],
            win_condition=-1,  # type: ignore
        )

        R = self.state.system.balls["cue"].params.R

        cue_pos = (
            self.state.system.table.w / 2,
            self.state.system.table.l / 4,
            R,
        )

        object_pos = (
            self.state.system.table.w / 2,
            self.state.system.table.l * 3 / 4,
            R,
        )

        self.state.system.reset_history()
        self.state.system.stop_balls()

        self.state.system.balls["cue"].state.rvw[0] = cue_pos
        self.state.system.balls["object"].state.rvw[0] = object_pos

        self.state.system.cue.set_state(V0=0.0)

        assert self.state.system.balls["cue"].state.s == pt.constants.stationary
        assert self.state.system.balls["object"].state.s == pt.constants.stationary
        assert not np.isnan(self.state.system.balls["cue"].state.rvw).any()
        assert not np.isnan(self.state.system.balls["object"].state.rvw).any()


@dataclass
class EpisodicTrackedStats:
    eval_episode_length: int = 0
    eval_episode_return: float = 0.0


@ENV_REGISTRY.register("pooltool_sumtothree")
class SumToThreeEnv(PoolToolEnv):
    config = dict(
        env_name="PoolTool-SumToThree",
        env_type="not_board_games",
        episode_length=10,
        reward_algorithm="binary",
        action_V0_low=0.3,
        action_V0_high=3.0,
        action_angle_low=-70,
        action_angle_high=70,
        observation_space="coordinate",
    )

    def __repr__(self) -> str:
        return "SumToThreeEnv"

    def __init__(self, cfg: EasyDict) -> None:
        self.cfg = cfg

        # Get the reward function
        self.calc_reward = get_reward_function(self.cfg.reward_algorithm)

        # Structure the action bounds
        self.action_bounds = {
            "V0": Bounds(
                low=self.cfg.action_V0_low,
                high=self.cfg.action_V0_high,
            ),
            "angle": Bounds(
                low=self.cfg.action_angle_low,
                high=self.cfg.action_angle_high,
            ),
        }

        self._init_flag = False
        self._tracked_stats = EpisodicTrackedStats()
        self._env: SumToThreeSimulator

    def close(self) -> None:
        # Probably not necessary
        for ball in self._env.state.system.balls.values():
            del ball.state
            del ball.history
            del ball.history_cts
            del ball
        for pocket in self._env.state.system.table.pockets.values():
            del pocket
        for cushion in self._env.state.system.table.cushion_segments.linear.values():
            del cushion
        for cushion in self._env.state.system.table.cushion_segments.circular.values():
            del cushion
        del self._env.state.system.table
        del self._env.state.system.cue
        del self._env.state.system
        del self._env.state.game
        del self._env
        gc.collect()

        self._init_flag = False

    def get_spaces(self, state: State) -> Spaces:
        return Spaces(
            observation=get_obs_space(
                state.system,
            ),
            action=get_action_space(
                self.action_bounds["V0"],
                self.action_bounds["angle"],
            ),
            reward=get_reward_space(
                self.cfg.reward_algorithm,
            ),
        )

    def reset(self) -> ObservationDict:
        if not self._init_flag:
            state = create_initial_state(random_pos=False)
            self._env = SumToThreeSimulator(state, self.get_spaces(state))
            self._init_flag = True
        else:
            self._env.reset()

        self.manage_seeds()
        self._tracked_stats = EpisodicTrackedStats()

        self._observation_space = self._env.spaces.observation
        self._action_space = self._env.spaces.action
        self._reward_space = self._env.spaces.reward

        return self._env.observation()

    def step(self, action: NDArray[np.float32]) -> BaseEnvTimestep:
        self._env.set_action(self._env.scale_action(action))
        self._env.simulate()

        rew = self.calc_reward(self._env.state)

        self._tracked_stats.eval_episode_length += 1
        self._tracked_stats.eval_episode_return += rew

        done = self._tracked_stats.eval_episode_length == self.cfg.episode_length
        info = asdict(self._tracked_stats) if done else {}

        return BaseEnvTimestep(
            obs=self._env.observation(),
            reward=np.array([rew], dtype=np.float32),
            done=done,
            info=info,
        )
