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
    - observation (:obj:`array`): An nd-array with 5 feature planes. Each feature plane
        is a grayscale (black/white) image of the table, with different features being
        shown. The first plane is the cue ball, the second plane is the object ball, the
        third plane is the object ball and the cue ball, the fourth plane is a line
        drawn between the object ball and cue ball, and the fifth plane renders the
        rails as straight lines. To visualize these planes, see the example in
        ../../image_representation.py. The shape of this array is in general
        ``(n_features, px, px//2)``, where ``n_features`` is the number of feature
        planes, ``px`` is the number of pixels representing the length of the table, and
        ``px//2`` is the width of the table..
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
import numpy as np
from ding.envs import BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from gym import spaces
from numpy.typing import NDArray
from zoo.pooltool.sum_to_three.reward import get_reward_bounds, get_reward_function
from zoo.pooltool.datatypes import (
    Bounds,
    ObservationDict,
    PoolToolEnv,
    PoolToolSimulator,
    Spaces,
    State,
)
from zoo.pooltool.image_representation import PygameRenderer, RenderConfig

import pooltool as pt

def get_obs_space(renderer: PygameRenderer) -> spaces.Box:
    """
    Overview:
        Generate the observation space based on the renderer's configuration.
    Arguments:
        - renderer (:obj:`PygameRenderer`): Renderer object which contains configuration details including dimensions and planes.
    Returns:
        - space (:obj:`spaces.Box`): The observation space defining the bounds and shape based on the renderer's output.
    """
    channels = len(renderer.render_config.planes)
    return spaces.Box(
        low=0.0,
        high=1.0,
        shape=(channels, renderer.coordinates.height, renderer.coordinates.width),
        dtype=np.float32,
    )

def get_action_space(V0: Bounds, angle: Bounds) -> spaces.Box:
    """
    Overview:
        Create an action space from specified bounds for the velocity and angle.
    Arguments:
        - V0 (:obj:`Bounds`): The bounds for the velocity of the cue.
        - angle (:obj:`Bounds`): The bounds for the angle of the cue.
    Returns:
        - space (:obj:`spaces.Box`): The action space for the environment.
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

@dataclass
class SumToThreeImageSimulator(PoolToolSimulator):
    renderer: PygameRenderer

    def set_action(self, action: NDArray[np.float32]) -> None:
        self.state.system.cue.set_state(
            V0=action[0],
            phi=pt.aim.at_ball(self.state.system, "object", cut=action[1]),
        )

    def observation_array(self) -> NDArray[np.float32]:
        """Return the system state as an image array"""
        return self.renderer.observation()

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

        assert self.state.system.balls["cue"].state.s == pt.constants.stationary
        assert self.state.system.balls["object"].state.s == pt.constants.stationary
        assert not np.isnan(self.state.system.balls["cue"].state.rvw).any()
        assert not np.isnan(self.state.system.balls["object"].state.rvw).any()


@dataclass
class EpisodicTrackedStats:
    eval_episode_length: int = 0
    eval_episode_return: float = 0.0


@ENV_REGISTRY.register("pooltool_sumtothree_image")
class SumToThreeImageEnv(PoolToolEnv):
    config = dict(
        env_name="PoolTool-SumToThree-Image",
        env_type="not_board_games",
        episode_length=10,
        reward_algorithm="binary",
        action_V0_low=0.3,
        action_V0_high=3.0,
        action_angle_low=-70,
        action_angle_high=70,
    )

    def __repr__(self) -> str:
        return "SumToThreeEnvImage"

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
        self._env: SumToThreeImageSimulator

    def close(self) -> None:
        self._env.renderer.close()

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

    def get_spaces(self, renderer: PygameRenderer) -> Spaces:
        return Spaces(
            observation=get_obs_space(
                renderer,
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
            renderer = PygameRenderer.build(
                state.system.table,
                RenderConfig.from_json(self.cfg.render_config_path)
            )
            renderer.set_state(state)
            renderer.init()
            self._env = SumToThreeImageSimulator(state, self.get_spaces(renderer), renderer=renderer)
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
