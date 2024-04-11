from __future__ import annotations

from abc import ABC, abstractmethod
from collections import namedtuple
import copy
import random
from typing import Any, Dict, List

from dataclasses import dataclass
import numpy as np
from ding.envs import BaseEnv
from easydict import EasyDict
from gym import spaces
from numpy.typing import NDArray

import pooltool as pt

ObservationDict = Dict[str, Any]

Bounds = namedtuple("Bounds", ["low", "high"])


@dataclass
class State:
    """
    Overview:
        A full representation of the game state.
    Attributes:
        - system (:obj:`pooltool.System`): Holds the billiard system objects (balls, \
            cue, and table) and their histories (e.g. ball trajectories). For details \
            see https://pooltool.readthedocs.io/en/latest/autoapi/pooltool/index.html#pooltool.System.
        - game (:obj:`pooltool.ruleset.Ruleset`): Holds the game status (e.g. the \
            score, whose turn it is). For details see
            https://pooltool.readthedocs.io/en/latest/autoapi/pooltool/ruleset/index.html#pooltool.ruleset.Ruleset.
    """
    system: pt.System
    game: pt.ruleset.Ruleset

    @classmethod
    def example(cls, game_type: pt.GameType = pt.GameType.SUMTOTHREE) -> State:
        """
        Overview:
            Returns an example state object.
        Arguments:
            - game_type (:obj:`pooltool.GameType`): The game type the state is built from.
        Returns:
            - state (:obj:`State`): An unsimulated state. It is the first player's turn, \
                and the balls are positioned in the game-starting configuration.
        """
        game = pt.get_ruleset(game_type)()
        game.players = [pt.Player("Player")]
        table = pt.Table.from_game_type(game_type)
        balls = pt.get_rack(
            game_type=game_type,
            table=table,
            ball_params=None,
            ballset=None,
            spacing_factor=1e-3,
        )
        cue = pt.Cue(cue_ball_id=game.shot_constraints.cueball(balls))
        system = pt.System(table=table, balls=balls, cue=cue)
        return cls(system, game)


@dataclass
class Spaces:
    """
    Overview:
        Holds definitions for the observation, action, and reward spaces.
    Attributes:
        - observation (:obj:`spaces.Space`): The observation space.
        - action (:obj:`spaces.Space`): The action space.
        - reward (:obj:`spaces.Space`): The reward space.
    """
    observation: spaces.Space
    action: spaces.Space
    reward: spaces.Space


@dataclass
class PoolToolSimulator(ABC):
    """
    Overview:
        The abstract base class that all pooltool simulators must inherit from. This \
        class has the following functionalities baked in: setting actions, running \
        simulations, and making observations. However functionality requires inheriting \
        classes to define how actions are set and what observations look like. This is \
        accomplished by implementing the methods ``self.observation_array`` and \
        ``self.set_action``, respectively.
    Attributes:
        - state (:obj:`State`): The complete billiards state. It holds the system \
            objects and the game state.
        - spaces (:obj:`Spaces`): The observation, action, and reward spaces.
    """
    state: State
    spaces: Spaces

    @abstractmethod
    def observation_array(self) -> NDArray[np.float32]:
        """
        Overview:
            The implemented method should return an observation of the current state.
        Returns:
            - observation (:obj:`NDArray[np.float32]`): A 32-bit float observation array \
                that matches the dimension and domain defined by ``self.spaces.observation``.
        """
        pass

    @abstractmethod
    def set_action(self, action: NDArray[np.float32]) -> None:
        """
        Overview:
            The implemented method should set the cue stick state by interpreting the \
            passed action. The action should match the action space defined by \
            ``self.spaces.action``.
        """
        pass

    def observation(self) -> ObservationDict:
        """
        Overview:
            Returns an observation dictionary (an observation array, an action mask, and whose turn it is)
        Returns:
            - observation_dict (:obj:`Dict[str, Any]`): Keys are ``observation``, ``action_mask``, and ``to_play``.
        """
        return dict(
            observation=self.observation_array(),
            action_mask=None,
            to_play=-1,
        )

    def scale_action(self, action: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Overview:
            Scales a normalized action (each dimension normalized from ``[-1, 1]``) to \
            the defined action space (``self.spaces.action``).
        Returns:
            - scaled_action (:obj:`NDArray[np.float32]`): The properly scaled action.
        """
        low = self.spaces.action.low  # type: ignore
        high = self.spaces.action.high  # type: ignore
        assert np.all(action >= -1) and np.all(action <= 1), f"{action=}"
        scaled_action = low + (0.5 * (action + 1.0) * (high - low))
        return np.clip(scaled_action, low, high)

    def simulate(self) -> None:
        """
        Overview:
            Simulates the system based on the (already) applied action, then updates the \
            game state to reflect the outcome of the shot (e.g. if the shot scored a \
            point, the player's point score is updated)
        Note:
            - The action should be applied before calling this method  with ``self.set_action``.
        """
        # In very (very) rare cases, pooltool can become stuck in an infinite loop of
        # event calculation. By setting ``max_events=200``, we intercept those cases and
        # end the simulation prematurely
        pt.simulate(self.state.system, inplace=True, max_events=200)
        self.state.game.process_shot(self.state.system)
        self.state.game.advance(self.state.system)

    def seed(self, seed_value: int) -> None:
        random.seed(seed_value)
        np.random.seed(seed_value)


class PoolToolEnv(BaseEnv):
    """
    Overview:
        The base pooltool environment. The purpose of this class is to hold shared \
        boilerplate code, reducing repetition.
    """
    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def manage_seeds(self) -> None:
        if (
            hasattr(self, "_seed")
            and hasattr(self, "_dynamic_seed")
            and self._dynamic_seed
        ):
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, "_seed"):
            self._env.seed(self._seed)

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> spaces.Space:
        return self._reward_space

    @classmethod
    def default_config(cls) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + "Dict"
        return cfg

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop("collector_env_num")
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop("evaluator_env_num")
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(evaluator_env_num)]
