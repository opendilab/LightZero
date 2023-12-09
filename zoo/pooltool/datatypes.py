"""This would be a great place to create PoolToolEnv and observation/action vector building"""
from __future__ import annotations

import copy
import random
from functools import cached_property
from typing import Any, Dict, List, Optional

import attrs
import numpy as np
from ding.envs import BaseEnv
from easydict import EasyDict
from gym import spaces
from numpy.typing import NDArray

import pooltool as pt
from pooltool.ai.datatypes import State as PoolToolState

ObservationDict = Dict[str, Any]

BALL_DIM = 2


@attrs.define
class Spaces:
    observation: spaces.Space
    action: spaces.Space
    reward: spaces.Space


@attrs.define(slots=False)
class SimulationEnv(PoolToolState):
    spaces: Spaces

    @cached_property
    def _id_map(self) -> Dict[str, slice]:
        return {ball_id: self._slice(i) for i, ball_id in enumerate(self._ids)}

    @cached_property
    def _ids(self) -> List[str]:
        return list(self.system.balls.keys())

    def _slice(self, ball_idx: int) -> slice:
        return slice(ball_idx * BALL_DIM, (ball_idx + 1) * BALL_DIM)

    def _null_obs(self) -> NDArray[np.float32]:
        return np.empty(len(self._ids) * BALL_DIM, dtype=np.float32)

    def _null_action(self) -> NDArray[np.float32]:
        return np.empty(2, dtype=np.float32)

    def observation(self) -> ObservationDict:
        return dict(
            observation=self.observation_array(),
            action_mask=None,
            to_play=-1,
        )

    def observation_array(self) -> NDArray[np.float32]:
        """Return the system state as a 1D array of ball coordinates"""
        obs = self._null_obs()
        for ball_id in self._ids:
            obs[self._id_map[ball_id]] = self.system.balls[ball_id].state.rvw[
                0, :BALL_DIM
            ]

        return obs

    def set_observation(self, obs: NDArray[np.float32]) -> None:
        """Set the system state from an observation array"""
        for ball_id in self._ids:
            self.system.balls[ball_id].state.rvw[0, :BALL_DIM] = obs[
                self._id_map[ball_id]
            ]

    def scale_action(self, action: NDArray[np.float32]) -> NDArray[np.float32]:
        """Scale the action from [-1, 1] to the given range [low, high]"""
        low = self.spaces.action.low  # type: ignore
        high = self.spaces.action.high  # type: ignore
        assert np.all(action >= -1) and np.all(action <= 1), f"{action=}"
        scaled_action = low + (0.5 * (action + 1.0) * (high - low))
        return np.clip(scaled_action, low, high)

    def set_action(self, scaled_action: NDArray[np.float32]) -> None:
        """Set the cue parameters from an action array"""
        self.system.cue.set_state(
            V0=scaled_action[0],
            phi=pt.aim.at_ball(self.system, "object", cut=scaled_action[1]),
        )

    def simulate(self) -> None:
        """Simulate the system"""
        pt.simulate(self.system, inplace=True, max_events=200)
        self.game.process_shot(self.system)
        self.game.advance(self.system)

    def seed(self, seed_value: int) -> None:
        random.seed(seed_value)
        np.random.seed(seed_value)

    @staticmethod
    def get_obs_space(balls: Dict[str, pt.Ball], table: pt.Table) -> spaces.Space:
        """Return observation, action, and reward spaces"""
        table_length = table.l
        table_width = table.l
        ball_radius = balls["cue"].params.R

        xmin, ymin = ball_radius, ball_radius
        xmax, ymax = table_width - ball_radius, table_length - ball_radius

        return spaces.Box(
            low=np.array([xmin, ymin] * len(balls), dtype=np.float32),
            high=np.array([xmax, ymax] * len(balls), dtype=np.float32),
            shape=(BALL_DIM * len(balls),),
            dtype=np.float32,
        )

    @classmethod
    def sum_to_three(
        cls,
        state: Optional[PoolToolState] = None,
        random_pos: bool = False,
    ) -> SimulationEnv:
        """Create a SumToThree environment, either fresh or from a state"""

        if state is None:
            # Setting win_condition to negative means the game never reaches a terminal
            # state. This is what we want, since we are defining a fixed episode length
            gametype = pt.GameType.SUMTOTHREE
            game = pt.get_ruleset(gametype)(
                players=[pt.Player("Player 1")],
                win_condition=-1,  # type: ignore
            )
            system = pt.System(
                cue=pt.Cue.default(),
                table=(table := pt.Table.from_game_type(gametype)),
                balls=pt.get_rack(gametype, table),
            )

            if random_pos:
                get_pos = lambda table, ball: (
                    (table.w - 2 * ball.params.R) * np.random.rand() + ball.params.R,
                    (table.l - 2 * ball.params.R) * np.random.rand() + ball.params.R,
                    ball.params.R,
                )
                system.balls["cue"].state.rvw[0] = get_pos(
                    system.table, system.balls["cue"]
                )
                system.balls["object"].state.rvw[0] = get_pos(
                    system.table, system.balls["object"]
                )

            state = PoolToolState(system, game)

        return SimulationEnv(
            system=state.system,
            game=state.game,
            spaces=Spaces(
                observation=cls.get_obs_space(state.system.balls, state.system.table),
                action=spaces.Box(
                    low=np.array([0.5, -70], dtype=np.float32),
                    high=np.array([3, +70], dtype=np.float32),
                    shape=(2,),
                    dtype=np.float32,
                ),
                reward=spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
            ),
        )


class BasePoolToolEnv(BaseEnv):
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
