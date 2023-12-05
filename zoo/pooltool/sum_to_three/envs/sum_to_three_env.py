from __future__ import annotations

import copy
import random
from functools import cached_property
from typing import Any, Callable, Dict, List

import attrs
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from gym import spaces
from numpy.typing import NDArray

import pooltool as pt
from pooltool.ai.datatypes import State as PoolToolState


def _reward(state: PoolToolState) -> float:
    """Calculate the reward

    A point is scored when both:

        (1) The player contacts the object ball with the cue ball
        (2) The sum of contacted rails by either balls is 3

    This reward is designed to offer a small reward for contacting the object ball, and
    a larger reward for scoring a point.
    """
    if not state.game.shot_info.turn_over:
        return 1.0

    if len(pt.filter_type(state.system.events, pt.EventType.BALL_BALL)):
        return 0.1

    return 0.0


@attrs.define
class Rewarder:
    calc: Callable[[PoolToolState], float]
    space: spaces.Space

    @classmethod
    def default(cls) -> Rewarder:
        return cls(
            calc=_reward,
            space=spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32,
            ),
        )


def scale_action(action: NDArray[np.float32], low: float, high: float):
    """Scale the action from [-1, 1] to the given range [low, high]"""
    assert np.all(action >= -1) and np.all(action <= 1), f"{action=}"

    scaled_action = low + (0.5 * (action + 1.0) * (high - low))
    return np.clip(scaled_action, low, high)


class SimulationEnv(PoolToolState):
    BALL_DIM = 2

    @cached_property
    def _id_map(self) -> Dict[str, slice]:
        return {ball_id: self._slice(i) for i, ball_id in enumerate(self._ids)}

    @cached_property
    def _ids(self) -> List[str]:
        return list(self.system.balls.keys())

    def _slice(self, ball_idx: int) -> slice:
        return slice(ball_idx * self.BALL_DIM, (ball_idx + 1) * self.BALL_DIM)

    def _null_obs(self) -> NDArray[np.float32]:
        return np.empty(len(self._ids) * self.BALL_DIM, dtype=np.float32)

    def _null_action(self) -> NDArray[np.float32]:
        return np.empty(2, dtype=np.float32)

    def observation_array(self) -> NDArray[np.float32]:
        """Return the system state as a 1D array of ball coordinates"""
        obs = self._null_obs()
        for ball_id in self._ids:
            obs[self._id_map[ball_id]] = self.system.balls[ball_id].state.rvw[0, :self.BALL_DIM]

        return obs

    def set_observation(self, obs: NDArray[np.float32]) -> None:
        """Set the system state from an observation array"""
        for ball_id in self._ids:
            self.system.balls[ball_id].state.rvw[0, :self.BALL_DIM] = obs[self._id_map[ball_id]]

    def set_action(self, action: NDArray[np.float32]) -> None:
        """Set the cue parameters from an action array"""
        self.system.cue.set_state(
            V0=action[0],
            phi=pt.aim.at_ball(self.system, "object", cut=action[1]),
        )

    def simulate(self) -> None:
        """Simulate the system"""
        pt.simulate(self.system, inplace=True, max_events=200)
        self.game.process_shot(self.system)
        self.game.advance(self.system)

    def seed(self, seed_value: int) -> None:
        random.seed(seed_value)
        np.random.seed(seed_value)

    def obs_space(self) -> spaces.Space:
        """Return observation, action, and reward spaces"""
        table_length = self.system.table.l
        table_width = self.system.table.l
        ball_radius = self.system.balls["cue"].params.R

        xmin, ymin = ball_radius, ball_radius
        xmax, ymax = table_width - ball_radius, table_length - ball_radius

        num_balls = len(self._ids)

        return spaces.Box(
            low=np.array([xmin, ymin] * num_balls, dtype=np.float32),
            high=np.array([xmax, ymax] * num_balls, dtype=np.float32),
            shape=(self.BALL_DIM * num_balls,),
            dtype=np.float32,
        )

    @classmethod
    def sum_to_three(cls) -> SimulationEnv:
        # Setting win_condition to negative means the game never reaches a terminal
        # state. This is what we want, since we are defining a fixed episode length
        gametype = pt.GameType.SUMTOTHREE
        return SimulationEnv(
            system=pt.System(
                cue=pt.Cue.default(),
                table=(table := pt.Table.from_game_type(gametype)),
                balls=pt.get_rack(gametype, table),
            ),
            game=pt.get_ruleset(gametype)(
                players=[pt.Player("Player 1")],
                win_condition=-1,  # type: ignore
            ),
        )


@attrs.define
class EpisodicTrackedStats:
    eval_episode_length: int = 0
    eval_episode_return: float = 0.0


ObservationDict = Dict[str, Any]

REWARDER = Rewarder.default()
OBS_SPACE = SimulationEnv.sum_to_three().obs_space()
ACTION_SPACE = spaces.Box(
    low=np.array([0.5, -70], dtype=np.float32),
    high=np.array([3, +70], dtype=np.float32),
    shape=(2,),
    dtype=np.float32,
)


@ENV_REGISTRY.register("pooltool_sumtothree")
class SumToThreeEnv(BaseEnv):
    config = dict(
        env_name="PoolTool-SumToThree",
        env_type="not_board_games",
        episode_length=10,
    )

    def __init__(self, cfg: EasyDict) -> None:
        self.cfg = cfg
        self._observation_space = OBS_SPACE
        self._action_space = ACTION_SPACE
        self._reward_space = REWARDER.space

        self._tracked_stats = EpisodicTrackedStats()

    def __repr__(self) -> str:
        return "SumToThreeEnv"

    def close(self) -> None:
        raise NotImplementedError()

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def reset(self) -> ObservationDict:
        self._env = SimulationEnv.sum_to_three()

        if (
            hasattr(self, "_seed")
            and hasattr(self, "_dynamic_seed")
            and self._dynamic_seed
        ):
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, "_seed"):
            self._env.seed(self._seed)

        self._tracked_stats = EpisodicTrackedStats()
        return self.observation()

    def observation(self) -> ObservationDict:
        return dict(
            observation=self._env.observation_array(),
            action_mask=None,
            to_play=-1,
        )

    def step(self, action: NDArray[np.float32]) -> BaseEnvTimestep:
        scaled_action = scale_action(
            action,
            self.action_space.low,
            self.action_space.high,
        )
        self._env.set_action(scaled_action)
        self._env.simulate()

        rew = REWARDER.calc(self._env)

        self._tracked_stats.eval_episode_length += 1
        self._tracked_stats.eval_episode_return += rew

        done = self._tracked_stats.eval_episode_length == self.cfg["episode_length"]
        if done:
            print(self._tracked_stats)

        info = attrs.asdict(self._tracked_stats) if done else {}

        return BaseEnvTimestep(
            obs=self.observation(),
            reward=np.array([rew], dtype=np.float32),
            done=done,
            info=info,
        )

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
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(evaluator_env_num)]

