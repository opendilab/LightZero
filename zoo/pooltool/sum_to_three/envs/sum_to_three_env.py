from __future__ import annotations

import gc
from typing import Any, Dict

import attrs
import numpy as np
from ding.envs import BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from gym import spaces
from numpy.typing import NDArray
from zoo.pooltool.datatypes import (
    ObservationDict,
    PoolToolEnv,
    PoolToolGym,
    Spaces,
    State,
)

import pooltool as pt
import pooltool.constants as const


def calc_reward(state: State) -> float:
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


BALL_DIM = 2


@attrs.define
class SumToThreeGym(PoolToolGym):
    def _slice(self, ball_idx: int) -> slice:
        return slice(ball_idx * BALL_DIM, (ball_idx + 1) * BALL_DIM)

    def _null_obs(self) -> NDArray[np.float32]:
        return np.empty(len(self.system.balls) * BALL_DIM, dtype=np.float32)

    def set_action(self, scaled_action: NDArray[np.float32]) -> None:
        self.system.cue.set_state(
            V0=scaled_action[0],
            phi=pt.aim.at_ball(self.system, "object", cut=scaled_action[1]),
        )

    def observation_array(self) -> NDArray[np.float32]:
        """Return the system state as a 1D array of ball coordinates"""
        obs = self._null_obs()
        for ball_idx, ball_id in enumerate(self.system.balls.keys()):
            obs[self._slice(ball_idx)] = self.system.balls[ball_id].state.rvw[
                0, :BALL_DIM
            ]

        return obs

    def set_observation(self, obs: NDArray[np.float32]) -> None:
        """Set the system state from an observation array"""
        for ball_idx, ball_id in enumerate(self.system.balls.keys()):
            self.system.balls[ball_id].state.rvw[0, :BALL_DIM] = obs[
                self._slice(ball_idx)
            ]

    @staticmethod
    def get_obs_space(balls: Dict[str, pt.Ball], table: pt.Table) -> Any:
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

    def reset(self) -> None:
        if len(self.game.players) == 1:
            self.reset_single_player_env()
        else:
            raise NotImplementedError()

    def reset_single_player_env(self) -> None:
        """Reset things to an initial state"""
        del self.game
        self.game = pt.get_ruleset(pt.GameType.SUMTOTHREE)(
            players=[pt.Player("Player 1")],
            win_condition=-1,  # type: ignore
        )

        R = self.system.balls["cue"].params.R

        cue_pos = (
            self.system.table.w / 2,
            self.system.table.l / 4,
            R,
        )

        object_pos = (
            self.system.table.w / 2,
            self.system.table.l * 3 / 4,
            R,
        )

        self.system.reset_history()
        self.system.stop_balls()

        self.system.balls["cue"].state.rvw[0] = cue_pos
        self.system.balls["object"].state.rvw[0] = object_pos

        assert self.system.balls["cue"].state.s == const.stationary
        assert self.system.balls["object"].state.s == const.stationary
        assert not np.isnan(self.system.balls["cue"].state.rvw).any()
        assert not np.isnan(self.system.balls["object"].state.rvw).any()

    @classmethod
    def from_state(cls, state: State) -> SumToThreeGym:
        """Create a SumToThree environment from a State"""
        return cls(
            system=state.system,
            game=state.game,
            spaces=Spaces(
                observation=cls.get_obs_space(
                    state.system.balls,
                    state.system.table,
                ),
                action=spaces.Box(
                    low=np.array([0.3, -70], dtype=np.float32),
                    high=np.array([3.0, +70], dtype=np.float32),
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

    @classmethod
    def single_player_env(cls, random_pos: bool = False) -> SumToThreeGym:
        """Create a 1 player environment (for training, evaluation, etc)"""
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

        return cls.from_state(State(system, game))


@attrs.define
class EpisodicTrackedStats:
    eval_episode_length: int = 0
    eval_episode_return: float = 0.0


@ENV_REGISTRY.register("pooltool_sumtothree")
class SumToThreeEnv(PoolToolEnv):
    config = dict(
        env_name="PoolTool-SumToThree",
        env_type="not_board_games",
        episode_length=10,
    )

    def __init__(self, cfg: EasyDict) -> None:
        self.cfg = cfg
        self._init_flag = False
        self._tracked_stats = EpisodicTrackedStats()

        self._env: SumToThreeGym

    def __repr__(self) -> str:
        return "SumToThreeEnv"

    def close(self) -> None:
        # Probably not necessary
        for ball in self._env.system.balls.values():
            del ball.state
            del ball.history
            del ball.history_cts
            del ball
        for pocket in self._env.system.table.pockets.values():
            del pocket
        for cushion in self._env.system.table.cushion_segments.linear.values():
            del cushion
        for cushion in self._env.system.table.cushion_segments.circular.values():
            del cushion
        del self._env.system.table
        del self._env.system.cue
        del self._env.system
        del self._env.game
        del self._env
        gc.collect()

        self._init_flag = False

    def reset(self) -> ObservationDict:
        if not self._init_flag:
            self._env = SumToThreeGym.single_player_env()
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

        rew = calc_reward(self._env)

        self._tracked_stats.eval_episode_length += 1
        self._tracked_stats.eval_episode_return += rew

        done = self._tracked_stats.eval_episode_length == self.cfg["episode_length"]

        info = attrs.asdict(self._tracked_stats) if done else {}

        return BaseEnvTimestep(
            obs=self._env.observation(),
            reward=np.array([rew], dtype=np.float32),
            done=done,
            info=info,
        )
