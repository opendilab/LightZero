from __future__ import annotations

import gc
from typing import Dict

from dataclasses import dataclass, asdict
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
from zoo.pooltool.sum_to_three.reward import get_reward_function


V0_BOUNDS = Bounds(0.3, 3.0)
ANGLE_BOUNDS = Bounds(-70, 70)
V0_INIT = 0.0
BALL_DIM = 2

    
def get_obs_space(balls: Dict[str, pt.Ball], table: pt.Table) -> spaces.Box:
    table_length = table.l
    table_width = table.w
    ball_radius = balls["cue"].params.R

    xmin, ymin = ball_radius, ball_radius
    xmax, ymax = table_width - ball_radius, table_length - ball_radius

    return spaces.Box(
        low=np.array([xmin, ymin] * len(balls), dtype=np.float32),
        high=np.array([xmax, ymax] * len(balls), dtype=np.float32),
        shape=(BALL_DIM * len(balls),),
        dtype=np.float32,
    )


def get_action_space(
    V0: Bounds = V0_BOUNDS, angle: Bounds = ANGLE_BOUNDS
) -> spaces.Box:
    return spaces.Box(
        low=np.array([V0.low, angle.low], dtype=np.float32),
        high=np.array([V0.high, angle.high], dtype=np.float32),
        shape=(2,),
        dtype=np.float32,
    )


def create_initial_state(random_pos: bool) -> State:
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

    system.cue.set_state(V0=V0_INIT)

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

    return State(system, game)


@dataclass
class SumToThreeSimulator(PoolToolSimulator):
    def _slice(self, ball_idx: int) -> slice:
        return slice(ball_idx * BALL_DIM, (ball_idx + 1) * BALL_DIM)

    def _null_obs(self) -> NDArray[np.float32]:
        return np.empty(len(self.state.system.balls) * BALL_DIM, dtype=np.float32)

    def set_action(self, action: NDArray[np.float32]) -> None:
        self.state.system.cue.set_state(
            V0=action[0],
            phi=pt.aim.at_ball(self.state.system, "object", cut=action[1]),
        )

    def observation_array(self) -> NDArray[np.float32]:
        """Return the system state as a 1D array of ball coordinates"""
        obs = self._null_obs()
        for ball_idx, ball_id in enumerate(self.state.system.balls.keys()):
            coords = self.state.system.balls[ball_id].state.rvw[0, :BALL_DIM]
            obs[self._slice(ball_idx)] = coords

        return obs

    def reset(self) -> None:
        if len(self.state.game.players) == 1:
            self.reset_single_player_env()
        else:
            raise NotImplementedError()

    def reset_single_player_env(self) -> None:
        """Reset things to an initial state"""
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

        self.state.system.cue.set_state(V0=V0_INIT)

        assert self.state.system.balls["cue"].state.s == pt.constants.stationary
        assert self.state.system.balls["object"].state.s == pt.constants.stationary
        assert not np.isnan(self.state.system.balls["cue"].state.rvw).any()
        assert not np.isnan(self.state.system.balls["object"].state.rvw).any()

    @classmethod
    def from_state(cls, state: State) -> SumToThreeSimulator:
        """Create a SumToThree environment from a State"""
        return cls(
            state,
            spaces=Spaces(
                observation=get_obs_space(
                    state.system.balls,
                    state.system.table,
                ),
                action=get_action_space(),
                reward=spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
            ),
        )

    @classmethod
    def single_player_env(cls, random_pos: bool = False) -> SumToThreeSimulator:
        """Create a 1 player environment (for training, evaluation, etc)"""
        return cls.from_state(create_initial_state(random_pos))

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
        reward_algorithm="simple",
    )

    def __init__(self, cfg: EasyDict) -> None:
        self.cfg = cfg
        self.calc_reward = get_reward_function(self.cfg.reward_algorithm)

        self._init_flag = False
        self._tracked_stats = EpisodicTrackedStats()
        self._env: SumToThreeSimulator

    def __repr__(self) -> str:
        return "SumToThreeEnv"

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

    def reset(self) -> ObservationDict:
        if not self._init_flag:
            self._env = SumToThreeSimulator.single_player_env()
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
