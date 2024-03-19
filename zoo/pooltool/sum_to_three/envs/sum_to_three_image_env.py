from __future__ import annotations

import gc
from typing import Any

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
from zoo.pooltool.image_representation import PygameRenderer, RenderConfig, RenderPlane

import pooltool as pt
import pooltool.constants as const

RENDER_CONFIG = RenderConfig(
    planes=[
        RenderPlane(ball_ids=["cue"]),
        RenderPlane(ball_ids=["object"]),
        RenderPlane(ball_ids=["cue", "object"]),
        RenderPlane(ball_ball_lines=[("cue", "object")]),
        RenderPlane(cushion_ids=["3", "12", "9", "18"]),
    ],
    line_width=1,
    antialias_circle=True,
    antialias_line=True,
)


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

    if len(pt.events.filter_type(state.system.events, pt.EventType.BALL_BALL)):
        return 0.1

    return 0.0


@attrs.define
class SumToThreeImageGym(PoolToolGym):
    renderer: PygameRenderer

    def observation_array(self) -> NDArray[np.float32]:
        """Return the system state as an image array"""
        return self.renderer.observation()

    def set_action(self, scaled_action: NDArray[np.float32]) -> None:
        self.system.cue.set_state(
            V0=scaled_action[0],
            phi=pt.aim.at_ball(self.system, "object", cut=scaled_action[1]),
        )

    def reset(self) -> None:
        if len(self.game.players) == 1:
            self.reset_single_player_env()
        else:
            raise NotImplementedError()

    def reset_single_player_env(self) -> None:
        """Return the passed environment, resetting things to an initial state"""
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
    def from_state(cls, state: State, px: int) -> SumToThreeImageGym:
        """Create a SumToThree environment from a State"""
        renderer = PygameRenderer.build(state.system.table, px, RENDER_CONFIG)
        renderer.init()

        env = cls(
            system=state.system,
            game=state.game,
            spaces=Spaces(
                observation=SumToThreeImageGym.get_obs_space(renderer),
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
            renderer=renderer,
        )

        env.renderer.set_state(env)
        return env

    @classmethod
    def single_player_env(cls, px: int, random_pos: bool = False) -> SumToThreeImageGym:
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

        return cls.from_state(State(system, game), px=px)

    @staticmethod
    def get_obs_space(renderer: PygameRenderer) -> Any:
        channels = len(renderer.render_config.planes)

        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(channels, renderer.coordinates.height, renderer.coordinates.width),
            dtype=np.float32,
        )


@attrs.define
class EpisodicTrackedStats:
    eval_episode_length: int = 0
    eval_episode_return: float = 0.0


@ENV_REGISTRY.register("pooltool_sumtothree_image")
class SumToThreeImageEnv(PoolToolEnv):
    config = dict(
        env_name="PoolTool-SumToThree-Image",
        env_type="not_board_games",
        px=20,
        episode_length=10,
    )

    def __init__(self, cfg: EasyDict) -> None:
        self.cfg = cfg
        self._init_flag = False
        self._tracked_stats = EpisodicTrackedStats()

        self._env: SumToThreeImageGym

    def __repr__(self) -> str:
        return "SumToThreeEnvImage"

    def close(self) -> None:
        self._env.renderer.close()

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
            self._env = SumToThreeImageGym.single_player_env(self.cfg.px)
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
