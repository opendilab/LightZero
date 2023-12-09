from __future__ import annotations
from pathlib import Path

from typing import Tuple

import attrs
from hbutils.system.filesystem.directory import shutil
import numpy as np
from ding.envs import BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from gym import spaces
from numpy.typing import NDArray
from pooltool.ani.camera import camera_states
from zoo.pooltool.datatypes import BasePoolToolEnv, ObservationDict, SimulationEnv

import pooltool as pt
from pooltool.ai.datatypes import State as PoolToolState

from pooltool import FrameStepper, ImageZip, image_stack
from pooltool.ani.image.utils import gif

def calc_reward(state: PoolToolState) -> float:
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
class EpisodicTrackedStats:
    eval_episode_length: int = 0
    eval_episode_return: float = 0.0


DUMMY_ENV = SimulationEnv.sum_to_three()


@ENV_REGISTRY.register("pooltool_sumtothree")
class SumToThreeEnv(BasePoolToolEnv):
    config = dict(
        env_name="PoolTool-SumToThree",
        env_type="not_board_games",
        episode_length=10,
        save_replay_path=None,
    )

    def __init__(self, cfg: EasyDict) -> None:
        self.cfg = cfg
        self._observation_space = DUMMY_ENV.spaces.observation
        self._action_space = DUMMY_ENV.spaces.action
        self._reward_space = DUMMY_ENV.spaces.reward

        self._save_replay_path = cfg.save_replay_path
        self._save_replay_count = 0

        self._tracked_stats = EpisodicTrackedStats()

    def __repr__(self) -> str:
        return "SumToThreeEnv"

    def close(self) -> None:
        raise NotImplementedError()

    def reset(self) -> ObservationDict:
        self._env = SimulationEnv.sum_to_three()
        self.manage_seeds()
        self.multisystem = pt.MultiSystem()
        self._tracked_stats = EpisodicTrackedStats()
        return self._env.observation()

    def step(self, action: NDArray[np.float32]) -> BaseEnvTimestep:
        self._env.set_action(self._env.scale_action(action))
        self._env.simulate()

        rew = calc_reward(self._env)

        self._tracked_stats.eval_episode_length += 1
        self._tracked_stats.eval_episode_return += rew

        done = self._tracked_stats.eval_episode_length == self.cfg["episode_length"]
        if done and self._save_replay_path is not None:
            self.save_episode_replay()

        info = attrs.asdict(self._tracked_stats) if done else {}

        self.multisystem.append(self._env.system.copy())

        return BaseEnvTimestep(
            obs=self._env.observation(),
            reward=np.array([rew], dtype=np.float32),
            done=done,
            info=info,
        )

    def save_episode_replay(self):
        """This barely works!

        FrameStepper inherits from ShowBase, which can only be spawned once per
        instance. This makes rendering difficult. On top of that, rendering multiple
        shots with the same FrameStepper leads to overlapping nodes, leading to a
        "brightening" of the scene with each shot passed through FrameStepper

        That said, this still works if you pass num_episodes_each_seed = 1 to
        eval_muzero, but use at your own risk.
        """
        STEPPER = FrameStepper()
        FPS = 6
        for i, system in enumerate(self.multisystem):
            image_stack_path = Path(self._save_replay_path) / f"seed_{self._seed:03d}-shot_{i:03d}"
            exporter = ImageZip(image_stack_path, ext="jpg", compress=False)
            imgs = image_stack(
                system=system,
                interface=STEPPER,
                size=(int(360 * 1.6), 360),
                fps=FPS,
                camera_state=camera_states["10_foot_overhead"],
                show_hud=False,
            )
            exporter.save(imgs)
            gif(sorted(list(image_stack_path.glob("*.jpg"))), str(image_stack_path) + ".gif", fps=FPS)
            shutil.rmtree(image_stack_path)
        self._save_replay_count += 1
