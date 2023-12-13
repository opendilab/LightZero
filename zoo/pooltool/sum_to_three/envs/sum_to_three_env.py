from __future__ import annotations

import gc
import shutil
from pathlib import Path

import attrs
import numpy as np
from ding.envs import BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from numpy.typing import NDArray
from zoo.pooltool.datatypes import BasePoolToolEnv

from pooltool import FrameStepper, ImageZip, image_stack
from pooltool.ai.bot.sumtothree_rl.core import (
    calc_reward,
    reset_single_player_env,
    single_player_env,
)
from pooltool.ai.datatypes import LightZeroEnv, ObservationDict
from pooltool.ani.camera import camera_states
from pooltool.ani.image.utils import gif

# from pooltool.system.datatypes import MultiSystem


@attrs.define
class EpisodicTrackedStats:
    eval_episode_length: int = 0
    eval_episode_return: float = 0.0


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
        self._init_flag = False
        self._env: LightZeroEnv

        self._save_replay_path = cfg.save_replay_path
        self._save_replay_count = 0

        self._tracked_stats = EpisodicTrackedStats()

    def __repr__(self) -> str:
        return "SumToThreeEnv"

    def close(self) -> None:
        # Trying to fix an apparent memory leak
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
            self._env = single_player_env()
            self._init_flag = True
        else:
            self._env = reset_single_player_env(self._env)

        self.manage_seeds()
        # self.multisystem = MultiSystem()
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
        if done and self._save_replay_path is not None:
            self.save_episode_replay()

        info = attrs.asdict(self._tracked_stats) if done else {}

        # self.multisystem.append(self._env.system.copy())

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
            image_stack_path = (
                Path(self._save_replay_path) / f"seed_{self._seed:03d}-shot_{i:03d}"
            )
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
            gif(
                sorted(list(image_stack_path.glob("*.jpg"))),
                str(image_stack_path) + ".gif",
                fps=FPS,
            )
            shutil.rmtree(image_stack_path)
        self._save_replay_count += 1
