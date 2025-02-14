"""
Overview:
    This module provides helps render billiard games using pygame. It includes classes
    and methods for setting up game a rendering system to visualize the game state on a
    Pygame surface.
Usage Example:
    This example sets up a basic pool game simulation with SUMTOTHREE game type,
    initializes the simulation environment, executes a cue strike, and renders the
    results using a Pygame-based renderer.

    ```python
    import pooltool as pt
    from zoo.pooltool.image_representation import RenderPlane, RenderConfig, PygameRenderer
    from zoo.pooltool.datatypes import State

    # Setting up game rules and players.
    game_type = pt.GameType.SUMTOTHREE
    game = pt.get_ruleset(game_type)()
    game.players = [pt.Player("Player")]

    # Setting up the table and balls based on the game type.
    table = pt.Table.from_game_type(game_type)
    balls = pt.get_rack(
        game_type=game_type,
        table=table,
    )

    # Preparing the cue for striking.
    cue = pt.Cue(cue_ball_id=game.shot_constraints.cueball(balls))
    system = pt.System(table=table, balls=balls, cue=cue)

    # Executing a strike and simulating the game.
    system.cue.set_state(V0=1.7, phi=pt.aim.at_ball(system, "object", cut=17))
    pt.simulate(system, inplace=True)

    # Setting up rendering configurations.
    config = RenderConfig.default()

    # Building and initializing the renderer.
    renderer = PygameRenderer.build(system.table, config)
    renderer.init()
    renderer.set_state(State(system, game))

    # Render the current state (the balls' final resting positions) as a panel of feature planes
    renderer.display_observation(renderer.observation())

    # Render the entire shot (in 0.1 second intervals) with the feature planes
    renderer.animate_observations()
    renderer.close()
    ```
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable, List, Tuple

import json
from dataclasses import dataclass, field, asdict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numba
import numpy as np
import pygame
import pygame.gfxdraw
from numpy.typing import NDArray
from pygame.surface import Surface
from pygame.time import Clock
from zoo.pooltool.datatypes import State

import pooltool as pt
import pooltool.constants as const

pygame.init()

Color = Tuple[int, int, int]
WHITE: Color = (255, 255, 255)

_GRAYSCALE_CONVERSION_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float64)


@numba.jit(nopython=True)
def array_to_grayscale(raw_data):
    """
    Overview:
        Convert image array to grayscale for RGB conversion.
    Arguments:
        - color (:obj:`Color`): A color represented as a thruple of unsigned 8-bit integers.
    Returns:
        - grayscale (:obj:`Color`): The corresponding grayscale color.
    """
    height, width, _ = raw_data.shape
    grayscale_data = np.empty((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            grayscale_data[i, j] = int(
                raw_data[i, j, 0] * _GRAYSCALE_CONVERSION_WEIGHTS[0]
                + raw_data[i, j, 1] * _GRAYSCALE_CONVERSION_WEIGHTS[1]
                + raw_data[i, j, 2] * _GRAYSCALE_CONVERSION_WEIGHTS[2]
            )

    return grayscale_data


@dataclass
class RenderPlane:
    """
    Overview:
        Specifies the IDs of renderings for a feature plane.
    Attributes:
        - ball_ids (:obj:`List[str]`): A list of ball ids that should be rendered in the
            feature plane. Balls are rendered as filled circles.
        - cushion_ids (:obj:`List[str]`): A list of cushion ids that should be rendered
            in the feature plane. Cushions are rendered as lines.
        - ball_ball_lines (:obj:`List[Tuple[str, str]]`): A list of two-ples specifying
            ball-ball pairs that lines should be drawn between.
    """

    ball_ids: List[str] = field(default_factory=list)
    cushion_ids: List[str] = field(default_factory=list)
    ball_ball_lines: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class RenderConfig:
    """
    Overview:
        Dataclass that holds definitions of the feature planes and render settings.
    Attributes:
        - planes (:obj:`List[RenderPlane]`): A list of feature plane definitions. The
            order defines the observation array.
        - line_width (:obj:`int`): The width (in pixels) of lines drawn.
        - antialias_circle (:obj:`int`): Whether circles should be antialiased.
        - antialias_line (:obj:`int`): Whether lines should be antialiased.
    Properties:
        - channels (:obj:`int`): The number of feature planes
        - observation_shape (:obj:`Tuple[int, int, int]`): The shape of the observations made with
            this config.
    """

    px: int
    planes: List[RenderPlane]
    line_width: int
    antialias_circle: bool
    antialias_line: bool

    @property
    def channels(self) -> int:
        return len(self.planes)

    @property
    def observation_shape(self) -> Tuple[int, int, int]:
        return (self.channels, self.px, self.px // 2)

    def to_json(self, file_path: Path) -> Path:
        with open(file_path, "w") as file:
            json.dump(asdict(self), file, indent=4)
        return file_path

    @staticmethod
    def from_json(file_path: Path) -> RenderConfig:
        with open(file_path, "r") as file:
            config_data = json.load(file)
            planes = [
                RenderPlane(
                    ball_ids=plane_data["ball_ids"],
                    cushion_ids=plane_data["cushion_ids"],
                    ball_ball_lines=[tuple(line) for line in plane_data["ball_ball_lines"]]
                )
                for plane_data in config_data["planes"]
            ]
            return RenderConfig(
                px=config_data["px"],
                planes=planes,
                line_width=config_data["line_width"],
                antialias_circle=config_data["antialias_circle"],
                antialias_line=config_data["antialias_line"]
            )

    @classmethod
    def default(cls) -> RenderConfig:
        return cls(
            px=100,
            planes=[
                # Render the cue ball.
                RenderPlane(ball_ids=["cue"]),
                # Render object ball.
                RenderPlane(ball_ids=["object"]),
                # Render cue and object balls together.
                RenderPlane(ball_ids=["cue", "object"]),
                # Render line between cue and object balls.
                RenderPlane(ball_ball_lines=[("cue", "object")]),
                # Render the 4 cushions.
                RenderPlane(cushion_ids=["3", "12", "9", "18"]),  
            ],
            line_width=1,
            antialias_circle=True,
            antialias_line=True,
        )


class PygameRenderer:
    def __init__(
        self,
        coordinates: CoordinateManager,
        render_config: RenderConfig,
    ):
        """
        Overview:
            Renders a pool or billiards simulation using the Pygame library. This renderer is
            capable of rendering different elements of the simulation, such as balls, cushions,
            and specific interactions between balls, on a Pygame display surface.
        Attributes:
            - coordinates (:obj:`CoordinateManager`): Manages the conversion of simulation coordinates
                to pixel coordinates for rendering.
            - render_config (:obj:`RenderConfig`): Contains rendering configurations like antialiasing
                settings and line widths.
        """
        assert render_config.observation_shape == (len(render_config.planes), coordinates.height, coordinates.width)

        self.coordinates: CoordinateManager = coordinates
        self.render_config: RenderConfig = render_config
        
        self.screen: Surface
        self.clock: Clock
        self.state: State

    def init(self) -> None:
        """
        Overview:
            Initializes the Pygame environment for rendering, setting up the screen and clock.
        """
        # For off-screen rendering
        os.environ["SDL_DRIVER"] = "dummy"

        self.screen = pygame.Surface((self.coordinates.width, self.coordinates.height))
        self.clock = pygame.time.Clock()

    def set_state(self, state: State) -> None:
        """
        Overview:
            Sets the current state of the game or simulation to be rendered.
        Arguments:
            - state (:obj:`State`): The current state of the game.
        """
        self.state = state

    def _draw_balls(self, ball_ids: List[str]) -> None:
        """
        Overview:
            Renders balls specified by their IDs.
        Arguments:
            - ball_ids (:obj:`List[str]`): A list of ball IDs to be rendered.
        """
        for ball_id in ball_ids:
            ball = self.state.system.balls.get(ball_id)

            if ball is None:
                continue

            if ball.state.s == const.pocketed:
                continue

            x, y, _ = ball.state.rvw[0]
            radius = ball.params.R

            coords = self.coordinates.coords_to_px(x, y)
            scaled_radius = self.coordinates.scale_dist(radius)

            if self.render_config.antialias_circle:
                pygame.gfxdraw.aacircle(
                    self.screen,
                    int(coords[0]),
                    int(coords[1]),
                    int(scaled_radius),
                    WHITE,
                )
                pygame.gfxdraw.filled_circle(
                    self.screen,
                    int(coords[0]),
                    int(coords[1]),
                    int(scaled_radius),
                    WHITE,
                )
            else:
                pygame.draw.circle(
                    surface=self.screen,
                    color=WHITE,
                    center=coords,
                    radius=scaled_radius,
                )

    def _draw_cushions(self, cushion_ids: List[str]) -> None:
        """
        Overview:
            Renders cushion segments specified by their IDs.
        Arguments:
            - cushion_ids (:obj:`List[str]`): A list of cushion IDs to be rendered.
        """
        for cushion_id in cushion_ids:
            cushion = self.state.system.table.cushion_segments.linear.get(cushion_id)

            if cushion is None:
                continue

            if self.render_config.antialias_line:
                pygame.draw.aaline(
                    surface=self.screen,
                    color=WHITE,
                    start_pos=self.coordinates.coords_to_px(*cushion.p1[:2]),
                    end_pos=self.coordinates.coords_to_px(*cushion.p2[:2]),
                    blend=1,
                )
            else:
                pygame.draw.line(
                    surface=self.screen,
                    color=WHITE,
                    start_pos=self.coordinates.coords_to_px(*cushion.p1[:2]),
                    end_pos=self.coordinates.coords_to_px(*cushion.p2[:2]),
                    width=self.render_config.line_width,
                )

    def _draw_ball_to_ball_lines(self, ball_ball_lines: List[Tuple[str, str]]) -> None:
        """
        Overview:
            Renders lines between specified pairs of balls.
        Arguments:
            - ball_ball_lines (:obj:`List[Tuple[str, str]]`): A list of tuples, each containing two ball IDs
                between which a line will be drawn.
        """
        for ball1_id, ball2_id in ball_ball_lines:
            ball1 = self.state.system.balls[ball1_id]
            ball2 = self.state.system.balls[ball2_id]

            x1, y1, _ = ball1.state.rvw[0]
            x2, y2, _ = ball2.state.rvw[0]

            coords1 = self.coordinates.coords_to_px(x1, y1)
            coords2 = self.coordinates.coords_to_px(x2, y2)

            if self.render_config.antialias_line:
                pygame.draw.aaline(
                    surface=self.screen,
                    color=WHITE,
                    start_pos=coords1,
                    end_pos=coords2,
                    blend=1,
                )
            else:
                pygame.draw.line(
                    surface=self.screen,
                    color=WHITE,
                    start_pos=coords1,
                    end_pos=coords2,
                    width=self.render_config.line_width,
                )

    def draw_plane(self, plane: RenderPlane) -> None:
        """
        Overview:
            Renders a specific plane based on the provided RenderPlane configuration.
        Arguments:
            - plane (:obj:`RenderPlane`): The render plane configuration to use for drawing.
        """
        self.screen.fill((0, 0, 0))

        self._draw_balls(plane.ball_ids)
        self._draw_cushions(plane.cushion_ids)
        self._draw_ball_to_ball_lines(plane.ball_ball_lines)

    def draw_all(self) -> None:
        """
        Overview:
            Renders all elements of the game using the current state configuration.
        """
        all_balls = list(self.state.system.balls.keys())
        all_cushions = list(self.state.system.table.cushion_segments.linear.keys())

        self.draw_plane(
            RenderPlane(
                ball_ids=all_balls,
                cushion_ids=all_cushions,
            )
        )

    def screen_as_array(self) -> NDArray[np.float32]:
        """
        Overview:
            Converts the current screen to a numpy array representation.
        Returns:
            - screen_array (:obj:`NDArray[np.float32]`): The screen represented as a numpy array.
        """
        array = array_to_grayscale(pygame.surfarray.array3d(self.screen))

        # H, W, C
        array = array.transpose((1, 0))

        # Convert to float and normalize to [0, 1]
        array = array.astype(np.float32) / 255.0

        return array

    def observation(self) -> NDArray[np.float32]:
        """
        Overview:
            Returns the current screen as a multi-dimensional observation array.
        Returns:
            - observation (:obj:`NDArray[np.float32]`): A multi-channel array
              representing the current observation. Shape is ``(n_features, px, px//2)``,
              where ``n_features`` is the number of feature planes, and ``px`` is the
              number of pixels representing the length of the table.
        """
        array = np.zeros(
            self.render_config.observation_shape,
            dtype=np.float32,
        )

        for plane_idx, plane in enumerate(self.render_config.planes):
            self.draw_plane(plane)
            array[plane_idx, ...] = self.screen_as_array()

        return array

    def display_observation(self, observation: NDArray[np.float32]):
        """
        Overview:
            Displays the observation array (as a series of feature planes) for the given system state.
        Arguments:
            - observation (:obj:`NDArray[np.float32]`): The observation array to display.
        """
        observation = self.observation()
        channels = observation.shape[0]

        assert channels == self.render_config.channels

        ncols = int(np.ceil(np.sqrt(channels)))
        nrows = int(np.ceil(channels / ncols))

        _, axes = plt.subplots(nrows, ncols, figsize=(12, 8), facecolor="gray")

        plt.tight_layout()

        for i in range(channels):
            row, col = divmod(i, ncols)
            ax = axes[row, col]
            ax.imshow(observation[i, ...], cmap="gray")
            ax.axis("off")
            ax.set_title(f"Channel {i+1}")

        for j in range(channels, nrows * ncols):
            axes.flat[j].axis("off")

        plt.show()
        plt.close()

    def animate_observations(self) -> None:
        """
        Overview:
            Animates the observation array (as a series of feature planes) for the entire shot animation. Just for visualization purposes.
        Arguments:
            - system: The pool or billiards system being simulated.
        """

        time_interval = 0.1
        system = self.state.system
        pt.continuize(system, dt=time_interval, inplace=True)

        channels = self.render_config.channels
        nrows = int(math.ceil(math.sqrt(channels)))
        ncols = int(math.ceil(channels / nrows))

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(12, 8), facecolor="gray"
        )
        axes = axes.flatten()  # Flatten to a 1D list of axes

        plt.tight_layout()

        ims = []
        timepoints = len(list(system.balls.values())[0].history_cts)

        for i in range(timepoints):
            for ball in system.balls.values():
                ball.state = ball.history_cts[i]
            observation = self.observation()
            frame_artists = []
            for j in range(observation.shape[0]):
                img = axes[j].imshow(observation[j, ...], cmap="gray", animated=True)
                frame_artists.append(img)
            ims.append(frame_artists)

        # Turn off any unused axes
        for j in range(observation.shape[0], len(axes)):
            axes[j].axis("off")

        _ = FuncAnimation(
            fig,
            lambda i: ims[i],
            frames=timepoints,
            interval=time_interval*1000,
            blit=True,
            repeat=True,
        )

        plt.show()
        plt.close()

    def close(self) -> None:
        """
        Overview:
            Properly shuts down the Pygame environment to clean up resources.
        """
        pygame.quit()

    @classmethod
    def build(
        cls, table: pt.Table, render_config: RenderConfig
    ) -> PygameRenderer:
        """
        Overview:
            Factory method to create a new instance of PygameRenderer.
        Arguments:
            - table (:obj:`pt.Table`): The billiard table for which to create coordinates.
            - render_config (:obj:`RenderConfig`): Rendering configurations to be applied.
        Returns:
            - instance (:obj:`PygameRenderer`): A newly created instance of PygameRenderer.
        """
        return cls(CoordinateManager.build(table, render_config.px), render_config)


@dataclass
class CoordinateManager:
    """
    Overview:
        Manages coordinate transformations and scaling from simulation space to pixel space
        for rendering purposes.
    Attributes:
        - width (:obj:`int`): The width of the rendering screen in pixels.
        - height (:obj:`int`): The height of the rendering screen in pixels.
        - coords_to_px (:obj:`Callable[[float, float], Tuple[float, float]]`): A callable that
          converts simulation coordinates (x, y) to pixel coordinates on the screen.
        - scale_dist (:obj:`Callable[[float], float]`): A callable that scales a distance in
          the simulation space to a pixel-equivalent distance in the rendering screen.
    """

    width: int
    height: int
    coords_to_px: Callable[[float, float], Tuple[float, float]]
    scale_dist: Callable[[float], float]

    @classmethod
    def build(cls, table: pt.Table, px: int) -> CoordinateManager:
        """
        Overview:
            A factory method that creates an instance of CoordinateManager based on the dimensions
            of a billiard table and a specified pixel density. This method calculates scaling factors
            and offset adjustments needed for rendering of table components.
        Arguments:
            - table (:obj:`pt.Table`): A billiard table object which includes details about cushion segments and their coordinates.
            - px (:obj:`int`): The desired pixel density for the height of the rendering. The width is automatically adjusted together
                maintain the aspect ratio. The value corresponds to the length of the table, which is twice its width.
        Returns:
            - (:obj:`CoordinateManager`): An instance of CoordinateManager configured for the provided billiard table and pixel density.
        Raises:
            - AssertionError: If the pixel density (`px`) is not even or if the table's dimensions do not conform to expected proportions.
        """
        assert px % 2 == 0, "px should be even for symmetric table representation"

        xs = []
        ys = []

        for cushion in table.cushion_segments.linear.values():
            xs.append(cushion.p1[0])
            xs.append(cushion.p2[0])
            ys.append(cushion.p1[1])
            ys.append(cushion.p2[1])

        screen_x_min, screen_x_max = min(xs), max(xs)
        screen_y_min, screen_y_max = min(ys), max(ys)

        table_x_min = table.cushion_segments.linear["3"].p1[0]
        table_y_min = table.cushion_segments.linear["18"].p1[1]

        assert screen_y_max - screen_y_min > screen_x_max - screen_x_min, "Assume y > x"

        px_y = px
        px_x = px // ((screen_y_max - screen_y_min) / (screen_x_max - screen_x_min))
        if (px_y % 2) > 0:
            px_x += 1

        sy = (px_y - 1) / (screen_y_max - screen_y_min)
        sx = (px_x - 1) / (screen_x_max - screen_x_min)

        offset_y = table_y_min - screen_y_min
        offset_x = table_x_min - screen_x_min

        def coords_to_px(x: float, y: float) -> Tuple[float, float]:
            return sx * (x + offset_x), sy * (y + offset_y)

        def scale_dist(d: float) -> float:
            return max(1.0, d * max(sy, sx))

        return CoordinateManager(int(px_x), int(px_y), coords_to_px, scale_dist)
