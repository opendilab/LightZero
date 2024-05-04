from __future__ import annotations

import numpy as np
from gym import spaces
from numpy.typing import NDArray

from zoo.pooltool.image_representation import PygameRenderer, RenderConfig

def get_image_obs_space(render_config: RenderConfig) -> spaces.Box:
    """
    Overview:
        Generate the observation space based on the renderer's configuration.
    Arguments:
        - render_config (:obj:`RenderConfig`): Render config object which contains configuration details including dimensions and planes.
    Returns:
        - space (:obj:`spaces.Box`): The observation space defining the bounds and shape based on the renderer's output.
    """
    return spaces.Box(
        low=0.0,
        high=1.0,
        shape=render_config.observation_shape,
        dtype=np.float32,
    )

def image_observation_array(renderer: PygameRenderer) -> NDArray[np.float32]:
    return renderer.observation()
