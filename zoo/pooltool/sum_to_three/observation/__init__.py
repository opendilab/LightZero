from __future__ import annotations


from enum import Enum
from zoo.pooltool.sum_to_three.observation.coordinate import get_coordinate_obs_space, coordinate_observation_array
from zoo.pooltool.sum_to_three.observation.image import get_image_obs_space, image_observation_array

class ObservationType(Enum):
    IMAGE = "image"
    COORDINATE = "coordinate"


__all__ = [
    "get_coordinate_obs_space",
    "coordinate_observation_array",
    "get_image_obs_space",
    "image_observation_array",
    "ObservationType",
]
