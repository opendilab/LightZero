import gym
from easydict import EasyDict

from ding.envs import DingEnvWrapper
from lzero.envs.lightzero_env_wrapper import LightZeroEnvWrapper


def get_wrappered_env(wrapper_cfg: EasyDict, env_name: str, env_type: str):
    assert env_type != "Atari" and env_type != "board_games", "Now we only support classic_control and box2d env " \
                                                              "and don't support Atari and board_games in LightZeroEnvWrapper! Please use the " \
                                                              "zoo/atari/envs/atari_lightzero_env.py and zoo/board_games/*/envs/*_env.py instead."
    return lambda: DingEnvWrapper(
        gym.make(env_name),
        cfg={
            'env_wrapper': [
                lambda env: LightZeroEnvWrapper(env, wrapper_cfg)
            ]
        }
    )
