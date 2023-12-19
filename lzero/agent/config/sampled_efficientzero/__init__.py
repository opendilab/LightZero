from easydict import EasyDict
from . import gym_cartpole_v0


supported_env_cfg = {
    gym_cartpole_v0.cfg.main_config.env.env_id: gym_cartpole_v0.cfg,
}

supported_env_cfg = EasyDict(supported_env_cfg)
