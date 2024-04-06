from easydict import EasyDict
from . import gomoku_play_with_bot
from . import gym_cartpole_v0
from . import tictactoe_play_with_bot


supported_env_cfg = {
    gomoku_play_with_bot.cfg.main_config.env.env_id: gomoku_play_with_bot.cfg,
    gym_cartpole_v0.cfg.main_config.env.env_id: gym_cartpole_v0.cfg,
    tictactoe_play_with_bot.cfg.main_config.env.env_id: tictactoe_play_with_bot.cfg,
}

supported_env_cfg = EasyDict(supported_env_cfg)
