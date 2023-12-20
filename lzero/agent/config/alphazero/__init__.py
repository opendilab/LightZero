from easydict import EasyDict
from . import tictactoe_play_with_bot

supported_env_cfg = {
    tictactoe_play_with_bot.cfg.main_config.env.env_id: tictactoe_play_with_bot.cfg,
}

supported_env_cfg = EasyDict(supported_env_cfg)
