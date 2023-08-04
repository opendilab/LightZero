import copy
import logging
import time
from collections import namedtuple
from typing import Dict, Any, Tuple, Union
from typing import List

import numpy as np
import torch
from ding.policy.base_policy import Policy
from ding.utils import POLICY_REGISTRY

from zoo.board_games.go.envs.katago_policy import GameState, str_coord



def get_image(path):
    from os import path as os_path

    import pygame
    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + '/' + path)
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc


from datetime import datetime


def generate_gif_filename(prefix="go", extension=".gif"):
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d-%H%M%S")
    filename = f"{prefix}-{timestamp}{extension}"
    return filename

def flatten_action_to_gtp_action(flatten_action, board_size):
    if flatten_action == board_size * board_size:
        return "pass"

    row = board_size - 1 - (flatten_action // board_size)
    col = flatten_action % board_size

    # 跳过字母 'I'
    if col >= ord('I') - ord('A'):
        col += 1

    col_str = chr(col + ord('A'))
    row_str = str(row + 1)

    gtp_action = col_str + row_str
    return gtp_action


@POLICY_REGISTRY.register('go_bot_policy_v0')
class GoBotPolicyV0(Policy):
    """
    Overview:
        Hard coded expert agent for go env.
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='go_bot_policy_v0',
        cuda=False,
        on_policy=True,
        learn=dict(
            multi_gpu=False,
        ),
        collect=dict(
            unroll_len=1,
        ),
        eval=dict(),
    )

    def legal_actions(self, board):
        legal_actions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == 0:
                    legal_actions.append(i * self.board_size + j)
        return legal_actions


    def _init_learn(self) -> None:
        pass

    def _init_collect(self) -> None:
        self.katago_policy = self._cfg.collect.katago_policy
        self.board_size = self._cfg.board_size
        katago_policy_init = True
        if katago_policy_init:
            # 使用函数生成基于当前时间的 GIF 文件名
            # gif_filename = generate_gif_filename(prefix=f"go_bs{self.board_size}", )
            # print(gif_filename)
            # self.save_gif_path = self.cfg.save_gif_path + gif_filename
            self.frames = []
            # TODO(pu): katago_game_state init
            self.katago_game_state = GameState(self.board_size)

    def _init_eval(self) -> None:
        pass

    def _forward_learn(self, data: dict) -> dict:
        pass

    def _katago_policy_bot(self, obs: Dict, temperature: float = 1) -> Dict[str, torch.Tensor]:
        # katago_policy_init = True
        # if katago_policy_init:
        #     # TODO(pu): katago_game_state init
        #     self.katago_game_state = GameState(self.board_size)

        self.katago_game_state = obs['katago_game_state']

        # ****** get katago action ******
        # TODO(pu): how to get the all history boards and moves?
        # s_time = time.time()
        bot_action = self.get_katago_action(to_play=obs['to_play'])
        # e_time = time.time()
        # print(f'katago_action time: {e_time - s_time}')

        # ****** update katago internal game state ******
        # TODO(pu): how to avoid this?
        katago_flatten_action = self.lz_flatten_to_katago_flatten(bot_action, self.board_size)
        # print('player katago play gtp action:', str_coord(katago_flatten_action, self.katago_game_state.board))
        action = bot_action
        return action

    def _forward_collect(self, obs: Dict, temperature: float = 1) -> Dict[str, torch.Tensor]:
        ready_env_id = list(obs.keys())
        output = {}
        for env_id in ready_env_id:
            action = self._katago_policy_bot(obs[env_id])
            output[env_id] = {
                'action': action,
                'probs': None,
            }
        return output

    def get_katago_action(self, to_play):
        command = ['get_katago_action']
        # self.current_player is the player who will play
        flatten_action = self.katago_policy.katago_command(self.katago_game_state, command, to_play=to_play)
        return flatten_action

    def update_katago_internal_game_state(self, katago_flatten_action, to_play):
        # Note: cannot use self.to_play, because self.to_play is updated after the self._player_step(action)
        # ****** update internal game state ******
        gtp_action = str_coord(katago_flatten_action, self.katago_game_state.board)
        if to_play == 1:
            command = ['play', 'b', gtp_action]
        else:
            command = ['play', 'w', gtp_action]
        self.katago_policy.katago_command(self.katago_game_state, command, to_play)

    def show_katago_board(self):
        command = ["showboard"]
        # self.current_player is the player who will play
        self.katago_policy.katago_command(self.katago_game_state, command)

    def lz_flatten_to_katago_flatten(self, lz_flatten_action, board_size):
        """ Convert lz Flattened Coordinate to katago Flattened Coordinate."""
        # self.arrsize = (board_size + 1) * (board_size + 2) + 1
        # xxxxxxxxxx
        # .........x
        # .........x
        # .........x
        # .........x
        # .........x
        # .........x
        # .........x
        # .........x
        # .........x
        # xxxxxxxxxx

        if lz_flatten_action == board_size * board_size:
            return 0  # Board.PASS_LOC
        # convert action index in [0, board_size**2) to coordinate (i, j)
        y, x = lz_flatten_action // board_size, lz_flatten_action % board_size
        return (board_size + 1) * (y + 1) + x + 1
        # 0 -> (0, 0) -> 11
        # 1 -> (0, 1) -> 12
        # 9 -> (1, 0) -> 21

        # row = action_number // self.board_size + 1
        # col = action_number % self.board_size + 1
        # return f"Play row {row}, column {col}"

    def _forward_eval(self, data: dict) -> dict:
        pass

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        pass

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        pass

    def default_model(self) -> Tuple[str, List[str]]:
        return 'bot_model', ['lzero.model.bot_model']

    def _monitor_vars_learn(self) -> List[str]:
        pass

    def reset(self):
        pass
