"""
cchess库来自：https://github.com/windshadow233/python-chinese-chess/tree/main/cchess
修改了cchess库一处代码，提升位运算速度
2346行：def popcount(x: BitBoard) -> int:
    \"\"\"
    计算 BitBoard 中 1 的个数
    Python 3.10+ 原生 bit_count() 比 bin().count('1') 快 10+ 倍
    \"\"\"
    return x.bit_count()


pikafish引擎可以自行去下载：
https://github.com/official-pikafish/Pikafish
https://www.pikafish.com/

Overview:
    中国象棋环境，封装 cchess 库以适配 LightZero 的 BaseEnv 接口
    中国象棋是一个双人对弈游戏，棋盘为 9x10（9列10行）
    
Mode:
    - ``self_play_mode``: 自对弈模式，用于 AlphaZero/MuZero 数据生成
    - ``play_with_bot_mode``: 与内置 bot 对战模式
    - ``eval_mode``: 评估模式

Observation Space:
    字典结构，包含以下键：
    - ``observation``: shape (N, 10, 9), float32. 
        - N = 14 * stack_obs_num + 1 = 14 * 4 + 1 = 57
        - 前 56 个通道为 4 帧历史观测堆叠，每一帧包含 14 个特征平面 (7种棋子 x 2种颜色)
        - 最后一个通道为当前玩家颜色平面 (全1表示红方/先手，全0表示黑方/后手)
        - 采用 Canonical View (规范视角)：始终以当前玩家视角观察棋盘 (自己棋子在下方/前7层)
    - ``action_mask``: shape (8100,), int8. 合法动作掩码，1表示合法，0表示非法
    - ``board``: shape (10, 9), int8. 棋盘可视化表示，用于调试或渲染
    - ``to_play``: shape (1,), int32. 当前该谁走 (-1: 结束/未知, 0: 黑方, 1: 红方)

Action Space:
    - Discrete(8100). 动作是移动的索引 (from_square * 90 + to_square)
    - 棋盘有 90 个位置 (0-89)，动作空间涵盖所有可能的起点-终点组合 (90 * 90 = 8100)
    - 实际合法动作远小于 8100 (通常几十到一百多)

Reward Space:
    - Box(-1, 1, (1,), float32).
    - +1: 当前玩家获胜 (Checkmate)
    - -1: 当前玩家失败 (被Checkmate或长将违规)
    - 0: 平局 (长闲循环、自然限招、无子可动等) 或 游戏未结束
"""

import copy
import os
from typing import List, Any, Tuple, Optional
from collections import deque

import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from ditk import logging
from easydict import EasyDict
from gymnasium import spaces

from . import cchess


def move_to_action(move: cchess.Move) -> int:
    """将 Move 对象转换为动作索引"""
    return move.from_square * 90 + move.to_square


def action_to_move(action: int) -> cchess.Move:
    """将动作索引转换为 Move 对象"""
    from_square = action // 90
    to_square = action % 90
    return cchess.Move(from_square, to_square)


@ENV_REGISTRY.register('cchess')
class ChineseChessEnv(BaseEnv):
    config = dict(
        env_id="ChineseChess",
        battle_mode='self_play_mode',
        battle_mode_in_simulation_env='self_play_mode',
        render_mode=None,  # 'human', 'svg', 'rgb_array'
        replay_path=None,
        agent_vs_human=False,
        prob_random_agent=0,
        prob_expert_agent=0,
        uci_engine_path=None,  # UCI引擎路径，如 'pikafish' 或 '/path/to/pikafish'
        engine_depth=5,  # 引擎搜索深度，通常1-20，深度越大越强
        channel_last=False,
        scale=False,
        stop_value=2,
        max_episode_steps=500,  # 最大回合数限制，防止无限回合
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict = None) -> None:
        self.cfg = cfg
        self.channel_last = cfg.channel_last
        self.scale = cfg.scale
        
        self.render_mode = cfg.render_mode
        self.replay_path = cfg.replay_path
        
        self.battle_mode = cfg.battle_mode
        assert self.battle_mode in ['self_play_mode', 'play_with_bot_mode', 'eval_mode']
        self.battle_mode_in_simulation_env = 'self_play_mode'
        
        self.agent_vs_human = cfg.agent_vs_human
        self.prob_random_agent = cfg.prob_random_agent
        self.prob_expert_agent = cfg.prob_expert_agent
        
        # UCI引擎配置
        self.uci_engine_path = cfg.get('uci_engine_path', None)
        self.engine_depth = cfg.get('engine_depth', 5)
        self.engine = None
        
        # 初始化UCI引擎（如果配置了）
        if self.uci_engine_path:
            try:
                from .cchess import engine
                self.engine = engine.SimpleEngine.popen_uci(self.uci_engine_path)
                logging.info(f"UCI引擎加载成功: {self.uci_engine_path}")
            except Exception as e:
                logging.warning(f"UCI引擎加载失败: {e}，将使用随机策略")
                self.engine = None
        
        # 最大步数限制
        self.max_episode_steps = cfg.max_episode_steps
        self.current_step = 0
        
        # 渲染相关
        self.frames = []  # 用于保存渲染图像帧
        
        # 初始化棋盘
        self.board = cchess.Board()
        
        self.players = [1, 2]  # 1: 红方(RED), 2: 黑方(BLACK)
        self._current_player = 1
        self._env = self

        # 历史观测堆叠
        self.stack_obs_num = 4
        self.obs_buffer = deque(maxlen=self.stack_obs_num)

        # 预计算：Board 棋子遍历所需的查找表
        self._piece_types = [cchess.PAWN, cchess.ROOK, cchess.KNIGHT, cchess.CANNON, 
                             cchess.ADVISOR, cchess.BISHOP, cchess.KING]
        self._colors = [cchess.RED, cchess.BLACK]
        
        # 预计算：BitBoard位索引到(row, col)的映射
        self._square_to_coord = np.array([(s // 9, s % 9) for s in range(90)], dtype=np.int32)

    def _mirror_action(self, action: int) -> int:
        """
        将动作在镜像坐标系统中转换（用于黑方视角转换）
        
        当黑方观测被旋转180度时，动作空间也需要相应转换。
        使用 cchess.square_mirror() 对起点和终点坐标进行镜像。
        
        Args:
            action: 原始动作索引 (from_square * 90 + to_square)
        
        Returns:
            镜像后的动作索引
        """
        from_square = action // 90
        to_square = action % 90
        from_square_mirror = cchess.square_mirror(from_square)
        to_square_mirror = cchess.square_mirror(to_square)
        return from_square_mirror * 90 + to_square_mirror

    def _get_raw_planes(self) -> np.ndarray:
        """
        获取当前棋盘的原始平面表示（固定语义：前7层红方，后7层黑方）
        不包含颜色通道，不进行视角转换
        
        优化：
        使用 lookup table 替代 python scan_forward 循环中的重复除法/取模计算
        虽然 scan_forward 本身在 Python 中循环，但减少了内部计算
        """
        state = np.zeros((14, 10, 9), dtype=np.float32)
        
        # 红方棋子 (前7层)
        for i, piece_type in enumerate(self._piece_types):
            mask = self.board.pieces_mask(piece_type, cchess.RED)
            if mask:
                # cchess.scan_forward 是 generator，我们手动解开以稍微加速
                # 或者更简单的：获取所有 set bits
                # 由于 cchess 库限制，这里还是使用 scan_forward，但后续坐标计算查表
                for square in cchess.scan_forward(mask):
                    r, c = self._square_to_coord[square]
                    state[i, r, c] = 1
                
        # 黑方棋子 (后7层)
        for i, piece_type in enumerate(self._piece_types):
            mask = self.board.pieces_mask(piece_type, cchess.BLACK)
            if mask:
                for square in cchess.scan_forward(mask):
                    r, c = self._square_to_coord[square]
                    state[i + 7, r, c] = 1
                
        return state

    def _update_obs_buffer(self):
        """更新观测缓存"""
        planes = self._get_raw_planes()
        self.obs_buffer.append(planes)

    def _player_step(self, action: int, flag: str, is_canonical_action: bool = True) -> BaseEnvTimestep:
        """
        执行一步棋
        
        Args:
            action: 动作索引
            flag: 标识字符串，用于日志记录
            is_canonical_action: 动作是否来自规范视角（Canonical View）
                - True: 动作来自策略网络（规范视角），黑方时需要镜像转换
                - False: 动作来自真实棋盘（如bot），不需要转换
        """
        # 关键修复：只有规范视角的动作在黑方时才需要转换
        if self._current_player == 2 and is_canonical_action:  # 黑方且是规范视角动作
            action_real = self._mirror_action(action)
        else:  # 红方 或 非规范视角动作（如bot）
            action_real = action
        
        legal_actions = self.legal_actions
        
        if action_real not in legal_actions:
            logging.warning(
                f"非法动作: {action} (real: {action_real}), 合法动作有 {len(legal_actions)} 个。"
                f"标志: {flag}, 玩家: {self._current_player}. 随机选择一个合法动作。"
            )
            action_real = self.random_action(canonical=False)  # 回退时使用真实坐标
        
        # 保存执行动作的玩家（用于奖励计算）
        acting_player = self._current_player
        
        move = action_to_move(action_real)  # 使用真实坐标
        self.board.push(move)
        
        # 增加步数计数
        self.current_step += 1
        
        # board.push() 已经自动切换了 turn，需要同步更新 _current_player
        self._current_player = 1 if self.board.turn else 2

        # 更新观测历史
        self._update_obs_buffer()
        
        # 检查游戏是否结束
        done = self.board.is_game_over()
        outcome = self.board.outcome()
        
        # 检查是否达到最大步数
        if self.current_step >= self.max_episode_steps:
            done = True
            outcome = None  # 达到最大步数视为平局
        
        # 默认 reward 为 0.0 (游戏未结束或和棋)
        reward_scalar = 0.0
        
        if done:
            # [DEBUG] 详细打印游戏结束原因，排查全和棋问题
            termination_reason = outcome.termination if outcome else "MaxSteps/Unknown"
            
            if outcome and outcome.winner is not None:
                # 有明确的胜者，奖励从执行动作的玩家视角计算
                winner_info = "RED" if outcome.winner == cchess.RED else "BLACK"
                if outcome.winner == cchess.RED:
                    # 红方胜
                    reward_scalar = 1.0 if acting_player == 1 else -1.0
                else:
                    # 黑方胜
                    reward_scalar = -1.0 if acting_player == 1 else 1.0
                logging.info(f"[ENV_DEBUG] Game Won! Winner: {winner_info}, ActingPlayer: {acting_player}, Reward: {reward_scalar}, Reason: {termination_reason}, Steps: {self.current_step}")
            else:
                # [优化策略] 统一判和逻辑：循环局面、最大步数、自然限招等均视为和棋 (0.0)
                logging.info(f"[ENV_DEBUG] Game Ended. Reason: {termination_reason}, Game DRAW, Steps: {self.current_step}")
        
        # 对外接口仍然使用 shape (1,) 的 ndarray
        reward = np.array([reward_scalar], dtype=np.float32)
        
        info = {}
        obs = self.observe()
        
        return BaseEnvTimestep(obs, reward, done, info)

    def step(self, action: int) -> BaseEnvTimestep:
        """
        环境的 step 函数
        """
        if self.battle_mode == 'self_play_mode':
            if self.prob_random_agent > 0:
                if np.random.rand() < self.prob_random_agent:
                    action = self.random_action(canonical=True)  # 规范视角的随机动作
            elif self.prob_expert_agent > 0:
                if np.random.rand() < self.prob_expert_agent:
                    action = self.random_action(canonical=True)  # TODO: 可以接入更强的 bot
            
            flag = "agent"
            # 自我对弈模式：动作来自策略网络（规范视角），需要转换
            timestep = self._player_step(action, flag, is_canonical_action=True)
            
            if timestep.done:
                # 【修复】在自我对弈中，使用规范视角（canonical view）
                # reward 已经是从执行动作的玩家（当前玩家）视角，直接使用
                # 不需要转换为 player 1 视角，因为观察也是规范视角
                reward_scalar = float(timestep.reward[0])
                timestep.info['eval_episode_return'] = reward_scalar
            
            return timestep
        
        elif self.battle_mode == 'play_with_bot_mode':
            # 玩家1的回合 (agent)
            flag = "bot_agent"
            timestep_player1 = self._player_step(action, flag, is_canonical_action=True)
            
            if timestep_player1.done:
                # player 1 执行后游戏结束，reward 已经是 player 1 视角
                timestep_player1.info['eval_episode_return'] = float(timestep_player1.reward[0])
                timestep_player1.obs['to_play'] = np.array([-1], dtype=np.int32)
                return timestep_player1
            
            # 玩家2（bot）的回合 - bot的动作来自真实棋盘，不需要转换
            bot_action = self.bot_action()  # 使用UCI引擎或随机策略
            flag = "bot_bot"
            timestep_player2 = self._player_step(bot_action, flag, is_canonical_action=False)
            
            # player 2 执行后游戏结束，reward 是 player 2 视角，需要转换为 player 1 视角
            reward_scalar = float(timestep_player2.reward[0])
            timestep_player2.info['eval_episode_return'] = -reward_scalar
            timestep_player2 = timestep_player2._replace(reward=-timestep_player2.reward)
            # [修正] 在 eval_mode 下，返回给 agent 的 observation 应该是轮到 agent (Player 1) 走
            # 所以 to_play 应该是 1 (RED)，而不是 -1
            timestep_player2.obs['to_play'] = np.array([1], dtype=np.int32)
            
            return timestep_player2
        
        elif self.battle_mode == 'eval_mode':
            # 玩家1的回合 (agent)
            flag = "eval_agent"
            timestep_player1 = self._player_step(action, flag, is_canonical_action=True)
            
            if timestep_player1.done:
                # player 1 执行后游戏结束，reward 已经是 player 1 视角
                timestep_player1.info['eval_episode_return'] = float(timestep_player1.reward[0])
                timestep_player1.obs['to_play'] = np.array([-1], dtype=np.int32)
                return timestep_player1
            
            # 玩家2的回合 (bot 或 human) - bot/human的动作来自真实棋盘，不需要转换
            if self.agent_vs_human:
                bot_action = self.human_to_action()
            else:
                bot_action = self.bot_action()  # 使用UCI引擎或随机策略
            
            flag = "eval_bot"
            timestep_player2 = self._player_step(bot_action, flag, is_canonical_action=False)
            
            # player 2 执行后游戏结束，reward 是 player 2 视角，需要转换为 player 1 视角
            reward_scalar = float(timestep_player2.reward[0])
            timestep_player2.info['eval_episode_return'] = -reward_scalar
            timestep_player2 = timestep_player2._replace(reward=-timestep_player2.reward)
            # [修正] 在 eval_mode 下，返回给 agent 的 observation 应该是轮到 agent (Player 1) 走
            # 所以 to_play 应该是 1 (RED)，而不是 -1
            timestep_player2.obs['to_play'] = np.array([1], dtype=np.int32)
            
            return timestep_player2

    def reset(self, start_player_index: int = 0, init_state: Optional[str] = None) -> dict:
        """
        重置环境
        """
        if init_state is None:
            self.board = cchess.Board()
        else:
            self.board = cchess.Board(fen=init_state)
        
        self.players = [1, 2]
        self.start_player_index = start_player_index
        
        # 重置步数计数器
        self.current_step = 0
        
        # 清空渲染帧
        self.frames = []
        
        # 确保 _current_player 与 board.turn 保持一致
        # board.turn: RED=True, BLACK=False
        self._current_player = 1 if self.board.turn else 2

        # 重置历史观测
        self.obs_buffer.clear()
        # 填充初始帧 (使用全0或初始状态重复)
        init_planes = self._get_raw_planes()
        for _ in range(self.stack_obs_num):
            self.obs_buffer.append(init_planes)
        
        # 设置动作空间和观察空间
        self._action_space = spaces.Discrete(90 * 90)  # 8100 个可能的动作
        self._reward_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # 计算Observation Shape: (14 * stack + 1, 10, 9)
        obs_channels = 14 * self.stack_obs_num + 1
        self._observation_space = spaces.Dict(
            {
                "observation": spaces.Box(low=0, high=1, shape=(obs_channels, 10, 9), dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=(90 * 90,), dtype=np.int8),
                "board": spaces.Box(low=0, high=7, shape=(10, 9), dtype=np.int8),
                "current_player_index": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),  # 0 或 1
                "to_play": spaces.Box(low=-1, high=2, shape=(1,), dtype=np.int32),  # -1, 1, 或 2
            }
        )
        
        obs = self.observe()
        return obs

    def current_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取当前堆叠和转换后的状态
        """
        # 1. 转换视角 (Canonical View)
        # 如果是黑方，需要将红方/黑方通道互换，并旋转棋盘
        stacked_obs = []
        for planes in self.obs_buffer:
            if self._current_player == 2: # 黑方
                # 原始: [0-6: 红, 7-13: 黑]
                # 目标: [0-6: 黑, 7-13: 红] (视角转换: 己方在前)
                red_planes = planes[:7]
                black_planes = planes[7:]
                
                # 交换并旋转 180 度
                # np.rot90(x, 2, axes=(1, 2)) 等价于旋转180度
                new_own = np.rot90(black_planes, 2, axes=(1, 2))
                new_opp = np.rot90(red_planes, 2, axes=(1, 2))
                
                transformed_planes = np.concatenate([new_own, new_opp], axis=0)
                stacked_obs.append(transformed_planes)
            else: # 红方
                # 原始即为目标: [0-6: 红(己), 7-13: 黑(敌)]
                stacked_obs.append(planes)
                
        # 2. 堆叠历史帧
        # shape: (14 * stack, 10, 9)
        state = np.concatenate(stacked_obs, axis=0)
        
        # 3. 添加颜色/ToPlay通道 (1层)
        # 在 Canonical View 下，通常网络总是视为"执红先手"视角
        # 但添加一个 feature map 全 1 (current) 或其他标记也是常见的
        # 这里保持原逻辑，如果是 player 1 (Red) 则全1，否则全0?
        # 不，既然已经旋转了视角，颜色通道应该表示 "当前是谁的回合" 还是 "我是谁"？
        # AlphaZero中，颜色通道是 constant 1 (if P1) or 0 (if P2). 
        # 但如果视角统一了，这个通道可以帮助区分先后手优势。
        color_plane = np.zeros((1, 10, 9), dtype=np.float32)
        if self._current_player == 1:
            color_plane[:] = 1.0
        
        state = np.concatenate([state, color_plane], axis=0)
        
        if self.scale:
            scale_state = state / 2 # 简单缩放，实际上binary plane不需要
        else:
            scale_state = state
        
        if self.channel_last:
            return np.transpose(state, [1, 2, 0]), np.transpose(scale_state, [1, 2, 0])
        else:
            return state, scale_state

    def observe(self) -> dict:
        """
        返回观察
        
        关键修复：对于黑方，需要将action_mask也进行镜像转换，
        使其与旋转后的观测空间保持一致。
        """
        legal_actions_list = self.legal_actions
        
        action_mask = np.zeros(90 * 90, dtype=np.int8)
        
        # 关键修复：如果是黑方，action_mask 需要镜像
        if self._current_player == 2:  # 黑方
            for action in legal_actions_list:
                action_mirror = self._mirror_action(action)
                action_mask[action_mirror] = 1
        else:  # 红方
            for action in legal_actions_list:
                action_mask[action] = 1
        
        # 获取棋盘的可视化表示
        board_visual = np.zeros((10, 9), dtype=np.int8)
        for square in range(90):
            piece = self.board.piece_at(square)
            if piece:
                row = cchess.square_row(square)
                col = cchess.square_column(square)
                # 棋子类型编码：1-7
                board_visual[row, col] = piece.piece_type
        
        if self.battle_mode in ['play_with_bot_mode', 'eval_mode']:
            return {
                "observation": self.current_state()[1],
                "action_mask": action_mask,
                "board": board_visual,
                "current_player_index": np.array([self.players.index(self._current_player)], dtype=np.int32),
                "to_play": np.array([-1], dtype=np.int32)
            }
        else:  # self_play_mode
            return {
                "observation": self.current_state()[1],
                "action_mask": action_mask,
                "board": board_visual,
                "current_player_index": np.array([self.players.index(self._current_player)], dtype=np.int32),
                "to_play": np.array([self._current_player], dtype=np.int32)
            }

    @property
    def legal_actions(self) -> List[int]:
        """
        返回所有合法动作的索引列表
        """
        legal_moves = list(self.board.legal_moves)
        return [move_to_action(move) for move in legal_moves]

    def get_done_winner(self) -> Tuple[bool, int]:
        """
        检查游戏是否结束并返回胜者
        Returns:
            - done: 游戏是否结束
            - winner: 胜者，1 表示红方，2 表示黑方，-1 表示和棋或游戏未结束
        """
        # 检查是否达到最大步数
        if self.current_step >= self.max_episode_steps:
            return True, -1  # 达到最大步数，视为平局
        
        done = self.board.is_game_over()
        if not done:
            return False, -1
        
        outcome = self.board.outcome()
        if outcome is None:
            return done, -1
        
        if outcome.winner is None:
            return True, -1  # 和棋
        elif outcome.winner == cchess.RED:
            return True, 1  # 红方胜
        else:
            return True, 2  # 黑方胜

    def get_done_reward(self) -> Tuple[bool, Optional[int]]:
        """
        检查游戏是否结束并从玩家1的视角返回奖励
        """
        done, winner = self.get_done_winner()
        if not done:
            return False, None
        
        if winner == 1:
            reward = 1
        elif winner == 2:
            reward = -1
        else:
            reward = 0
        
        return done, reward

    def random_action(self, canonical: bool = False) -> int:
        """
        随机选择一个合法动作
        
        Args:
            canonical: 是否返回规范视角的动作
                - False: 返回真实坐标（默认，用于bot等）
                - True: 返回规范视角坐标（用于self_play_mode中的随机agent）
        
        Returns:
            动作索引（真实坐标或规范视角坐标）
        """
        legal_actions_list = self.legal_actions  # 真实坐标
        action_real = np.random.choice(legal_actions_list)
        
        # 如果需要规范视角且当前是黑方，转换为镜像坐标
        if canonical and self._current_player == 2:
            return self._mirror_action(action_real)
        else:
            return action_real
    
    def bot_action(self) -> int:
        """
        使用UCI引擎或随机策略选择动作
        """
        if self.engine is not None:
            try:
                from .cchess import engine as engine_module
                # 使用引擎计算最佳走法，按深度限制
                limit = engine_module.Limit(depth=self.engine_depth)
                result = self.engine.play(self.board, limit)
                return move_to_action(result.move)
            except Exception as e:
                logging.warning(f"引擎调用失败: {e}，使用随机策略")
                return self.random_action()
        else:
            return self.random_action()

    def human_to_action(self) -> int:
        """
        从人类输入获取动作
        """
        print(self.board.unicode(axes=True, axes_type=0))
        while True:
            try:
                uci = input(f"请输入走法（UCI格式，如 h2e2）: ")
                move = cchess.Move.from_uci(uci)
                action = move_to_action(move)
                if action in self.legal_actions:
                    return action
                else:
                    print("非法走法，请重新输入")
            except KeyboardInterrupt:
                print("退出")
                import sys
                sys.exit(0)
            except Exception as e:
                print(f"输入错误: {e}，请重新输入")

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def __repr__(self) -> str:
        return "LightZero ChineseChess Env"

    @property
    def current_player(self) -> int:
        return self._current_player

    @property
    def current_player_index(self) -> int:
        return 0 if self._current_player == 1 else 1

    @property
    def next_player(self) -> int:
        return self.players[0] if self._current_player == self.players[1] else self.players[1]

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> spaces.Space:
        return self._reward_space

    def copy(self) -> 'ChineseChessEnv':
        """
        高效复制环境
        替代 copy.deepcopy(self)，只复制必要的动态状态
        """
        cls = self.__class__
        new_env = cls.__new__(cls)
        
        # 复制不可变配置
        new_env.cfg = self.cfg
        new_env.channel_last = self.channel_last
        new_env.scale = self.scale
        new_env.render_mode = self.render_mode
        new_env.replay_path = self.replay_path
        new_env.battle_mode = self.battle_mode
        new_env.battle_mode_in_simulation_env = self.battle_mode_in_simulation_env
        new_env.agent_vs_human = self.agent_vs_human
        new_env.prob_random_agent = self.prob_random_agent
        new_env.prob_expert_agent = self.prob_expert_agent
        new_env.uci_engine_path = self.uci_engine_path
        new_env.engine_depth = self.engine_depth
        new_env.max_episode_steps = self.max_episode_steps
        new_env.players = self.players
        new_env.start_player_index = self.start_player_index
        
        # 预计算表
        new_env._piece_types = self._piece_types
        new_env._colors = self._colors
        new_env._square_to_coord = self._square_to_coord
        
        # 复制动态状态 (需要拷贝)
        new_env.current_step = self.current_step
        new_env._current_player = self._current_player
        new_env.frames = [] # frames 一般不需要在 simulate 中复制
        new_env.engine = None # simulator 不需要 engine
        new_env._env = new_env
        
        # 关键：Board 的 copy，cchess.Board.copy() 已经是浅拷贝优化过的
        new_env.board = self.board.copy()
        
        # 关键：obs_buffer 的 copy
        # deque 本身浅拷贝即可，里面的 numpy array 是新的
        new_env.stack_obs_num = self.stack_obs_num
        new_env.obs_buffer = copy.copy(self.obs_buffer)
        
        # 空间定义
        new_env._action_space = self._action_space
        new_env._reward_space = self._reward_space
        new_env._observation_space = self._observation_space
        
        return new_env

    def simulate_action(self, action: int) -> Any:
        """
        模拟执行动作并返回新的模拟环境（用于 AlphaZero/MuZero 的 MCTS）
        
        Args:
            action: 动作索引。如果当前是黑方，这个动作是基于镜像坐标系统的，需要转回真实坐标
        """
        # 关键修复：如果是黑方，将镜像动作转换回真实坐标
        if self._current_player == 2:  # 黑方
            action_real = self._mirror_action(action)
        else:  # 红方
            action_real = action
        
        if action_real not in self.legal_actions:
            raise ValueError(f"动作 {action} (real: {action_real}) 不合法，当前玩家: {self._current_player}")
        
        # 创建新环境 (使用高效拷贝)
        new_env = self.copy()
        
        move = action_to_move(action_real)  # 使用真实坐标
        new_env.board.push(move)
        
        # 增加步数计数
        new_env.current_step += 1
        
        # board.push() 已经自动切换了 turn，需要同步更新 _current_player
        # board.turn: RED=True(1), BLACK=False(0)
        new_env._current_player = 1 if new_env.board.turn else 2

        # 关键：同步更新历史观测
        new_env._update_obs_buffer()
        
        return new_env

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.battle_mode = 'eval_mode'
        return [cfg for _ in range(evaluator_env_num)]

    def render(self, mode: str = None) -> None:
        """
        渲染棋盘
        
        根据LightZero官方文档：https://opendilab.github.io/LightZero/tutorials/envs/customize_envs.html
        
        Args:
            mode: 渲染模式
                - 'state_realtime_mode': 实时打印棋盘状态（文本）
                - 'image_realtime_mode': 实时显示SVG图像（暂不支持窗口显示）
                - 'image_savefile_mode': 保存SVG到frames，游戏结束后可转为文件
                - 'human': 等同于'state_realtime_mode'
                - 'svg': 返回SVG字符串（棋类游戏特有）
        """
        mode = mode or self.render_mode
        
        if mode is None:
            return None
        
        # LightZero标准模式：state_realtime_mode
        if mode in ['state_realtime_mode', 'human']:
            # 实时打印Unicode棋盘到控制台
            print("\n" + "=" * 50)
            print(f"步数: {self.current_step} | 当前玩家: {'红方' if self._current_player == 1 else '黑方'}")
            print(self.board.unicode(axes=True, axes_type=1))
            print("=" * 50)
            return None
        
        # LightZero标准模式：image_savefile_mode
        elif mode == 'image_savefile_mode':
            # 保存SVG到frames列表，游戏结束后可用save_render_output转为文件
            try:
                from .cchess import svg
                last_move = self.board.peek() if self.board.move_stack else None
                svg_str = svg.board(
                    self.board,
                    lastmove=last_move,
                    size=400
                )
                self.frames.append(svg_str)
            except Exception as e:
                logging.warning(f"SVG渲染失败: {e}")
            return None
        
        # LightZero标准模式：image_realtime_mode
        elif mode == 'image_realtime_mode':
            # 实时显示图像（对于SVG，暂不支持窗口显示）
            logging.warning("image_realtime_mode暂不支持实时窗口显示，请使用image_savefile_mode")
            return None
        
        # 棋类游戏特有：直接返回SVG字符串
        elif mode == 'svg':
            try:
                from .cchess import svg
                last_move = self.board.peek() if self.board.move_stack else None
                svg_str = svg.board(
                    self.board,
                    lastmove=last_move,
                    size=400
                )
                return svg_str
            except Exception as e:
                logging.warning(f"SVG渲染失败: {e}")
                return None
        
        # 其他模式
        else:
            logging.warning(f"不支持的渲染模式: {mode}")
            return None
    
    def save_render_output(self, replay_path: str = None, format: str = 'svg') -> None:
        """
        保存渲染输出到文件
        
        Args:
            replay_path: 保存路径，如果为None则使用self.replay_path
            format: 保存格式，目前支持'svg'
        """
        if not self.frames:
            logging.warning("没有可保存的渲染帧")
            return
        
        save_path = replay_path or self.replay_path
        if save_path is None:
            save_path = './replay_output'
        
        os.makedirs(save_path, exist_ok=True)
        
        if format == 'svg':
            for i, svg_str in enumerate(self.frames):
                file_path = os.path.join(save_path, f'step_{i:04d}.svg')
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(svg_str)
            logging.info(f"已保存 {len(self.frames)} 个SVG文件到 {save_path}")
        else:
            logging.warning(f"不支持的保存格式: {format}")
        
        # 清空frames
        self.frames = []
    
    def close(self) -> None:
        """关闭环境，释放资源"""
        if self.engine is not None:
            try:
                self.engine.quit()
                logging.info("UCI引擎已关闭")
            except Exception as e:
                logging.warning(f"关闭引擎时出错: {e}")
            finally:
                self.engine = None
