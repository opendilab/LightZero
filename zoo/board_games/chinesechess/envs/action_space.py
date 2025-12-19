"""
中国象棋动作空间模块

支持两种动作空间：
1. 8100维度：90*90 全连接空间，简单但稀疏
2. 2086维度：只包含合理走法的紧凑空间（参考 ChineseChess-AlphaZero）

2086动作空间组成：
- 基础走法(2038个)：车/炮/兵/将的直线移动 + 马的日字跳跃
- 士的斜角移动(16个)：红方8个 + 黑方8个
- 象的田字移动(32个)：红方16个 + 黑方16个

动作格式：
- 2086空间使用4位数字字符串 "CRCT"：C=列(0-8), R=行(0-9)
- UCI格式使用字母+数字 "a0b2"：列(a-i), 行(0-9)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


def _create_action_labels_2086() -> List[str]:
    """
    创建2086维度的动作标签列表

    格式: "CRCT" - 4位数字字符串
    - 第1位(C): 起点列 (0-8)
    - 第2位(R): 起点行 (0-9)
    - 第3位(C): 终点列 (0-8)
    - 第4位(T): 终点行 (0-9)

    Returns:
        List[str]: 2086个动作标签
    """
    labels_array = []

    # 基础走法：直线移动 + 马步
    for row in range(10):
        for col in range(9):
            # 目标位置：同行 + 同列 + 马步
            destinations = (
                [(row, t) for t in range(9)] +  # 同行9个位置
                [(t, col) for t in range(10)] +  # 同列10个位置
                [(row + dr, col + dc) for (dr, dc) in  # 马步8个方向
                 [(-2, -1), (-1, -2), (-2, 1), (1, -2),
                  (2, -1), (-1, 2), (2, 1), (1, 2)]]
            )
            for (dest_row, dest_col) in destinations:
                # 过滤：不能原地不动，不能越界
                if ((row, col) != (dest_row, dest_col) and
                    0 <= dest_row < 10 and 0 <= dest_col < 9):
                    move = f"{col}{row}{dest_col}{dest_row}"
                    labels_array.append(move)

    # 红方士的走法 (九宫内斜角移动)
    # 士在 d0,f0,e1,d2,f2 五个位置
    labels_array.extend([
        '3041', '5041',  # d0->e1, f0->e1
        '3241', '5241',  # d2->e1, f2->e1
        '4130', '4150',  # e1->d0, e1->f0
        '4132', '4152',  # e1->d2, e1->f2
    ])

    # 黑方士的走法
    # 士在 d9,f9,e8,d7,f7 五个位置
    labels_array.extend([
        '3948', '5948',  # d9->e8, f9->e8
        '3748', '5748',  # d7->e8, f7->e8
        '4839', '4859',  # e8->d9, e8->f9
        '4837', '4857',  # e8->d7, e8->f7
    ])

    # 红方象的走法 (田字移动，不能过河)
    # 象在 c0,g0,a2,e2,i2,c4,g4 七个位置
    labels_array.extend([
        '2002', '2042',  # c0->a2, c0->e2
        '6042', '6082',  # g0->e2, g0->i2
        '2402', '2442',  # c4->a2, c4->e2
        '6442', '6482',  # g4->e2, g4->i2
        '0220', '4220',  # a2->c0, e2->c0
        '4260', '8260',  # e2->g0, i2->g0
        '0224', '4224',  # a2->c4, e2->c4
        '4264', '8264',  # e2->g4, i2->g4
    ])

    # 黑方象的走法
    # 象在 c9,g9,a7,e7,i7,c5,g5 七个位置
    labels_array.extend([
        '2907', '2947',  # c9->a7, c9->e7
        '6947', '6987',  # g9->e7, g9->i7
        '2507', '2547',  # c5->a7, c5->e7
        '6547', '6587',  # g5->e7, g5->i7
        '0729', '4729',  # a7->c9, e7->c9
        '4769', '8769',  # e7->g9, i7->g9
        '0725', '4725',  # a7->c5, e7->c5
        '4765', '8765',  # e7->g5, i7->g5
    ])

    return labels_array


def _flip_move_2086(move: str) -> str:
    """翻转动作（用于黑方视角转换）"""
    return ''.join([
        str(8 - int(move[0])),
        str(9 - int(move[1])),
        str(8 - int(move[2])),
        str(9 - int(move[3]))
    ])


# 预计算的动作标签
ACTION_LABELS_2086 = _create_action_labels_2086()
ACTION_LABELS_2086_FLIPPED = [_flip_move_2086(x) for x in ACTION_LABELS_2086]

# 动作标签 -> 索引的映射
ACTION_TO_INDEX_2086 = {label: idx for idx, label in enumerate(ACTION_LABELS_2086)}
ACTION_TO_INDEX_2086_FLIPPED = {label: idx for idx, label in enumerate(ACTION_LABELS_2086_FLIPPED)}

# 用于策略翻转的索引映射
UNFLIPPED_INDEX = [ACTION_LABELS_2086.index(x) for x in ACTION_LABELS_2086_FLIPPED]


class ActionSpaceConverter:
    """
    动作空间转换器

    支持以下格式之间的转换：
    - UCI格式: "a0b2" (字母列+数字行)
    - 2086格式: "0012" (数字列+数字行)
    - 8100索引: from_square * 90 + to_square
    - 2086索引: 在ACTION_LABELS_2086中的位置
    """

    # UCI列名到数字的映射
    COL_UCI_TO_NUM = {c: str(i) for i, c in enumerate('abcdefghi')}
    COL_NUM_TO_UCI = {str(i): c for i, c in enumerate('abcdefghi')}

    def __init__(self, action_space_size: int = 2086):
        """
        初始化转换器

        Args:
            action_space_size: 动作空间大小，2086 或 8100
        """
        self.action_space_size = action_space_size

        if action_space_size == 2086:
            self.labels = ACTION_LABELS_2086
            self.label_to_index = ACTION_TO_INDEX_2086
        else:
            self.labels = None
            self.label_to_index = None

    @staticmethod
    def uci_to_2086_label(uci: str) -> str:
        """
        UCI格式转2086标签

        "a0b2" -> "0012"
        """
        col1 = ActionSpaceConverter.COL_UCI_TO_NUM[uci[0]]
        row1 = uci[1]
        col2 = ActionSpaceConverter.COL_UCI_TO_NUM[uci[2]]
        row2 = uci[3]
        return f"{col1}{row1}{col2}{row2}"

    @staticmethod
    def label_2086_to_uci(label: str) -> str:
        """
        2086标签转UCI格式

        "0012" -> "a0b2"
        """
        col1 = ActionSpaceConverter.COL_NUM_TO_UCI[label[0]]
        row1 = label[1]
        col2 = ActionSpaceConverter.COL_NUM_TO_UCI[label[2]]
        row2 = label[3]
        return f"{col1}{row1}{col2}{row2}"

    @staticmethod
    def uci_to_8100_index(uci: str) -> int:
        """
        UCI格式转8100索引

        "a0b2" -> from_square * 90 + to_square
        """
        from_col = ord(uci[0]) - ord('a')
        from_row = int(uci[1])
        to_col = ord(uci[2]) - ord('a')
        to_row = int(uci[3])

        from_square = from_row * 9 + from_col
        to_square = to_row * 9 + to_col

        return from_square * 90 + to_square

    @staticmethod
    def index_8100_to_uci(action: int) -> str:
        """
        8100索引转UCI格式

        action -> "a0b2"
        """
        from_square = action // 90
        to_square = action % 90

        from_col = from_square % 9
        from_row = from_square // 9
        to_col = to_square % 9
        to_row = to_square // 9

        return f"{chr(ord('a') + from_col)}{from_row}{chr(ord('a') + to_col)}{to_row}"

    @classmethod
    def uci_to_2086_index(cls, uci: str) -> int:
        """
        UCI格式转2086索引

        "a0b2" -> 索引
        """
        label = cls.uci_to_2086_label(uci)
        return ACTION_TO_INDEX_2086.get(label, -1)

    @classmethod
    def index_2086_to_uci(cls, index: int) -> str:
        """
        2086索引转UCI格式

        索引 -> "a0b2"
        """
        label = ACTION_LABELS_2086[index]
        return cls.label_2086_to_uci(label)

    @staticmethod
    def index_8100_to_2086(action_8100: int) -> int:
        """
        8100索引转2086索引

        Returns:
            2086索引，如果不在2086空间中返回-1
        """
        uci = ActionSpaceConverter.index_8100_to_uci(action_8100)
        return ActionSpaceConverter.uci_to_2086_index(uci)

    @staticmethod
    def index_2086_to_8100(action_2086: int) -> int:
        """
        2086索引转8100索引
        """
        uci = ActionSpaceConverter.index_2086_to_uci(action_2086)
        return ActionSpaceConverter.uci_to_8100_index(uci)

    @staticmethod
    def mirror_2086_index(action: int) -> int:
        """
        镜像2086索引（用于黑方视角转换）

        Args:
            action: 原始2086索引

        Returns:
            镜像后的2086索引
        """
        label = ACTION_LABELS_2086[action]
        flipped_label = _flip_move_2086(label)
        return ACTION_TO_INDEX_2086.get(flipped_label, -1)

    @staticmethod
    def flip_policy(policy: np.ndarray) -> np.ndarray:
        """
        翻转策略向量（用于黑方视角转换）

        Args:
            policy: shape=(2086,) 的策略向量

        Returns:
            翻转后的策略向量
        """
        return np.asarray([policy[idx] for idx in UNFLIPPED_INDEX])


def create_action_mask_2086(legal_actions_8100: List[int]) -> np.ndarray:
    """
    从8100格式的合法动作列表创建2086格式的动作掩码

    Args:
        legal_actions_8100: 8100空间中的合法动作索引列表

    Returns:
        shape=(2086,) 的动作掩码
    """
    mask = np.zeros(2086, dtype=np.int8)

    for action_8100 in legal_actions_8100:
        action_2086 = ActionSpaceConverter.index_8100_to_2086(action_8100)
        if action_2086 >= 0:
            mask[action_2086] = 1

    return mask


def create_action_mask_2086_from_uci(legal_moves_uci: List[str]) -> np.ndarray:
    """
    从UCI格式的合法动作列表创建2086格式的动作掩码

    Args:
        legal_moves_uci: UCI格式的合法动作列表，如 ["a0b2", "h2e2"]

    Returns:
        shape=(2086,) 的动作掩码
    """
    mask = np.zeros(2086, dtype=np.int8)

    for uci in legal_moves_uci:
        action_2086 = ActionSpaceConverter.uci_to_2086_index(uci)
        if action_2086 >= 0:
            mask[action_2086] = 1

    return mask


# 预计算：8100 -> 2086 的映射表（加速转换）
_INDEX_8100_TO_2086_TABLE = np.full(8100, -1, dtype=np.int32)
for idx, label in enumerate(ACTION_LABELS_2086):
    uci = ActionSpaceConverter.label_2086_to_uci(label)
    idx_8100 = ActionSpaceConverter.uci_to_8100_index(uci)
    _INDEX_8100_TO_2086_TABLE[idx_8100] = idx

# 预计算：2086 -> 8100 的映射表
_INDEX_2086_TO_8100_TABLE = np.zeros(2086, dtype=np.int32)
for idx, label in enumerate(ACTION_LABELS_2086):
    uci = ActionSpaceConverter.label_2086_to_uci(label)
    _INDEX_2086_TO_8100_TABLE[idx] = ActionSpaceConverter.uci_to_8100_index(uci)


def fast_8100_to_2086(action_8100: int) -> int:
    """快速将8100索引转换为2086索引（使用预计算表）"""
    return _INDEX_8100_TO_2086_TABLE[action_8100]


def fast_2086_to_8100(action_2086: int) -> int:
    """快速将2086索引转换为8100索引（使用预计算表）"""
    return _INDEX_2086_TO_8100_TABLE[action_2086]


def fast_create_action_mask_2086(legal_actions_8100: List[int]) -> np.ndarray:
    """
    快速创建2086格式的动作掩码（使用预计算表）

    Args:
        legal_actions_8100: 8100空间中的合法动作索引列表

    Returns:
        shape=(2086,) 的动作掩码
    """
    mask = np.zeros(2086, dtype=np.int8)

    for action_8100 in legal_actions_8100:
        action_2086 = _INDEX_8100_TO_2086_TABLE[action_8100]
        if action_2086 >= 0:
            mask[action_2086] = 1

    return mask


if __name__ == "__main__":
    # 测试代码
    print(f"2086动作空间大小: {len(ACTION_LABELS_2086)}")
    print(f"前10个动作: {ACTION_LABELS_2086[:10]}")
    print(f"士的动作: {ACTION_LABELS_2086[-48:-32]}")
    print(f"象的动作: {ACTION_LABELS_2086[-32:]}")

    # 测试转换
    test_uci = "h2e2"  # 炮二平五
    label = ActionSpaceConverter.uci_to_2086_label(test_uci)
    idx_2086 = ActionSpaceConverter.uci_to_2086_index(test_uci)
    idx_8100 = ActionSpaceConverter.uci_to_8100_index(test_uci)

    print(f"\n转换测试 (UCI: {test_uci}):")
    print(f"  -> 2086标签: {label}")
    print(f"  -> 2086索引: {idx_2086}")
    print(f"  -> 8100索引: {idx_8100}")
    print(f"  -> 从8100还原UCI: {ActionSpaceConverter.index_8100_to_uci(idx_8100)}")
    print(f"  -> 从2086还原UCI: {ActionSpaceConverter.index_2086_to_uci(idx_2086)}")

    # 验证快速转换
    print(f"\n快速转换验证:")
    print(f"  8100->2086: {fast_8100_to_2086(idx_8100)}")
    print(f"  2086->8100: {fast_2086_to_8100(idx_2086)}")
