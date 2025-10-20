# game_segment_priorzero.py
from lzero.mcts.buffer.game_segment import GameSegment as OriginalGameSegment
import numpy as np

class GameSegment(OriginalGameSegment):
    """
    [PRIORZERO-MODIFIED]
    继承自原始 GameSegment 并添加了存储 MCTS 策略的功能。
    """
    def __init__(self, action_space, game_segment_length=200, config=None, task_id=None):
        super().__init__(action_space, game_segment_length, config, task_id)
        # [PRIORZERO-NEW] 新增 mcts_policy_segment 用于存储 RFT 的目标
        self.mcts_policy_segment = []

    def append(self, action, obs, reward, action_mask, to_play, timestep):
        super().append(action, obs, reward, action_mask, to_play, timestep)
        # 在 append 时，我们还没有 MCTS 策略，所以先用一个占位符
        self.mcts_policy_segment.append(None)

    def store_search_stats(self, root_visit_dist, value, *args, **kwargs):
        """
        [PRIORZERO-MODIFIED]
        在存储搜索统计信息时，将 MCTS 访问计数分布也存起来。
        """
        super().store_search_stats(root_visit_dist, value, *args, **kwargs)
        # 最后一个被 append 的状态对应的 MCTS 策略
        # root_visit_dist 是一个 list, 我们需要它是一个 numpy array
        policy_array = np.array(root_visit_dist, dtype=np.float32)
        # 归一化
        if policy_array.sum() > 0:
            policy_array /= policy_array.sum()
        else: # 如果没有访问，则为均匀分布
            policy_array = np.ones_like(policy_array) / len(policy_array)

        # 存储到最后一个位置
        self.mcts_policy_segment[-1] = policy_array

    def game_segment_to_array(self):
        """
        [PRIORZERO-MODIFIED]
        将 mcts_policy_segment 也转换为 numpy 数组。
        """
        super().game_segment_to_array()
        self.mcts_policy_segment = np.array(self.mcts_policy_segment, dtype=object)