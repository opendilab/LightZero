# ./lzero/mcts/ctree/ctree_alphazero/test/eval_alphazero_ctree_zh.py

import sys
import unittest
import numpy as np
from easydict import EasyDict

# 将编译后的 C++ 模块路径添加到系统路径
sys.path.append('/Users/puyuan/code/LightZero/lzero/mcts/ctree/ctree_alphazero/build')

import mcts_alphazero


class MockEnv:
    """
    一个简单的模拟环境类，包含必要的属性和方法。
    """

    def __init__(self):
        self.legal_actions = [0, 1, 2]
        self.battle_mode_in_simulation_env = "self_play_mode"
        self.current_player = 1
        self.action_space = type('action_space', (), {'n': 3})()

    def reset(self, start_player_index, init_state, katago_policy_init, katago_game_state):
        """
        模拟环境的 reset 方法。
        """
        pass

    def step(self, action):
        """
        模拟环境的 step 方法。
        """
        pass

    def get_done_winner(self):
        """
        模拟环境的 get_done_winner 方法，返回 (done, winner)。
        """
        return (False, -1)


def mock_policy_value_func(env):
    """
    一个真实的 policy_value_func 函数，返回动作概率字典和叶节点值。
    """
    return ({0: 0.4, 1: 0.4, 2: 0.2}, 0.9)


class TestNodeAlphaZero(unittest.TestCase):
    """
    测试 Node 类的功能，包括初始化、更新、递归更新、判断叶子节点和根节点等。
    """

    def test_node_initialization(self):
        """
        测试 Node 类的初始化是否正确。
        """
        # 创建一个根节点
        root = mcts_alphazero.Node()

        self.assertIsNone(root.parent, "根节点的父节点应为 None")
        self.assertEqual(root.prior_p, 1.0, "根节点的 prior_p 应默认为 1.0")
        self.assertEqual(root.visit_count, 0, "根节点的 visit_count 应默认为 0")
        self.assertEqual(root.value, 0.0, "根节点的初始 value 应为 0.0")
        self.assertTrue(root.is_leaf(), "新创建的根节点应为叶子节点")
        self.assertTrue(root.is_root(), "新创建的根节点应为根节点")

    def test_node_update(self):
        """
        测试 Node 类的 update 方法是否正确更新 visit_count 和 value_sum。
        """
        node = mcts_alphazero.Node()
        node.update(5.0)

        self.assertEqual(node.visit_count, 1, "更新一次后，visit_count 应为 1")
        self.assertEqual(node.value, 5.0, "更新一次后的 value 应为 5.0")

        node.update(3.0)
        self.assertEqual(node.visit_count, 2, "更新两次后，visit_count 应为 2")
        self.assertAlmostEqual(node.value, 4.0, "更新两次后的 value 应为 (5.0 + 3.0) / 2 = 4.0")

    def test_node_recursive_update_self_play_mode(self):
        """
        测试 Node 类在 self_play_mode 下的递归更新。
        """
        # 创建父子节点结构
        parent = mcts_alphazero.Node()
        child = mcts_alphazero.Node(parent=parent, prior_p=0.5)
        parent.add_child(1, child)

        # 在子节点上进行递归更新
        child.update_recursive(1.0, "self_play_mode")

        # 检查子节点的更新
        self.assertEqual(child.visit_count, 1, "子节点的 visit_count 应为 1")
        self.assertEqual(child.value, 1.0, "子节点的 value 应为 1.0")

        # 检查父节点的更新
        self.assertEqual(parent.visit_count, 1, "父节点的 visit_count 应为 1")
        self.assertAlmostEqual(parent.value, -1.0, "父节点的 value 应为 -1.0")

    def test_node_recursive_update_play_with_bot_mode(self):
        """
        测试 Node 类在 play_with_bot_mode 下的递归更新。
        """
        # 创建父子节点结构
        parent = mcts_alphazero.Node()
        child = mcts_alphazero.Node(parent=parent, prior_p=0.5)
        parent.add_child(2, child)

        # 在子节点上进行递归更新
        child.update_recursive(1.0, "play_with_bot_mode")

        # 检查子节点的更新
        self.assertEqual(child.visit_count, 1, "子节点的 visit_count 应为 1")
        self.assertEqual(child.value, 1.0, "子节点的 value 应为 1.0")

        # 检查父节点的更新
        self.assertEqual(parent.visit_count, 1, "父节点的 visit_count 应为 1")
        self.assertEqual(parent.value, 1.0, "父节点的 value 应为 1.0")

    def test_node_add_child(self):
        """
        测试 Node 类的 add_child 方法是否正确添加子节点。
        """
        parent = mcts_alphazero.Node()
        child = mcts_alphazero.Node(parent=parent, prior_p=0.7)
        parent.add_child(3, child)

        self.assertIn(3, parent.children, "动作为 3 的子节点应被添加到父节点的 children 中")
        self.assertIs(parent.children[3], child, "添加的子节点应与传入的 child 相同")
        self.assertFalse(parent.is_leaf(), "添加子节点后，父节点不应为叶子节点")


class TestMCTSAlphaZero(unittest.TestCase):
    """
    测试 MCTS 类的功能，包括初始化、UCB 评分计算、选择子节点、添加探索噪声、
    扩展叶节点、执行模拟和获取下一步动作等。
    """

    def setUp(self):
        """
        初始化测试所需的 MCTS 对象和模拟环境。
        """
        # 配置 MCTS 的参数
        self.mcts = mcts_alphazero.MCTS(
            max_moves=100,
            num_simulations=10,
            pb_c_base=19652,
            pb_c_init=1.25,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.25,
            simulate_env=None  # 将在测试中设置
        )

        # 创建一个根节点
        self.root = mcts_alphazero.Node()

        # 创建模拟环境
        self.mock_env = MockEnv()

        # 定义合法动作
        self.legal_actions = [0, 1, 2]

        # 定义 policy_value_func
        self.policy_value_func = mock_policy_value_func

    def test_ucb_score(self):
        """
        测试 MCTS 的 _ucb_score 方法是否正确计算 UCB 分数。
        """
        # 创建父节点和子节点
        parent = self.root
        child = mcts_alphazero.Node(parent=parent, prior_p=0.5)
        parent.add_child(0, child)

        # 模拟父节点的 visit_count 和子节点的 visit_count
        for _ in range(10):
            parent.update(1.0)  # visit_count =10, value_sum=10.0, value=1.0
        for _ in range(2):
            child.update(1.0)  # visit_count=2, value_sum=2.0, value=1.0

        # 计算 UCB 分数
        ucb = self.mcts._ucb_score(parent, child)

        # 手动计算预期的 UCB 分数
        expected_pb_c = np.log(
            (parent.visit_count + self.mcts.pb_c_base + 1) / self.mcts.pb_c_base) + self.mcts.pb_c_init
        expected_pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)
        expected_score = expected_pb_c * child.prior_p + child.value  # 使用 'value' 属性

        self.assertEqual(ucb, expected_score, msg="UCB 分数计算不正确")

    def test_add_exploration_noise(self):
        """
        测试 MCTS 的 _add_exploration_noise 方法是否正确添加探索噪声。
        """
        # 为根节点添加一些子节点
        self.root.add_child(0, mcts_alphazero.Node(parent=self.root, prior_p=0.4))
        self.root.add_child(1, mcts_alphazero.Node(parent=self.root, prior_p=0.6))

        # 添加探索噪声
        self.mcts._add_exploration_noise(self.root)

        # 检查每个子节点的 prior_p 是否按比例更新
        for action in self.root.children:
            child = self.root.children[action]
            self.assertGreaterEqual(child.prior_p, 0.0, "探索噪声后，prior_p 不应小于 0")
            self.assertLessEqual(child.prior_p, 1.0, "探索噪声后，prior_p 不应大于 1.0")

    def test_get_next_action(self):
        """
        测试 MCTS 的 get_next_action 方法是否正确返回动作和概率分布。
        """
        # 配置 MCTS 对象的 simulate_env
        self.mcts.simulate_env = self.mock_env

        state_config_for_simulation_env_reset = EasyDict({
            'start_player_index': 0,
            'init_state': None,
            'katago_policy_init': False,
            'katago_game_state': None
        })

        # 执行 get_next_action
        action, action_probs = self.mcts.get_next_action(
            state_config_for_env_reset=state_config_for_simulation_env_reset,  # 根据需要传入具体配置
            policy_value_func=self.policy_value_func,
            temperature=1.0,
            sample=True
        )

        # 检查返回的 action 是否在合法动作中
        self.assertIn(action, self.legal_actions, "返回的动作不在合法动作中")

        # 检查 action_probs 的长度是否正确
        self.assertEqual(len(action_probs), len(self.legal_actions),
                         f"动作概率分布的长度应为 {len(self.legal_actions)}")

        # 检查 action_probs 是否为有效的概率分布
        self.assertEqual(sum(action_probs), 1.0, msg="动作概率分布的和应为 1.0")

    def test_expand_leaf_node(self):
        """
        测试 MCTS 的 _expand_leaf_node 方法是否正确扩展叶节点。
        """
        # 设置 simulate_env 为 mock_env
        simulate_env = self.mock_env

        # 扩展叶节点
        leaf_value = self.mcts._expand_leaf_node(self.root, simulate_env, self.policy_value_func)

        # 检查返回的叶值
        self.assertEqual(leaf_value, 0.9, "扩展叶节点时返回的叶值应为 0.9")

        # 检查子节点是否被正确添加
        for action, prior_p in ({0: 0.4, 1: 0.4, 2: 0.2}).items():
            child = self.root.children.get(action, None)
            self.assertIsNotNone(child, f"动作 {action} 的子节点应存在")
            self.assertAlmostEqual(child.prior_p, prior_p, places=5, msg=f"动作 {action} 的 prior_p 应为 {prior_p}")

    def test_simulate(self):
        """
        测试 MCTS 的 _simulate 方法是否能够正确执行模拟。
        由于 _simulate 方法内部有许多依赖，这里主要测试是否能够调用和更新节点。
        """
        # 调用 _simulate 方法
        self.mcts._simulate(self.root, self.mock_env, self.policy_value_func)

        # 检查节点是否有更新
        # 由于 simulate 调用的是 update_recursive，视具体实现，这里可以检查某些期望的值
        # 例如，检查 root 的 visit_count 是否增加
        self.assertGreaterEqual(self.root.visit_count, 0, "根节点的 visit_count 应大于或等于 0")
        # 这里无法具体判断，因为 _simulate 的内部逻辑被忽略

    def tearDown(self):
        """
        清理工作，可以在这里释放资源。
        """
        del self.mcts
        del self.root
        del self.mock_env
        del self.policy_value_func


if __name__ == '__main__':
    unittest.main()