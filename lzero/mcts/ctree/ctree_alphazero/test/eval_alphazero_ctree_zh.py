import numpy as np
import pytest
from easydict import EasyDict

from lzero.mcts.ctree.ctree_alphazero.test.eval_mcts_alphazero import find_and_add_to_sys_path

# Use the function to add the desired path to sys.path
find_and_add_to_sys_path("lzero/mcts/ctree/ctree_alphazero/build")

import mcts_alphazero


class MockEnv:
    """
    一个简单的模拟环境类，包含必要的属性和方法。
    用于替代真实环境以便进行单元测试。
    """

    def __init__(self):
        # 定义合法动作集合
        self.legal_actions = [0, 1, 2]
        # MCTS 模式：自我对弈模式
        self.battle_mode_in_simulation_env = "self_play_mode"
        # 当前玩家编号
        self.current_player = 1
        # 当前步数
        self.timestep = 0
        # 动作空间，定义动作数量
        self.action_space = type('action_space', (), {'n': 3})()

    def reset(self, start_player_index, init_state, katago_policy_init, katago_game_state):
        """
        模拟环境的 reset 方法。
        初始化环境的状态。
        """
        self.current_player = 1
        self.timestep = 0

    def step(self, action):
        """
        模拟环境的 step 方法。
        执行动作并切换玩家。
        """
        self.current_player = 2 if self.current_player == 1 else 1
        self.timestep += 1

    def get_done_winner(self):
        """
        模拟环境的 get_done_winner 方法。
        返回游戏是否结束以及获胜玩家。
        返回值中的第1项表示done。第二项表示winner, 其中1表示玩家1赢，2表示玩家2赢，-1表示平局或未结束。
        (False, -1) 表示游戏未结束。
        """
        return (False, -1)


def mock_policy_value_func(env):
    """
    一个模拟的 policy_value_func 函数。
    返回动作的概率分布和叶节点的估值。
    """
    return ({0: 0.4, 1: 0.4, 2: 0.2}, 0.9)


@pytest.fixture
def mcts_fixture():
    """
    使用 pytest fixture 初始化 MCTS 对象和测试环境。
    提供一个标准化的测试环境，避免重复代码。
    """
    mcts = mcts_alphazero.MCTS(
        max_moves=100,  # 最大步数
        num_simulations=100,  # 每次搜索的模拟次数
        pb_c_base=19652,  # UCB 分数计算的参数
        pb_c_init=1.25,  # UCB 分数计算的初始值
        root_dirichlet_alpha=0.3,  # 根节点的 Dirichlet 噪声参数
        root_noise_weight=0.25,  # 根节点噪声权重
        simulate_env=None  # 环境将在测试中设置
    )
    root = mcts_alphazero.Node()  # 创建根节点
    mock_env = MockEnv()  # 创建模拟环境
    policy_value_func = mock_policy_value_func  # 使用模拟的策略值函数
    legal_actions = [0, 1, 2]  # 定义合法动作集合
    return mcts, root, mock_env, policy_value_func, legal_actions


def test_node_initialization():
    """
    测试 Node 类的初始化是否正确。
    检查节点的默认值和基本属性。
    """
    root = mcts_alphazero.Node()

    assert root.parent is None, "根节点的父节点应为 None"
    assert root.prior_p == 1.0, "根节点的 prior_p 应默认为 1.0"
    assert root.visit_count == 0, "根节点的 visit_count 应默认为 0"
    assert root.value == 0.0, "根节点的初始 value 应为 0.0"
    assert root.is_leaf(), "新创建的根节点应为叶子节点"
    assert root.is_root(), "新创建的根节点应为根节点"


def test_node_update():
    """
    测试 Node 类的 update 方法是否正确更新 visit_count 和 value_sum。
    验证节点的访问计数和估值更新逻辑。
    """
    node = mcts_alphazero.Node()
    node.update(5.0)

    assert node.visit_count == 1, "更新一次后，visit_count 应为 1"
    assert node.value == 5.0, "更新一次后的 value 应为 5.0"

    node.update(3.0)
    assert node.visit_count == 2, "更新两次后，visit_count 应为 2"
    assert node.value == 4.0, "更新两次后的 value 应为 (5.0 + 3.0) / 2 = 4.0"


def test_node_recursive_update_self_play_mode():
    """
    测试 Node 类在 self_play_mode 下的递归更新。
    自我对弈模式中，父节点的值取反。
    """
    parent = mcts_alphazero.Node()
    child = mcts_alphazero.Node(parent=parent, prior_p=0.5)
    parent.add_child(1, child)

    child.update_recursive(1.0, "self_play_mode")

    assert child.visit_count == 1, "子节点的 visit_count 应为 1"
    assert child.value == 1.0, "子节点的 value 应为 1.0"
    assert parent.visit_count == 1, "父节点的 visit_count 应为 1"
    assert parent.value == -1.0, "父节点的 value 应为 -1.0"


def test_node_recursive_update_play_with_bot_mode():
    """
    测试 Node 类在 play_with_bot_mode 下的递归更新。
    人机对战模式中，父节点的值不取反。
    """
    parent = mcts_alphazero.Node()
    child = mcts_alphazero.Node(parent=parent, prior_p=0.5)
    parent.add_child(2, child)

    child.update_recursive(1.0, "play_with_bot_mode")

    assert child.visit_count == 1, "子节点的 visit_count 应为 1"
    assert child.value == 1.0, "子节点的 value 应为 1.0"
    assert parent.visit_count == 1, "父节点的 visit_count 应为 1"
    assert parent.value == 1.0, "父节点的 value 应为 1.0"


def test_node_add_child():
    """
    测试 Node 类的 add_child 方法是否正确添加子节点。
    验证子节点的添加逻辑和父子关系的正确性。
    """
    parent = mcts_alphazero.Node()
    child = mcts_alphazero.Node(parent=parent, prior_p=0.7)
    parent.add_child(3, child)

    assert 3 in parent.children, "动作为 3 的子节点应被添加到父节点的 children 中"
    assert parent.children[3] is child, "添加的子节点应与传入的 child 相同"
    assert not parent.is_leaf(), "添加子节点后，父节点不应为叶子节点"


def test_ucb_score(mcts_fixture):
    """
    测试 MCTS 的 _ucb_score 方法是否正确计算 UCB 分数。
    验证 UCB 公式的实现是否符合预期。
    """
    mcts, root, _, _, _ = mcts_fixture

    parent = root
    child = mcts_alphazero.Node(parent=parent, prior_p=0.5)
    parent.add_child(0, child)

    for _ in range(10):
        parent.update(1.0)
    for _ in range(2):
        child.update(1.0)

    ucb = mcts._ucb_score(parent, child)

    expected_pb_c = np.log(
        (parent.visit_count + mcts.pb_c_base + 1) / mcts.pb_c_base) + mcts.pb_c_init
    expected_pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)
    expected_score = expected_pb_c * child.prior_p + child.value

    assert ucb == expected_score, "UCB 分数计算不正确"


def test_add_exploration_noise(mcts_fixture):
    """
    测试 MCTS 的 _add_exploration_noise 方法是否正确添加探索噪声。
    验证根节点的子节点在加入噪声后的 prior_p 是否合理。
    """
    mcts, root, _, _, _ = mcts_fixture

    root.add_child(0, mcts_alphazero.Node(parent=root, prior_p=0.4))
    root.add_child(1, mcts_alphazero.Node(parent=root, prior_p=0.6))

    mcts._add_exploration_noise(root)

    for action in root.children:
        child = root.children[action]
        assert 0.0 <= child.prior_p <= 1.0, "探索噪声后，prior_p 超出 [0, 1] 范围"


def test_get_next_action(mcts_fixture):
    """
    测试 MCTS 的 get_next_action 方法是否正确返回动作和概率分布。
    检查模拟搜索过程和最终动作选择是否符合预期。
    """
    mcts, root, mock_env, policy_value_func, legal_actions = mcts_fixture
    mcts.simulate_env = mock_env

    state_config_for_simulation_env_reset = EasyDict({
        'start_player_index': 0,
        'init_state': None,
        'katago_policy_init': False,
        'katago_game_state': None
    })

    # 执行 get_next_action
    action, action_probs, root = mcts.get_next_action(
        state_config_for_env_reset=state_config_for_simulation_env_reset,
        policy_value_func=policy_value_func,
        temperature=1.0,
        sample=False,
    )

    # 检查根节点访问次数是否为 10（num_simulations）
    assert root.visit_count == 100, f"根节点的访问次数应为 10，但得到 {root.visit_count}"

    # 确保返回的动作是访问次数最多的动作
    max_visits_action = np.argmax(action_probs)
    assert action == max_visits_action, f"返回的动作应为访问次数最多的动作 ({max_visits_action})"

    # 验证动作概率分布
    assert len(action_probs) == len(mock_env.legal_actions), "动作概率分布的长度应与合法动作数一致"
    assert sum(action_probs) == pytest.approx(1.0), "动作概率分布的和应为 1.0"

    # 打印结果（仅在调试时需要）
    print(f"动作: {action}, 动作概率分布: {action_probs}")