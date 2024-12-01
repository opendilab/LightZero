import math
import random

# 游戏类，表示井字棋的状态
class Game:
    def __init__(self):
        # 初始化棋盘，使用列表表示9个格子，初始为空格
        self.board = [' ' for _ in range(9)]
        # 当前玩家，1表示玩家1（X），-1表示玩家2（O）
        self.current_player = 1

    def get_current_player(self):
        # 返回当前玩家
        return self.current_player

    def get_legal_moves(self):
        # 返回所有合法的走法，即棋盘中为空的位置的索引
        return [i for i in range(9) if self.board[i] == ' ']

    def make_move(self, move):
        # 执行走法，如果目标位置不为空则抛出异常
        if self.board[move] != ' ':
            raise ValueError("无效的走法")
        # 根据当前玩家标记棋子
        self.board[move] = 'X' if self.current_player == 1 else 'O'
        # 切换玩家
        self.current_player *= -1

    def is_game_over(self):
        # 定义所有可能的获胜线路
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 行
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 列
            [0, 4, 8], [2, 4, 6]              # 对角线
        ]
        # 检查是否有玩家获胜
        for line in lines:
            a, b, c = line
            if self.board[a] == self.board[b] == self.board[c] and self.board[a] != ' ':
                return True, self.board[a]  # 返回游戏结束和胜利者
        # 检查是否平局
        if ' ' not in self.board:
            return True, 0  # 平局
        # 游戏未结束
        return False, None

    def clone(self):
        # 克隆当前游戏状态，用于模拟
        cloned_game = Game()
        cloned_game.board = self.board.copy()
        cloned_game.current_player = self.current_player
        return cloned_game

    def print_board(self):
        # 打印当前棋盘状态
        print("当前棋盘状态：")
        print(f"{self.board[0]} | {self.board[1]} | {self.board[2]}")
        print("---------")
        print(f"{self.board[3]} | {self.board[4]} | {self.board[5]}")
        print("---------")
        print(f"{self.board[6]} | {self.board[7]} | {self.board[8]}")
        print()

# 节点类，用于MCTS的树结构
class Node:
    def __init__(self, game, parent=None):
        self.game = game          # 当前游戏状态
        self.parent = parent      # 父节点
        self.children = {}        # 子节点，键为走法，值为节点
        self.visits = 0           # 访问次数
        self.value = 0.0          # 累计奖励值

# 选择子节点的策略（使用UCB1公式）
def select_child(self):
    best_score = -float('inf')
    best_move = None
    best_child = None
    for move, child in self.children.items():
        if child.visits == 0:
            score = float('inf')  # 未被访问过的节点优先选择
        else:
            exploitation = child.value / child.visits  # 利用
            exploration = math.sqrt(2 * math.log(self.visits) / child.visits)  # 探索
            score = exploitation + exploration
        if score > best_score:
            best_score = score
            best_move = move
            best_child = child
    return best_move, best_child

# 为节点扩展所有可能的子节点
def expand(self, game):
    legal_moves = game.get_legal_moves()
    for move in legal_moves:
        new_game = game.clone()
        new_game.make_move(move)
        child_node = Node(new_game, parent=self)
        self.children[move] = child_node

# 模拟游戏直到结束，返回游戏结果
def simulate(self):
    game = self.game.clone()
    while True:
        is_over, result = game.is_game_over()
        if is_over:
            break
        legal_moves = game.get_legal_moves()
        move = random.choice(legal_moves)  # 随机选择走法
        game.make_move(move)
    return result  # 返回 'X', 'O' 或 0

# 将上述函数绑定到Node类
Node.select_child = select_child
Node.expand = expand
Node.simulate = simulate

# MCTS算法实现
def mcts(root_node, simulations=1000):
    for _ in range(simulations):
        node = root_node
        game = node.game.clone()
        # 选择阶段
        while node.children and not game.is_game_over()[0]:
            move, node = node.select_child()
            game.make_move(move)
        # 扩展阶段
        if not node.children and not game.is_game_over()[0]:
            node.expand(game)
        # 模拟阶段
        if not game.is_game_over()[0]:
            result = node.simulate()
        else:
            _, result = game.is_game_over()
        # 回溯阶段
        while node:
            node.visits += 1
            if result == 'X':
                node.value += 1.0 if node.game.current_player == -1 else -1.0
            elif result == 'O':
                node.value += -1.0 if node.game.current_player == -1 else 1.0
            else:
                node.value += 0.0  # 平局
            node = node.parent
    # 选择访问次数最多的走法作为最佳走法
    best_move = max(root_node.children.keys(), key=lambda move: root_node.children[move].visits)
    return best_move

# 人类玩家的走法输入
def human_move(game):
    while True:
        try:
            move_input = input("请输入你的走法（1-9）：")
            move = int(move_input) - 1  # 转换为索引
            if move not in game.get_legal_moves():
                print("无效的走法，请重新输入。")
            else:
                game.make_move(move)
                break
        except ValueError:
            print("无效的输入，请输入一个数字。")

# 机器人玩家的走法（使用MCTS）
def bot_move(game):
    root_node = Node(game.clone())
    best_move = mcts(root_node, simulations=50)  # 可以根据性能调整模拟次数
    game.make_move(best_move)
    print(f"Bot选择了走法：{best_move + 1}")

# 主函数，游戏循环
def main():
    game = Game()
    game.print_board()

    while not game.is_game_over()[0]:
        if game.get_current_player() == 1:
            human_move(game)  # 玩家1（X）走法
        else:
            bot_move(game)    # 玩家2（O）走法
        game.print_board()
        is_over, result = game.is_game_over()
        if is_over:
            if result == 'X':
                print("玩家1（X）获胜！")
            elif result == 'O':
                print("玩家2（O）获胜！")
            else:
                print("平局！")
            break

# 运行主函数
if __name__ == "__main__":
    main()