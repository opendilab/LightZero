"""第二章实践案例：基于纯搜索的井字棋智能体（独立单文件，无需安装 LightZero）。

直接运行进入菜单：
    python mcts_tictactoe_zh.py
人机对战（人类执 X 先手，MCTS 执 O 后手）：
    python mcts_tictactoe_zh.py --mode play --simulations 1000 --seed 0
评估 MCTS（默认执 X）对随机策略，并保存胜/平/负率曲线：
    python mcts_tictactoe_zh.py --mode evaluate --num-games 100 --seed 0 --no-show
    python mcts_tictactoe_zh.py --mode evaluate --mcts-player O --no-show

实现对应 Game、Node、mcts、play_match、evaluate_and_plot 五个教学接口。
价值 W 统一从 X 的视角记录（X 胜 +1，O 胜 -1，平局 0）；选择时转换到
当前落子方的视角，因此 X 最大化 Q，O 最小化 Q，双方都探索访问较少的动作。
有限预算与有限样本的评估不保证胜率单调上升，也不能证明已达到完美策略。
"""

import argparse
import math
import platform
import random


class Game:
    """井字棋环境：动作索引为 0–8，玩家 X 为 1，玩家 O 为 -1。"""

    WINNING_LINES = (
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),  # 行
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),  # 列
        (0, 4, 8),
        (2, 4, 6),  # 对角线
    )

    def __init__(self):
        self.board = [' ' for _ in range(9)]
        self.current_player = 1

    def get_current_player(self):
        return self.current_player

    def get_legal_moves(self):
        # 即使棋盘还有空格，已有胜者时也不能继续落子。
        if self.is_game_over()[0]:
            return []
        return [i for i in range(9) if self.board[i] == ' ']

    def make_move(self, move):
        if isinstance(move, bool) or not isinstance(move, int) or not 0 <= move < 9:
            raise ValueError("走法必须是 0–8 之间的整数索引")
        if self.is_game_over()[0]:
            raise ValueError("游戏已经结束，不能继续落子")
        if self.board[move] != ' ':
            raise ValueError(f"无效的走法：{move}，该位置已有棋子")
        self.board[move] = 'X' if self.current_player == 1 else 'O'
        self.current_player *= -1

    def is_game_over(self):
        """返回 (是否结束, 胜者)：1=X，-1=O，0=平局，None=未结束。"""
        for a, b, c in self.WINNING_LINES:
            if self.board[a] != ' ' and self.board[a] == self.board[b] == self.board[c]:
                return True, 1 if self.board[a] == 'X' else -1
        if ' ' not in self.board:
            return True, 0
        return False, None

    def clone(self):
        cloned_game = Game()
        cloned_game.board = self.board.copy()
        cloned_game.current_player = self.current_player
        return cloned_game

    def print_board(self):
        print("当前棋盘（数字表示可落子位置）：")
        cells = [mark if mark != ' ' else str(i + 1) for i, mark in enumerate(self.board)]
        print("\n---------\n".join(" | ".join(cells[i:i + 3]) for i in (0, 3, 6)))
        print()


class Node:
    """MCTS 树节点；visits 为 N，value 为 X 视角的累计回报 W。"""

    def __init__(self, game, parent=None, move_from_parent=None):
        self.game = game
        self.parent = parent
        self.move_from_parent = move_from_parent
        self.children = {}
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.children) == len(self.game.get_legal_moves())

    def best_child(self, c_param=math.sqrt(2), rng=None):
        """UCT = 当前玩家视角的 Q + c * sqrt(ln N_parent / N_child)。

        c 默认 sqrt(2)，根号内不再重复乘 2。探索奖励始终为正，O 回合只
        反转利用项。未访问的子节点优先；分数相同时随机选择以避免位置偏置。
        """
        if not self.children:
            return None
        rng = rng if rng is not None else random
        unvisited = [child for child in self.children.values() if child.visits == 0]
        if unvisited:
            return rng.choice(unvisited)

        best_score = -math.inf
        candidates = []
        for child in self.children.values():
            exploitation = self.game.current_player * child.value / child.visits
            exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                candidates = [child]
            elif score == best_score:
                candidates.append(child)
        return rng.choice(candidates)


def _require_positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须是正整数")


def mcts(root_game, simulations=1000, c_param=math.sqrt(2), rng=None):
    """运行指定次数的四步搜索，返回访问次数最多的动作；终局返回 None。

    接收 Game 而非 Node，与书中的调用方式一致。搜索和随机 rollout 均在
    副本上进行，不修改实战棋盘。simulations 必须为正整数；可传入
    random.Random(seed) 复现实验，省略 rng 时使用标准库全局随机数生成器。
    """
    _require_positive_int(simulations, 'simulations')
    if not math.isfinite(c_param) or c_param < 0:
        raise ValueError("c_param 必须是有限的非负数")
    if root_game.is_game_over()[0]:
        return None
    rng = rng if rng is not None else random
    root_node = Node(root_game.clone())

    for _ in range(simulations):
        node = root_node
        game = root_game.clone()

        # 1. 选择：沿完全扩展的节点按 UCT 向下走，遇到终局立即停止。
        while not game.is_game_over()[0] and node.is_fully_expanded():
            node = node.best_child(c_param=c_param, rng=rng)
            game.make_move(node.move_from_parent)

        # 2. 扩展：每轮只新增一个子节点，并从这个子节点开始评估。
        if not game.is_game_over()[0]:
            unexpanded_moves = [move for move in game.get_legal_moves() if move not in node.children]
            move = rng.choice(unexpanded_moves)
            game.make_move(move)
            child = Node(game.clone(), parent=node, move_from_parent=move)
            node.children[move] = child
            node = child

        # 3. 模拟：双方均匀随机落子直至终局；已终止的节点直接使用胜负结果。
        sim_game = game.clone()
        while not sim_game.is_game_over()[0]:
            sim_game.make_move(rng.choice(sim_game.get_legal_moves()))
        _, winner = sim_game.is_game_over()

        # 4. 回溯：全树统一存储 X 视角的回报；玩家视角转换只在选择时进行。
        while node is not None:
            node.visits += 1
            node.value += winner
            node = node.parent

    # 正预算、非终局保证至少扩展一个子节点，1 次模拟也能返回合法动作。
    most_visits = max(child.visits for child in root_node.children.values())
    candidates = [move for move, child in root_node.children.items() if child.visits == most_visits]
    return rng.choice(candidates)


def human_move(game):
    """读取 1–9 的人类输入并落子；非法输入不会改变棋盘。"""
    if game.is_game_over()[0]:
        return
    while True:
        try:
            move = int(input("请输入你的走法（1–9）：")) - 1
        except ValueError:
            print("无效的输入，请输入 1–9 之间的整数。")
            continue
        if move not in game.get_legal_moves():
            print("无效的走法，请选择棋盘上仍显示数字的位置（1–9）。")
            continue
        game.make_move(move)
        return


def bot_move(game, simulations=1000, rng=None):
    move = mcts(game, simulations=simulations, rng=rng)
    if move is not None:
        game.make_move(move)
        print(f"Bot 选择了走法：{move + 1}")


def play_game(simulations=1000, rng=None):
    """人类执 X 先手，MCTS 执 O 后手。"""
    _require_positive_int(simulations, 'simulations')
    game = Game()
    print("你执 X 先手，MCTS Bot 执 O 后手。按 Ctrl+C 退出。")
    game.print_board()
    while not game.is_game_over()[0]:
        if game.current_player == 1:
            human_move(game)
        else:
            bot_move(game, simulations=simulations, rng=rng)
        game.print_board()
    _, winner = game.is_game_over()
    print({1: "玩家（X）获胜！", -1: "MCTS Bot（O）获胜！", 0: "平局！"}[winner])


def play_match(mcts_simulations, num_games=100, mcts_player=1, rng=None):
    """MCTS 对随机策略，返回 (MCTS 胜局数, 随机策略胜局数, 平局数)。

    默认 MCTS 执 X 先手，与书中评估一致；mcts_player=-1 可单独评估后手。
    """
    _require_positive_int(mcts_simulations, 'mcts_simulations')
    _require_positive_int(num_games, 'num_games')
    if isinstance(mcts_player, bool) or mcts_player not in (1, -1):
        raise ValueError("mcts_player 必须是 1（X）或 -1（O）")
    rng = rng if rng is not None else random
    mcts_wins = random_wins = draws = 0
    for _ in range(num_games):
        game = Game()
        while not game.is_game_over()[0]:
            if game.current_player == mcts_player:
                move = mcts(game, simulations=mcts_simulations, rng=rng)
            else:
                move = rng.choice(game.get_legal_moves())
            game.make_move(move)
        _, winner = game.is_game_over()
        if winner == mcts_player:
            mcts_wins += 1
        elif winner == -mcts_player:
            random_wins += 1
        else:
            draws += 1
    return mcts_wins, random_wins, draws


def evaluate_and_plot(
    sim_counts=None,
    num_games=100,
    mcts_player=1,
    seed=0,
    output_path='mcts_performance.png',
    show=True,
):
    """评估各搜索预算，保存胜/平/负率曲线，并返回各预算的原始计数。

    中文字体按系统自动选择；没有可用中文字体时使用英文图注。show=False
    可在无图形界面的环境中运行。曲线为样本频率，不能据此保证零失误。
    """
    sim_counts = [1, 5, 10, 20, 50, 100, 200, 500] if sim_counts is None else list(sim_counts)
    if not sim_counts:
        raise ValueError("sim_counts 不能为空")
    for count in sim_counts:
        _require_positive_int(count, 'sim_counts 中的模拟次数')
    _require_positive_int(num_games, 'num_games')
    if isinstance(mcts_player, bool) or mcts_player not in (1, -1):
        raise ValueError("mcts_player 必须是 1（X）或 -1（O）")

    import matplotlib
    from matplotlib import font_manager
    if show:
        import matplotlib.pyplot as plt
    else:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

    preferred_fonts = {
        'Darwin': ['PingFang SC', 'Heiti TC', 'Arial Unicode MS'],
        'Windows': ['Microsoft YaHei', 'SimHei'],
        'Linux': ['Noto Sans CJK SC', 'WenQuanYi Zen Hei'],
    }
    candidates = preferred_fonts.get(platform.system(), []) + [
        'Noto Sans CJK SC',
        'Microsoft YaHei',
        'SimHei',
        'Arial Unicode MS',
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in candidates if name in available_fonts), None)
    player = 'X' if mcts_player == 1 else 'O'
    print(f"MCTS 执 {player}（{'先手' if mcts_player == 1 else '后手'}），每个预算 {num_games} 局，seed={seed}")
    print("模拟次数\tMCTS 胜率\t平局率\t随机策略胜率")
    results = []
    for simulations in sim_counts:
        # 各预算重新使用给定种子，确保结果不依赖预算列表的排列顺序。
        wins, losses, draws = play_match(simulations, num_games, mcts_player, rng=random.Random(seed))
        results.append(dict(simulations=simulations, wins=wins, losses=losses, draws=draws))
        print(f"{simulations}\t\t{wins / num_games:.1%}\t\t{draws / num_games:.1%}\t{losses / num_games:.1%}")

    # rc_context 避免字体配置影响导入本模块的其他绘图代码。
    settings = {'axes.unicode_minus': False}
    if font:
        settings['font.sans-serif'] = [font]
    with matplotlib.rc_context(settings):
        if show:
            fig, ax = plt.subplots(figsize=(9, 6))
        else:
            # 独立画布不会切换全局后端，也不会关闭调用方已经打开的图。
            fig = Figure(figsize=(9, 6))
            FigureCanvasAgg(fig)
            ax = fig.subplots()
        try:
            labels = ('MCTS 胜率', '平局率', '随机策略胜率') if font else ('MCTS win', 'Draw', 'Random win')
            for key, label in zip(('wins', 'draws', 'losses'), labels):
                ax.plot(sim_counts, [row[key] / num_games for row in results], marker='o', label=label)
            ax.set_xscale('log')
            ax.set_xticks(sim_counts)
            ax.set_xticklabels([str(count) for count in sim_counts])
            ax.set_ylim(0, 1.05)
            ax.set_xlabel('每步模拟次数（对数坐标）' if font else 'Simulations per move (log scale)')
            ax.set_ylabel('对局比例' if font else 'Fraction of games')
            title = f'MCTS（{player}）对随机策略，每个预算 {num_games} 局' if font else (
                f'MCTS ({player}) vs random, {num_games} games per budget'
            )
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_path, dpi=200)
            print(f"性能曲线已保存至：{output_path}")
            print("注：有限样本下胜率可能波动，观测到零负局不等于已证明完美策略。")
            if show:
                plt.show()
        finally:
            if show:
                plt.close(fig)
    return results


def _positive_int_arg(text):
    try:
        value = int(text)
        _require_positive_int(value, '参数')
        return value
    except ValueError as error:
        raise argparse.ArgumentTypeError('请输入正整数') from error


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=('play', 'evaluate'), help='省略时显示交互菜单')
    parser.add_argument('--simulations', type=_positive_int_arg, default=1000, help='人机对战每步模拟次数')
    parser.add_argument('--sim-counts', type=_positive_int_arg, nargs='+', help='评估使用的模拟次数列表')
    parser.add_argument('--num-games', type=_positive_int_arg, default=100, help='每个评估预算的对局数')
    parser.add_argument('--mcts-player', choices=('X', 'O'), default='X', help='评估中 MCTS 的执棋方')
    parser.add_argument('--seed', type=int, default=0, help='随机种子（默认 0）')
    parser.add_argument('--output', default='mcts_performance.png', help='性能曲线保存路径')
    parser.add_argument('--no-show', action='store_true', help='只保存图片，不弹出窗口')
    args = parser.parse_args()
    try:
        mode = args.mode
        if mode is None:
            print("请选择模式：\n1. 人机对战 (Play)\n2. 算法评估 (Evaluate)")
            while True:
                choice = input("请输入选项 (1 或 2，默认 1)：").strip()
                if choice in ('', '1', '2'):
                    break
                print("无效的选项，请输入 1 或 2。")
            mode = 'evaluate' if choice == '2' else 'play'
        if mode == 'play':
            play_game(simulations=args.simulations, rng=random.Random(args.seed))
        else:
            evaluate_and_plot(
                sim_counts=args.sim_counts,
                num_games=args.num_games,
                mcts_player=1 if args.mcts_player == 'X' else -1,
                seed=args.seed,
                output_path=args.output,
                show=not args.no_show,
            )
    except (EOFError, KeyboardInterrupt):
        print("\n已退出。")


if __name__ == '__main__':
    main()
