"""
reference: https://mathspp.com/blog/minimax-algorithm-and-alpha-beta-pruning
"""


class Tree:

    def __init__(self, children):
        self.children = children

    def __str__(self):
        return f"Tree({', '.join(str(sub) for sub in self.children)})"


class Terminal(Tree):

    def __init__(self, value):
        super().__init__([])
        self.value = value

    def __str__(self):
        return f"T({self.value})"


def minimax(tree, maximising_player):
    if isinstance(tree, Terminal):
        return tree.value

    val, func = (float("-inf"), max) if maximising_player else (float("+inf"), min)
    for subtree in tree.children:
        val = func(minimax(subtree, not maximising_player), val)
    return val


def pruning(tree, maximising_player, alpha=float("-inf"), beta=float("+inf"), first_level=False):
    print(tree)

    if isinstance(tree, Terminal):
        return tree.value

    val, func = (float("-inf"), max) if maximising_player else (float("+inf"), min)
    for subtree in tree.children:
        val = func(pruning(subtree, not maximising_player, alpha, beta, first_level=False), val)
        if maximising_player:
            if val > alpha:
                best_subtree = subtree
            alpha = max(alpha, val)
        else:
            if val < beta:
                best_subtree = subtree
            beta = min(beta, val)

        # if (maximising_player and val >= beta) or (not maximising_player and val <= alpha):
        #     break
        if (maximising_player and val >= beta):
            break
        if (not maximising_player and val <= alpha):
            break
    if first_level is True:
        return val, best_subtree
    else:
        return val


tree = Tree(
    [
        Tree(
            [
                Tree([
                    Terminal(3),
                    Terminal(4),
                ]),
                Tree([
                    Terminal(8),
                    Tree([
                        Terminal(-2),
                        Terminal(10),
                    ]),
                    Terminal(5),
                ])
            ]
        ),
        Terminal(7),
    ]
)

val, best_subtree = pruning(tree, True, first_level=True)
print(val)  # 7
print(best_subtree.value)  # 7
# print(pruning(ctree, False))  # 5
