"""
reference: https://mathspp.com/blog/minimax-algorithm-and-alpha-beta-pruning
"""


class Tree:

    def __init__(self, children):
        self.children = children


class Terminal(Tree):

    def __init__(self, value):
        # A terminal state is a ctree with no children:
        super().__init__([])
        self.value = value


def minimax(tree, maximising_player):
    if isinstance(tree, Terminal):
        return tree.value

    val, func = (float("-inf"), max) if maximising_player else (float("+inf"), min)
    for subtree in tree.children:
        val = func(minimax(subtree, not maximising_player), val)
    return val


# v2
# def minimax(ctree, maximising_player):
#     if isinstance(ctree, Terminal):
#         return ctree.value
#
#     v, f = (float("-inf"), max) if maximising_player else (float("+inf"), min)
#     return f((minimax(sub, not maximising_player) for sub in ctree.children), default=v)

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

print(minimax(tree, True))
print(minimax(tree, False))
