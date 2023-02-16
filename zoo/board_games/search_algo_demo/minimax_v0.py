"""
reference: https://mathspp.com/blog/minimax-algorithm-and-alpha-beta-pruning
"""


class Choice:

    def __init__(self, left, right):
        self.left = left
        self.right = right


class Terminal:

    def __init__(self, value):
        self.value = value


tree = Choice(Choice(
    Terminal(9),
    Terminal(5),
), Choice(
    Terminal(-3),
    Terminal(-2),
))


def minimax(tree, maximising_player):
    if isinstance(tree, Choice):
        lv = minimax(tree.left, not maximising_player)
        rv = minimax(tree.right, not maximising_player)
        if maximising_player:
            return max(lv, rv)
        else:
            return min(lv, rv)
    else:
        return tree.value


print(minimax(tree, True))
print(minimax(tree, False))
