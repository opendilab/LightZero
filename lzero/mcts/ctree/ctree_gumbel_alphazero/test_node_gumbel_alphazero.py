import sys
sys.path.append('/Users/puyuan/code/LightZero/lzero/mcts/ctree/ctree_gumbel_alphazero/build')

import mcts_gumbel_alphazero
n = mcts_gumbel_alphazero.Node()
print(n.is_leaf())
print(n.update(5.0))
print(n.value())