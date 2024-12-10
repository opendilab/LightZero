import sys
sys.path.append('/Users/puyuan/code/LightZero/lzero/mcts/ctree/ctree_alphazero/build')

import mcts_alphazero
node = mcts_alphazero.Node()
print(node)
print(node.is_leaf())
print(node.update(5.0))
print(node.value)
