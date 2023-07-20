import sys
sys.path.append('/Users/puyuan/code/LightZero/lzero/mcts/ctree/alphazero/ctree_alphazero/build')

import node_alphazero
n = node_alphazero.Node()
n.update(5.0)
print(n.get_value())