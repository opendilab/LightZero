import pickle
import sys
sys.path.append('..')
import os
from reasoners.visualization import visualize
from reasoners.visualization.tree_snapshot import NodeData
from reasoners.algorithm.mcts import MCTSNode
mcts_result = pickle.load(open('/mnt/afs/niuyazhe/code/llm-reasoners/logs/gsm8k_MCTS/04242024-170944/algo_output/1.pkl', 'rb'))
print(mcts_result.terminal_state)
# import pdb; pdb.set_trace()
def gsm_node_data_factory(x: MCTSNode):
    if not x.state:
        return {}
    return {"question": x.state[-1].sub_question, "answer": x.state[-1].sub_answer}
visualize(mcts_result, node_data_factory=gsm_node_data_factory)