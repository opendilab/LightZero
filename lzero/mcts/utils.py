import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from graphviz import Digraph


@dataclass
class BufferedData:
    data: Any
    index: str
    meta: dict


def get_augmented_data(board_size, play_data):
    """
    Overview:
        augment the data set by rotation and flipping
    Arguments:
        play_data: [(state, mcts_prob, winner_z), ..., ...]
    """
    extend_data = []
    for data in play_data:
        state = data['state']
        mcts_prob = data['mcts_prob']
        winner = data['winner']
        for i in [1, 2, 3, 4]:
            # rotate counterclockwise
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(np.flipud(mcts_prob.reshape(board_size, board_size)), i)
            extend_data.append(
                {
                    'state': equi_state,
                    'mcts_prob': np.flipud(equi_mcts_prob).flatten(),
                    'winner': winner
                }
            )
            # flip horizontally
            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
            extend_data.append(
                {
                    'state': equi_state,
                    'mcts_prob': np.flipud(equi_mcts_prob).flatten(),
                    'winner': winner
                }
            )
    return extend_data


def prepare_observation_list(observation_lst):
    """
    Overview:
        Prepare the observations to satisfy the input format of torch
        [B, S, W, H, C] -> [B, S x C, W, H]
        batch, stack num, width, height, channel
    """
    # B, S, W, H, C
    observation_lst = np.array(observation_lst)
    # 1, 4, 8, 1, 1 -> 1, 4, 1, 8, 1
    #   [B, S, W, H, C] -> [B, S, C, W, H]
    observation_lst = np.transpose(observation_lst, (0, 1, 4, 2, 3))

    # 1, 4, 8, 1, 1 -> 1, 4, 1, 8, 1
    # observation_lst = np.moveaxis(observation_lst, -1, 2)

    shape = observation_lst.shape
    # 1, 4, 1, 8, 1 -> 1, 4*1, 8, 1
    #  [B, S, C, W, H] -> [B, S*C, W, H]

    observation_lst = observation_lst.reshape((shape[0], -1, shape[-2], shape[-1]))

    return observation_lst


def mask_nan(x: torch.Tensor) -> torch.Tensor:
    nan_part = torch.isnan(x)
    x[nan_part] = 0.
    return x


def get_max_entropy(action_shape: int) -> None:
    p = 1.0 / action_shape
    return -action_shape * p * np.log2(p)


def obtain_tree_topology(root, to_play=0):
    node_stack = []
    edge_topology_list = []
    node_topology_list = []
    node_id_list = []
    node_stack.append(root)
    while len(node_stack) > 0:
        node = node_stack[-1]
        node_stack.pop()
        node_dict = {}
        node_dict['node_id'] = node.hidden_state_index_x
        node_dict['visit_count'] = node.visit_count
        node_dict['policy_prior'] = node.prior
        node_dict['value'] = node.value
        node_topology_list.append(node_dict)

        node_id_list.append(node.hidden_state_index_x)
        for a in node.legal_actions:
            child = node.get_child(a)
            if child.expanded:
                child.parent_hidden_state_index_x = node.hidden_state_index_x
                edge_dict = {}
                edge_dict['parent_id'] = node.hidden_state_index_x
                edge_dict['child_id'] = child.hidden_state_index_x
                edge_topology_list.append(edge_dict)
                node_stack.append(child)
    return edge_topology_list, node_id_list, node_topology_list


def plot_simulation_graph(env_root, current_step, graph_directory=None):
    edge_topology_list, node_id_list, node_topology_list = obtain_tree_topology(env_root)
    dot = Digraph(comment='this is direction')
    for node_topology in node_topology_list:
        node_name = str(node_topology['node_id'])
        label = f"node_id: {node_topology['node_id']}, \l visit_count: {node_topology['visit_count']}, \l policy_prior: {round(node_topology['policy_prior'], 4)}, \l value: {round(node_topology['value'], 4)}"
        dot.node(node_name, label=label)
    for edge_topology in edge_topology_list:
        parent_id = str(edge_topology['parent_id'])
        child_id = str(edge_topology['child_id'])
        label = parent_id + '-' + child_id
        dot.edge(parent_id, child_id, label=label)
    if graph_directory is None:
        graph_directory = './data_visualize/'
    if not os.path.exists(graph_directory):
        os.makedirs(graph_directory)
    graph_path = graph_directory + 'simulation_visualize_' + str(current_step) + 'step.gv'
    dot.format = 'png'
    dot.render(graph_path, view=False)

