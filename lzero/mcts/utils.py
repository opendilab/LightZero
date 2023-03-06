"""
The following code is adapted from https://github.com/YeWR/EfficientZero/core/utils.py
"""

import os
import numpy as np
import torch
from scipy.stats import entropy
from graphviz import Digraph
from typing import Any, List, Optional, Union
from dataclasses import dataclass


def to_torch_float_tensor(data_list: List, device):
    output_data_list = []
    for data in data_list:
        output_data_list.append(torch.from_numpy(data).to(device).float())
    return output_data_list


def to_detach_cpu_numpy(data_list: List):
    output_data_list = []
    for data in data_list:
        output_data_list.append(data.detach().cpu().numpy())
    return output_data_list


def ez_network_output_unpack(network_output):

    hidden_state = network_output.hidden_state  # shape:（batch_size, lstm_hidden_size, num_unroll_steps+1, num_unroll_steps+1）
    value_prefix = network_output.value_prefix  # shape: (batch_size, support_support_size), the ``value_prefix`` at the next ``num_unroll_steps`` step.
    reward_hidden_state = network_output.reward_hidden_state  # shape: {tuple: 2} -> (1, batch_size, 512)
    value = network_output.value  # shape: (batch_size, support_support_size)
    policy_logits = network_output.policy_logits  # shape: (batch_size, action_space_size)
    return hidden_state, value_prefix, reward_hidden_state, value, policy_logits

@dataclass
class BufferedData:
    data: Any
    index: str
    meta: dict


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


def select_action(visit_counts, temperature=1, deterministic=True):
    """
    Overview:
        select action from the root visit counts.
    Arguments:
        - temperature (:obj:`float`): the temperature for the distribution
        - deterministic (:obj:`bool`):  True -> select the argmax, False -> sample from the distribution
    """
    action_probs = [visit_count_i ** (1 / temperature) for visit_count_i in visit_counts]
    action_probs = [x / sum(action_probs) for x in action_probs]

    if deterministic:
        action_pos = np.argmax([v for v in visit_counts])
    else:
        action_pos = np.random.choice(len(visit_counts), p=action_probs)

    visit_count_distribution_entropy = entropy(action_probs, base=2)
    return action_pos, visit_count_distribution_entropy


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
    #   [B, S, W, H, C] -> [B, S x C, W, H]
    observation_lst = np.transpose(observation_lst, (0, 1, 4, 2, 3))

    shape = observation_lst.shape
    # 1, 4, 1, 8, 1 -> 1, 4*1, 8, 1
    observation_lst = observation_lst.reshape((shape[0], -1, shape[-2], shape[-1]))

    return observation_lst


def concat_output_value(output_lst):
    # concat the values of the model output list
    value_lst = []
    for output in output_lst:
        value_lst.append(output.value)

    value_lst = np.concatenate(value_lst)

    return value_lst


def concat_output(output_lst):
    # concat the model output
    value_lst, reward_lst, policy_logits_lst, hidden_state_lst = [], [], [], []
    reward_hidden_state_c_lst, reward_hidden_state_h_lst = [], []
    for output in output_lst:
        value_lst.append(output.value)
        try:
            reward_lst.append(output.value_prefix)
        except:
            reward_lst.append(output.reward)

        policy_logits_lst.append(output.policy_logits)
        hidden_state_lst.append(output.hidden_state)
        try:
            reward_hidden_state_c_lst.append(output.reward_hidden_state[0].squeeze(0))
            reward_hidden_state_h_lst.append(output.reward_hidden_state[1].squeeze(0))
        except:
            pass

    value_lst = np.concatenate(value_lst)
    reward_lst = np.concatenate(reward_lst)
    policy_logits_lst = np.concatenate(policy_logits_lst)
    hidden_state_lst = np.concatenate(hidden_state_lst)
    try:
        reward_hidden_state_c_lst = np.expand_dims(np.concatenate(reward_hidden_state_c_lst), axis=0)
        reward_hidden_state_h_lst = np.expand_dims(np.concatenate(reward_hidden_state_h_lst), axis=0)
        return value_lst, reward_lst, policy_logits_lst, hidden_state_lst, (
            reward_hidden_state_c_lst, reward_hidden_state_h_lst
        )
    except:
        return value_lst, reward_lst, policy_logits_lst, hidden_state_lst


def mask_nan(x: torch.Tensor) -> torch.Tensor:
    nan_part = torch.isnan(x)
    x[nan_part] = 0.
    return x


def get_max_entropy(action_shape: int) -> None:
    p = 1.0 / action_shape
    return -action_shape * p * np.log2(p)
