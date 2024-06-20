import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from graphviz import Digraph


def generate_random_actions_discrete(
    num_actions: int, action_space_size: int, num_of_sampled_actions: int, reshape=False
):
    """
    Overview:
        Generate a list of random actions.
    Arguments:
        - num_actions (:obj:`int`): The number of actions to generate.
        - action_space_size (:obj:`int`): The size of the action space.
        - num_of_sampled_actions (:obj:`int`): The number of sampled actions.
        - reshape (:obj:`bool`): Whether to reshape the actions.
    Returns:
        A list of random actions.
    """
    actions = [np.random.randint(0, action_space_size, num_of_sampled_actions).reshape(-1) for _ in range(num_actions)]

    # If num_of_sampled_actions == 1, flatten the actions to a list of numbers
    if num_of_sampled_actions == 1:
        actions = [action[0] for action in actions]

    # Reshape actions if needed
    if reshape and num_of_sampled_actions > 1:
        actions = [action.reshape(num_of_sampled_actions, 1) for action in actions]

    return actions


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


def prepare_observation(observation_list, model_type='conv'):
    """
    Prepare the observations to satisfy the input format of the model.

    For model_type='conv':
        [B, S, C, W, H] -> [B, S x C, W, H]
        where B is batch size, S is stack num, W is width, H is height, and C is the number of channels.

    For model_type='mlp':
        [B, S, O] -> [B, S x O]
        where B is batch size, S is stack num, O is obs shape.

    Arguments:
        - observation_list (List): list of observations.
        - model_type (str): type of the model. (default is 'conv')

    Returns:
        - np.ndarray: Reshaped array of observations.
    """
    assert model_type in [
        'conv', 'mlp', 'rgcn', 'mlp_md'
    ], "model_type must be either 'conv', 'mlp', 'rgcn' or 'mlp_md'"
    observation_array = np.array(observation_list)
    batch_size = observation_array.shape[0]

    if model_type == 'conv':
        if observation_array.ndim == 3:
            # Add a channel dimension if it's missing
            observation_array = observation_array[..., np.newaxis]
        elif observation_array.ndim == 5:
            # Reshape to [B, S*C, W, H]
            _, stack_num, channels, width, height = observation_array.shape
            observation_array = observation_array.reshape(batch_size, stack_num * channels, width, height)

    elif model_type == 'mlp' or model_type == 'mlp_md':
        if observation_array.ndim == 3:
            # Flatten the last two dimensions
            observation_array = observation_array.reshape(batch_size, -1)
        else:
            raise ValueError("For 'mlp' model_type, the observation must have 3 dimensions [B, S, O]")

    elif model_type == 'rgcn':
        if observation_array.ndim == 4:
            # TODO(rjy): strage process
            # observation_array should be reshaped to [B, S*M, O], where M is the agent number
            # now observation_array.shape = [B, S, M, O]
            observation_array = observation_array.reshape(batch_size, -1, observation_array.shape[-1])
        elif observation_array.ndim == 3:
            # Flatten the last two dimensions
            observation_array = observation_array.reshape(batch_size, -1)
        else:
            raise ValueError(
                "For 'rgcn' model_type, the observation must have 3 dimensions [B, S, O] or 4 dimensions [B, S, M, O]"
            )

    return observation_array


def obtain_tree_topology(root, to_play=-1):
    node_stack = []
    edge_topology_list = []
    node_topology_list = []
    node_id_list = []
    node_stack.append(root)
    while len(node_stack) > 0:
        node = node_stack[-1]
        node_stack.pop()
        node_dict = {}
        node_dict['node_id'] = node.simulation_index
        node_dict['visit_count'] = node.visit_count
        node_dict['policy_prior'] = node.prior
        node_dict['value'] = node.value
        node_topology_list.append(node_dict)

        node_id_list.append(node.simulation_index)
        for a in node.legal_actions:
            child = node.get_child(a)
            if child.expanded:
                child.parent_simulation_index = node.simulation_index
                edge_dict = {}
                edge_dict['parent_id'] = node.simulation_index
                edge_dict['child_id'] = child.simulation_index
                edge_topology_list.append(edge_dict)
                node_stack.append(child)
    return edge_topology_list, node_id_list, node_topology_list


def plot_simulation_graph(env_root, current_step, graph_directory=None):
    edge_topology_list, node_id_list, node_topology_list = obtain_tree_topology(env_root)
    dot = Digraph(comment='this is direction')
    for node_topology in node_topology_list:
        node_name = str(node_topology['node_id'])
        label = f"node_id: {node_topology['node_id']}, \n visit_count: {node_topology['visit_count']}, \n policy_prior: {round(node_topology['policy_prior'], 4)}, \n value: {round(node_topology['value'], 4)}"
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
