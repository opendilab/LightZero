import os
from dataclasses import dataclass
from typing import Any, List

import numpy as np
from graphviz import Digraph


def generate_random_actions_discrete(num_actions: int, action_space_size: int, num_of_sampled_actions: int,
                                     reshape=False):
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
    actions = [
        np.random.randint(0, action_space_size, num_of_sampled_actions).reshape(-1)
        for _ in range(num_actions)
    ]

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

def prepare_observation(observation_list: List[Any], model_type: str = 'conv') -> np.ndarray:
    """
    Prepare the observations to satisfy the input format of the model. This function is robust against
    inhomogeneous shapes and mixed types (np.ndarray and list) in the input list.

    For model_type='conv':
        [B, S, C, W, H] -> [B, S x C, W, H]
        where B is batch size, S is stack num, W is width, H is height, and C is the number of channels.

    For model_type='mlp':
        [B, S, O] -> [B, S x O]
        where B is batch size, S is stack num, O is obs shape.

    Arguments:
        - observation_list (List): A list of observations, which can be a mix of np.ndarray and lists.
        - model_type (str): type of the model. (default is 'conv')

    Returns:
        - np.ndarray: A single, reshaped NumPy array of observations.
    """
    assert model_type in ['conv', 'mlp', 'conv_context', 'mlp_context'], "model_type must be either 'conv' or 'mlp' or their context variants"

    if not observation_list:
        # Handle empty list case
        return np.array([])

    # ==================== START OF ROBUST FIX ====================
    try:
        # First, try the fast path assuming the data is clean.
        observation_array = np.array(observation_list)
    except (ValueError, TypeError):
        # If np.array() fails, it's likely due to inhomogeneous shapes or mixed types.
        # We now enter the robust slow path to clean the data.
        import logging
        logger = logging.getLogger(__name__)

        # 1. Find a valid np.ndarray in the list to use as a template for shape and dtype.
        template_obs = None
        for obs in observation_list:
            if isinstance(obs, np.ndarray):
                template_obs = obs
                break
        
        if template_obs is None:
            # This is a critical error: the list contains no np.ndarray to infer shape from.
            # It might be all lists, or something else entirely. We try to convert the first element.
            try:
                template_obs = np.array(observation_list[0])
                logger.warning(f"No np.ndarray found in observation_list. Using the first element as template. Shape: {template_obs.shape}")
            except Exception as e:
                 raise ValueError(f"Could not create a template observation. The observation list contains no np.ndarray "
                                  f"and the first element could not be converted. First element type: {type(observation_list[0])}. Error: {e}")

        target_shape = template_obs.shape
        target_dtype = template_obs.dtype
        target_size = template_obs.size
        
        processed_obs_list = []
        for i, obs in enumerate(observation_list):
            # 2. Convert every element to a np.ndarray.
            if not isinstance(obs, np.ndarray):
                try:
                    obs = np.array(obs, dtype=target_dtype)
                except (ValueError, TypeError):
                    logger.error(f"Failed to convert element at index {i} to a numpy array. Element: {obs}")
                    # As a last resort, create a zero array with the target shape.
                    obs = np.zeros(target_shape, dtype=target_dtype)

            # 3. Ensure every np.ndarray has the target shape.
            if obs.shape != target_shape:
                logger.warning(
                    f"[OBSERVATION_SHAPE_MISMATCH] Standardizing observation at index {i}. "
                    f"Expected shape {target_shape}, but got {obs.shape}. Padding/truncating."
                )
                obs_flat = obs.flatten()
                if obs_flat.size < target_size:
                    # Pad with zeros
                    padded = np.zeros(target_size, dtype=target_dtype)
                    padded[:obs_flat.size] = obs_flat
                    obs = padded.reshape(target_shape)
                else:
                    # Truncate
                    obs = obs_flat[:target_size].reshape(target_shape)
            
            processed_obs_list.append(obs)

        # 4. Now, np.array() on the cleaned list should succeed.
        observation_array = np.array(processed_obs_list)
    # ===================== END OF ROBUST FIX =====================

    batch_size = observation_array.shape[0]

    if model_type in ['conv', 'conv_context']:
        # This part handles stacking frames for CNNs.
        if observation_array.ndim == 3:
            # Case: [B, W, H] -> [B, 1, W, H] (Add a channel dimension)
            observation_array = observation_array[..., np.newaxis]
        
        if observation_array.ndim == 5:
            # Case: [B, S, C, W, H] -> [B, S*C, W, H] (Stack frames and channels)
            _, stack_num, channels, width, height = observation_array.shape
            observation_array = observation_array.reshape(batch_size, stack_num * channels, width, height)

    elif model_type in ['mlp', 'mlp_context']:
        # This part handles flattening features for MLPs.
        if observation_array.ndim > 2:
            # Case: [B, S, O] or [B, S, W, H, C] etc. -> [B, S*O] (Flatten all but batch dim)
            observation_array = observation_array.reshape(batch_size, -1)
        # If ndim is 2 ([B, O]), it's already in the correct format.

    return observation_array


# def prepare_observation(observation_list, model_type='conv'):
#     """
#     Prepare the observations to satisfy the input format of the model.

#     For model_type='conv':
#         [B, S, C, W, H] -> [B, S x C, W, H]
#         where B is batch size, S is stack num, W is width, H is height, and C is the number of channels.

#     For model_type='mlp':
#         [B, S, O] -> [B, S x O]
#         where B is batch size, S is stack num, O is obs shape.

#     Arguments:
#         - observation_list (List): list of observations.
#         - model_type (str): type of the model. (default is 'conv')

#     Returns:
#         - np.ndarray: Reshaped array of observations.
#     """
#     assert model_type in ['conv', 'mlp', 'conv_context', 'mlp_context'], "model_type must be either 'conv' or 'mlp'"

#     # [FIX] Handle inhomogeneous shapes in observation_list
#     # For text-based environments (e.g., Jericho), observations may have varying shapes
#     try:
#         observation_array = np.array(observation_list)
#     except ValueError as e:
#         # If shapes are inhomogeneous, check if we can handle it
#         if "inhomogeneous" in str(e) or "sequence" in str(e):
#             # For MLP models with text observations, pad/truncate to consistent shape
#             if model_type in ['mlp', 'mlp_context']:
#                 import logging
#                 logger = logging.getLogger(__name__)

#                 # Find the target shape (use the first element as reference)
#                 if len(observation_list) > 0:
#                     first_obs = observation_list[0]
#                     if isinstance(first_obs, np.ndarray):
#                         target_shape = first_obs.shape

#                         # Check if all observations can be reshaped to target_shape
#                         processed_obs = []
#                         for obs in observation_list:
#                             if isinstance(obs, np.ndarray):
#                                 if obs.shape == target_shape:
#                                     processed_obs.append(obs)
#                                 elif obs.size == np.prod(target_shape):
#                                     # Can reshape to target shape
#                                     processed_obs.append(obs.reshape(target_shape))
#                                 else:
#                                     # Pad or truncate
#                                     logger.warning(
#                                         f"[OBSERVATION_SHAPE_MISMATCH] Expected {target_shape}, "
#                                         f"got {obs.shape}. Padding/truncating."
#                                     )
#                                     obs_flat = obs.flatten()
#                                     target_size = np.prod(target_shape)
#                                     if obs_flat.size < target_size:
#                                         # Pad with zeros
#                                         padded = np.zeros(target_size)
#                                         padded[:obs_flat.size] = obs_flat
#                                         processed_obs.append(padded.reshape(target_shape))
#                                     else:
#                                         # Truncate
#                                         processed_obs.append(obs_flat[:target_size].reshape(target_shape))
#                             else:
#                                 processed_obs.append(first_obs)  # Fallback

#                         observation_array = np.array(processed_obs)
#                     else:
#                         raise ValueError(f"Cannot process non-ndarray observations: {type(first_obs)}")
#                 else:
#                     raise ValueError("observation_list is empty")
#             else:
#                 raise e
#         else:
#             raise e

#     batch_size = observation_array.shape[0]

#     if model_type in ['conv', 'conv_context']:
#         if observation_array.ndim == 3:
#             # Add a channel dimension if it's missing
#             observation_array = observation_array[..., np.newaxis]
#         elif observation_array.ndim == 5:
#             # Reshape to [B, S*C, W, H]
#             _, stack_num, channels, width, height = observation_array.shape
#             observation_array = observation_array.reshape(batch_size, stack_num * channels, width, height)

#     elif model_type in ['mlp', 'mlp_context']:
#         if observation_array.ndim == 3:
#             # Flatten the last two dimensions
#             observation_array = observation_array.reshape(batch_size, -1)
#         else:
#             raise ValueError("For 'mlp' model_type, the observation must have 3 dimensions [B, S, O]")

#     return observation_array


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
