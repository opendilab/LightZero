import triton
import triton.language as tl


@triton.jit
def add_exploration_noise_kernel(
    # Pointers to arrays
    prior_ptr,  # float32 pointer to prior values [num_nodes, num_actions]
    noise_ptr,  # float32 pointer to noise values [num_nodes, num_actions]
    legal_actions_ptr,  # int32 pointer to legal actions mask [num_nodes, num_actions]
    # Shape parameters
    num_nodes,  # number of nodes
    num_actions,  # maximum number of actions
    exploration_fraction,  # float scalar
    # Block size parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate pid - the program ID
    pid = tl.program_id(axis=0)

    # Calculate number of elements each program should handle
    num_elements = num_nodes * num_actions

    # Handle BLOCK_SIZE elements per program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Create masks for valid elements
    mask = offsets < num_elements

    # Calculate node and action indices
    node_idx = offsets // num_actions
    action_idx = offsets % num_actions

    # Load values using masks
    prior = tl.load(prior_ptr + offsets, mask=mask)
    noise = tl.load(noise_ptr + offsets, mask=mask)
    is_legal = tl.load(legal_actions_ptr + offsets, mask=mask)

    # Compute new prior values
    new_prior = tl.where(is_legal, prior * (1.0 - exploration_fraction) + noise * exploration_fraction, prior)

    # Store results
    tl.store(prior_ptr + offsets, new_prior, mask=mask)


@triton.jit
def compute_mean_q_kernel(
    # Node data
    legal_actions_mask_ptr,  # Legal actions mask [num_nodes, num_actions]
    child_rewards_ptr,  # Rewards of child nodes [num_nodes, num_actions]
    child_values_ptr,  # Values of child nodes [num_nodes, num_actions]
    child_visit_counts_ptr,  # Visit counts of child nodes [num_nodes, num_actions]
    # Input parameters
    is_root_ptr,  # Whether the node is root [num_nodes]
    parent_q_ptr,  # Q value of parent node [num_nodes]
    discount_factor,  # Discount factor for reward
    # Output
    mean_q_ptr,  # Output mean Q values [num_nodes]
    # Metadata
    num_nodes,  # Total number of nodes
    num_actions,  # Total number of possible actions
    BLOCK_SIZE: tl.constexpr,
):
    # Get thread index
    pid = tl.program_id(0)
    if pid >= num_nodes:
        return

    # Accumulator variables
    total_unsigned_q = 0.0
    total_visits = 0

    # Base offset for current node
    base_offset = pid * num_actions

    # Iterate through all actions
    for i in range(num_actions):
        offset = base_offset + i

        # Check if action is legal
        is_legal = tl.load(legal_actions_mask_ptr + offset)
        visit_count = tl.load(child_visit_counts_ptr + offset)

        # Only process legal and visited child nodes
        if is_legal and (visit_count > 0):
            reward = tl.load(child_rewards_ptr + offset)
            value = tl.load(child_values_ptr + offset)

            # Calculate Q value
            qsa = reward + discount_factor * value
            total_unsigned_q += qsa
            total_visits += 1

    # Calculate mean Q
    is_root = tl.load(is_root_ptr + pid)
    parent_q = tl.load(parent_q_ptr + pid)

    # For root nodes, parent_q is 0, so we can combine the conditions:
    # When is_root=True: (0 + total_unsigned_q) / (total_visits + 0)
    # When is_root=False: (parent_q + total_unsigned_q) / (total_visits + 1)
    divisor = total_visits + (1 - is_root)
    mean_q = (parent_q + total_unsigned_q) / tl.maximum(divisor, 1)

    # Store result
    tl.store(mean_q_ptr + pid, mean_q)


@triton.jit
def value_kernel(
    # Node data
    value_sum_ptr,  # Sum of values for each node [num_nodes]
    visit_count_ptr,  # Visit count for each node [num_nodes]
    # Output
    value_ptr,  # Output value for each node [num_nodes]
    # Metadata
    num_nodes,  # Total number of nodes
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute value = value_sum / visit_count for each node
    Returns 0 if visit_count is 0
    """
    # Get thread index
    pid = tl.program_id(0)
    if pid >= num_nodes:
        return

    # Load node data
    value_sum = tl.load(value_sum_ptr + pid)
    visit_count = tl.load(visit_count_ptr + pid)

    # Compute value
    value = tl.where(visit_count > 0, value_sum / visit_count, 0.0)

    # Store result
    tl.store(value_ptr + pid, value)


@triton.jit
def batch_backpropagate_kernel(
    # Search path data (flattened and padded)
    path_node_value_sum_ptr,  # Value sum of nodes in paths [num_paths, max_path_len]
    path_node_visit_count_ptr,  # Visit count of nodes in paths [num_paths, max_path_len]
    path_node_reward_ptr,  # Reward of nodes in paths [num_paths, max_path_len]
    path_node_to_play_ptr,  # To play of nodes in paths [num_paths, max_path_len]
    path_length_ptr,  # Length of each path [num_paths]

    # Input parameters
    to_play_batch_ptr,  # Which player to play for each path [num_paths]
    values_ptr,  # Values to propagate [num_paths]
    discount_factor,  # Discount factor, float scalar

    # MinMax stats data
    min_max_stats_min_ptr,  # Min values for normalization [num_paths]
    min_max_stats_max_ptr,  # Max values for normalization [num_paths]

    # Metadata
    num_paths,  # Number of search paths
    max_path_len,  # Maximum path length after padding
    BLOCK_SIZE: tl.constexpr,
):
    """
    Parallel backpropagation for multiple search paths
    """
    # Get thread index (each thread processes one path)
    pid = tl.program_id(0)
    if pid >= num_paths:
        return

    # Load path length and to_play
    path_len = tl.load(path_length_ptr + pid)
    to_play = tl.load(to_play_batch_ptr + pid)
    bootstrap_value = tl.load(values_ptr + pid)

    # Base offset for current path
    base_offset = pid * max_path_len

    # Backward pass through the path
    for i in range(max_path_len):
        # Convert to reverse index
        idx = path_len - 1 - i
        if idx < 0:  # Skip padding
            break

        offset = base_offset + idx

        # Load node data
        node_to_play = tl.load(path_node_to_play_ptr + offset)
        node_reward = tl.load(path_node_reward_ptr + offset)

        # Update value sum based on to_play
        if to_play == -1:  # play-with-bot mode
            value_sum_update = bootstrap_value
        else:  # self-play mode
            value_sum_update = tl.where(node_to_play == to_play, bootstrap_value, -bootstrap_value)

        # Update node stats
        old_value_sum = tl.load(path_node_value_sum_ptr + offset)
        old_visit_count = tl.load(path_node_visit_count_ptr + offset)

        tl.store(path_node_value_sum_ptr + offset, old_value_sum + value_sum_update)
        tl.store(path_node_visit_count_ptr + offset, old_visit_count + 1)

        # Calculate node value for min-max stats
        node_value = tl.where(old_visit_count > 0, old_value_sum / old_visit_count, 0.0)

        # Update min-max stats
        q_value = node_reward + discount_factor * (-node_value)
        min_val = tl.load(min_max_stats_min_ptr + pid)
        max_val = tl.load(min_max_stats_max_ptr + pid)

        min_val = tl.minimum(min_val, q_value)
        max_val = tl.maximum(max_val, q_value)

        tl.store(min_max_stats_min_ptr + pid, min_val)
        tl.store(min_max_stats_max_ptr + pid, max_val)

        # Update bootstrap value
        if to_play == -1:  # play-with-bot mode
            bootstrap_value = node_reward + discount_factor * bootstrap_value
        else:  # self-play mode
            bootstrap_value = tl.where(
                node_to_play == to_play, -node_reward + discount_factor * bootstrap_value,
                node_reward + discount_factor * bootstrap_value
            )


@triton.jit
def select_child_kernel(
    # Node data
    legal_actions_mask_ptr,  # Legal actions mask [num_nodes, num_actions]
    child_visit_counts_ptr,  # Visit counts of child nodes [num_nodes, num_actions]
    child_rewards_ptr,  # Rewards of child nodes [num_nodes, num_actions]
    child_values_ptr,  # Values of child nodes [num_nodes, num_actions]
    child_priors_ptr,  # Priors of child nodes [num_nodes, num_actions]
    # Parent node data
    visit_counts_ptr,  # Visit counts of parent nodes [num_nodes]
    mean_q_ptr,  # Mean Q values of parent nodes [num_nodes]
    # MinMax stats
    min_max_stats_min_ptr,  # Min values for normalization [num_nodes]
    min_max_stats_max_ptr,  # Max values for normalization [num_nodes]
    # Constants
    pb_c_base,
    pb_c_init,
    discount_factor,
    # Output
    selected_actions_ptr,  # Selected actions [num_nodes]
    # Metadata
    num_nodes,  # Total number of nodes
    num_actions,  # Maximum number of actions
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate pid - the program ID
    pid = tl.program_id(axis=0)

    # Calculate number of elements each program should handle
    num_elements = num_nodes * num_actions

    # Handle BLOCK_SIZE elements per program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Create masks for valid elements
    mask = offsets < num_elements

    # Calculate node and action indices
    node_idx = offsets // num_actions
    action_idx = offsets % num_actions

    # Load parent node data using node_idx
    parent_visit_count = tl.load(visit_counts_ptr + node_idx, mask=mask)
    mean_q = tl.load(mean_q_ptr + node_idx, mask=mask)
    min_value = tl.load(min_max_stats_min_ptr + node_idx, mask=mask)
    max_value = tl.load(min_max_stats_max_ptr + node_idx, mask=mask)

    # Load action data using offsets
    is_legal = tl.load(legal_actions_mask_ptr + offsets, mask=mask)
    child_visit_count = tl.load(child_visit_counts_ptr + offsets, mask=mask)
    child_reward = tl.load(child_rewards_ptr + offsets, mask=mask)
    child_value = tl.load(child_values_ptr + offsets, mask=mask)
    child_prior = tl.load(child_priors_ptr + offsets, mask=mask)

    # Compute UCB score
    pb_c = tl.log((parent_visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= tl.sqrt(parent_visit_count) / (child_visit_count + 1)

    prior_score = pb_c * child_prior
    value_score = child_reward + discount_factor * child_value

    # Normalize value score if child has been visited
    value_score = tl.where(child_visit_count > 0, (max_value - value_score) / (max_value - min_value), value_score)

    # Calculate final score
    ucb_score = tl.where(is_legal, prior_score + value_score, -float('inf'))

    # TODO: check if this is correct and faster than use num_actions as BLOCK_SIZE
    # For each node, find max score and corresponding action
    curr_node = node_idx[0]
    max_score = -float('inf')
    selected_action = -1

    for i in range(BLOCK_SIZE):
        if mask[i]:  # Check if this element is valid
            # When we move to a new node, store the result for previous node
            if node_idx[i] != curr_node and i > 0:
                tl.store(selected_actions_ptr + curr_node, selected_action)
                curr_node = node_idx[i]
                max_score = -float('inf')
                selected_action = -1

            # Update max score and selected action
            score = ucb_score[i]
            selected_action = tl.where(score > max_score, action_idx[i], selected_action)
            max_score = tl.maximum(max_score, score)

    # Store result for the last node
    if mask[0]:  # At least one element was processed
        tl.store(selected_actions_ptr + curr_node, selected_action)


@triton.jit
def expand_kernel(
    # Node data
    node_to_play_ptr,  # [num_nodes]
    node_latent_index_ptr,  # [num_nodes]
    node_batch_index_ptr,  # [num_nodes]
    node_reward_ptr,  # [num_nodes]
    node_expanded_ptr,  # [num_nodes]

    # Policy data
    policy_logits_ptr,  # [num_nodes, num_actions]
    legal_actions_mask_ptr,  # [num_nodes, num_actions]

    # Child node data
    child_prior_ptr,  # [num_nodes, num_actions]
    child_expanded_ptr,  # [num_nodes, num_actions]

    # Input parameters
    to_play_batch_ptr,  # [num_nodes]
    latent_index_batch_ptr,  # [num_nodes]
    batch_index_ptr,  # [num_nodes]
    reward_batch_ptr,  # [num_nodes]

    # Metadata
    num_nodes,
    num_actions,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate pid - the program ID
    pid = tl.program_id(axis=0)

    # Calculate number of elements each program should handle
    num_elements = num_nodes * num_actions

    # Handle BLOCK_SIZE elements per program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Create masks for valid elements
    mask = offsets < num_elements

    # Calculate node and action indices
    node_idx = offsets // num_actions
    action_idx = offsets % num_actions

    # Load node parameters for the current block
    to_play = tl.load(to_play_batch_ptr + node_idx, mask=mask)
    latent_index = tl.load(latent_index_batch_ptr + node_idx, mask=mask)
    batch_index = tl.load(batch_index_ptr + node_idx, mask=mask)
    reward = tl.load(reward_batch_ptr + node_idx, mask=mask)

    # Load policy data
    is_legal = tl.load(legal_actions_mask_ptr + offsets, mask=mask)
    logits = tl.load(policy_logits_ptr + offsets, mask=mask)

    # Compute masked logits
    masked_logits = tl.where(is_legal, logits, -float('inf'))

    # Find max logit for each node using parallel reduction
    policy_max = tl.max(masked_logits, axis=1)  # reduce across actions

    # Compute exp(logit - max) with numerical stability
    temp_policy = tl.exp(logits - policy_max)
    temp_policy = tl.where(is_legal, temp_policy, 0.0)

    # Compute sum for normalization using parallel reduction
    policy_sum = tl.sum(temp_policy, axis=1)  # reduce across actions

    # Normalize policy
    prior = tl.where(is_legal, temp_policy / policy_sum, 0.0)

    # Store results
    tl.store(node_to_play_ptr + node_idx, to_play, mask=mask)
    tl.store(node_latent_index_ptr + node_idx, latent_index, mask=mask)
    tl.store(node_batch_index_ptr + node_idx, batch_index, mask=mask)
    tl.store(node_reward_ptr + node_idx, reward, mask=mask)
    tl.store(node_expanded_ptr + node_idx, 1, mask=mask)  # Mark as expanded
    tl.store(child_prior_ptr + offsets, prior, mask=mask)
    tl.store(child_expanded_ptr + offsets, 0, mask=mask)  # Mark children as not expanded
