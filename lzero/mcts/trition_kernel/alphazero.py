import triton
import triton.language as tl
import torch
import numpy as np


@triton.jit
def ucb_score_kernel(
    parent_visit_count_ptr,
    child_visit_count_ptr,
    child_prior_p_ptr,
    child_value_ptr,
    output_ptr,
    pb_c_base,
    pb_c_init,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    parent_visits = tl.load(parent_visit_count_ptr + offsets, mask=mask)
    child_visits = tl.load(child_visit_count_ptr + offsets, mask=mask)
    prior_p = tl.load(child_prior_p_ptr + offsets, mask=mask)
    value = tl.load(child_value_ptr + offsets, mask=mask)

    # Calculate UCB score
    pb_c = tl.log((parent_visits + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c = pb_c * tl.sqrt(parent_visits) / (child_visits + 1)
    prior_score = pb_c * prior_p
    score = prior_score + value

    # Store result
    tl.store(output_ptr + offsets, score, mask=mask)


@triton.jit
def _select_child_kernel(
    # Pointers to input/output tensors
    node_children_ptr,          # (N, 2) tensor of [action, child_ptr] pairs
    legal_actions_ptr,          # Array of legal actions
    child_visit_counts_ptr,     # Visit counts for each child
    child_prior_p_ptr,         # Prior probabilities for each child
    parent_visit_count_ptr,     # Parent node visit count
    pb_c_base,                 # MCTS hyperparameter
    pb_c_init,                 # MCTS hyperparameter
    n_children,                # Number of children
    n_legal_actions,           # Number of legal actions
    BLOCK_SIZE: tl.constexpr   # Number of elements to process in parallel
):
    # Get program ID
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Initialize best score tracking
    best_score = tl.float32(-float('inf'))
    best_action = tl.int32(-1)
    best_child_idx = tl.int32(-1)
    
    # Load parent visit count
    parent_visits = tl.load(parent_visit_count_ptr)
    
    # Calculate UCB constant term
    pb_c = tl.log((parent_visits + pb_c_base + 1) / pb_c_base) + pb_c_init
    
    # Iterate through children in blocks
    for i in range(block_start, block_start + BLOCK_SIZE):
        if i < n_children:
            # Load child data
            action = tl.load(node_children_ptr + i * 2)
            child_idx = tl.load(node_children_ptr + i * 2 + 1)
            
            # Check if action is legal
            is_legal = False
            for j in range(n_legal_actions):
                legal_action = tl.load(legal_actions_ptr + j)
                if action == legal_action:
                    is_legal = True
                    break
            
            if is_legal:
                # Load child statistics
                visit_count = tl.load(child_visit_counts_ptr + child_idx)
                prior_p = tl.load(child_prior_p_ptr + child_idx)
                
                # Calculate UCB score
                pb_c_score = pb_c * tl.sqrt(parent_visits) / (visit_count + 1)
                prior_score = pb_c_score * prior_p
                value_score = -tl.load(child_value_ptr + child_idx)  # Negative because of alternating perspective
                ucb_score = prior_score + value_score
                
                # Update best score if needed
                if ucb_score > best_score:
                    best_score = ucb_score
                    best_action = action
                    best_child_idx = child_idx

    # Store results in output tensors
    if pid == 0:
        tl.store(output_action_ptr, best_action)
        tl.store(output_child_ptr, best_child_idx)


def select_child(node, legal_actions, pb_c_base=19652, pb_c_init=1.25):
    """
    Triton implementation of MCTS select_child operation
    
    Args:
        node: Current node in the MCTS tree
        legal_actions: Tensor of legal actions
        pb_c_base: MCTS hyperparameter
        pb_c_init: MCTS hyperparameter
    
    Returns:
        Tuple of (selected_action, selected_child_node)
    """
    # Convert node children to tensor format
    children_data = torch.tensor(
        [[action, id(child)] for action, child in node.children.items()],
        device='cuda',
        dtype=torch.int64
    )
    
    # Prepare input tensors
    legal_actions = torch.tensor(legal_actions, device='cuda', dtype=torch.int32)
    visit_counts = torch.tensor(
        [child.visit_count for child in node.children.values()],
        device='cuda',
        dtype=torch.float32
    )
    prior_probs = torch.tensor(
        [child.prior_p for child in node.children.values()],
        device='cuda',
        dtype=torch.float32
    )
    parent_visit_count = torch.tensor([node.visit_count], device='cuda', dtype=torch.float32)
    
    # Prepare output tensors
    output_action = torch.tensor([-1], device='cuda', dtype=torch.int32)
    output_child = torch.tensor([-1], device='cuda', dtype=torch.int64)
    
    # Configure grid and block sizes
    BLOCK_SIZE = 32
    grid = (triton.cdiv(len(node.children), BLOCK_SIZE),)
    
    # Launch kernel
    _select_child_kernel[grid](
        children_data.ptr,
        legal_actions.ptr,
        visit_counts.ptr,
        prior_probs.ptr,
        parent_visit_count.ptr,
        pb_c_base,
        pb_c_init,
        len(node.children),
        len(legal_actions),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Get results
    selected_action = output_action.item()
    selected_child_ptr = output_child.item()
    
    # Return original node if no valid selection was made
    if selected_action == -1:
        return -1, node
        
    return selected_action, node.children[selected_action]