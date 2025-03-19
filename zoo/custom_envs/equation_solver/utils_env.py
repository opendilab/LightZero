import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np

from itertools import product
from functools import lru_cache
from operator import add, sub, mul, truediv
from utils_custom_functions import custom_identity
from sympy import symbols, sympify, simplify, powdenest, ratsimp, E, I, pi, zoo,  Basic, Number, Integer, Float
from collections import deque
from torch_geometric.data import Data

############################################### 1D integer encoding ###############################################



def make_feature_dict(main_eqn, state_rep):

    if state_rep == 'integer_1d' or state_rep == 'graph_integer_1d':
        return make_feature_dict_integer_1d(main_eqn)
    elif state_rep == 'integer_2d' or state_rep == 'graph_integer_2d':
        return make_feature_dict_integer_2d(main_eqn)
    else:
        raise ValueError(f"Unsupported encoding type: {encoding}")


def make_feature_dict_integer_1d(main_eqn):
    """
    Generate a feature dictionary mapping symbols, numbers, and operations to unique integers.

    Args:
        lhs (sympy.Expr): Left-hand side expression.
        rhs (sympy.Expr): Right-hand side expression.

    Returns:
        dict: A dictionary mapping elements to integer encodings.

    Example:
        >>> from sympy import symbols
        >>> a, b, x = symbols('a b x')
        >>> lhs, rhs = a/x + b, 0
        >>> make_feature_dict(lhs, rhs)
        {'=': 0, x: 1, a: 2, b: 3, 0: 4, -1: 5, 'add': -1, 'pow': -2, 'mul': -3, 'sqrt': -4}
    """

    # Build feature dictionary
    free_symbols = list(main_eqn.free_symbols)
    special_constants = ['-1', 'I']
    integers = list(range(0,4))
    all_symbols = free_symbols + special_constants + integers

    x = symbols('x')
    feature_dict = {'=': 0, x: 1}
    ctr = 2
    for symbol in all_symbols:
        if symbol not in feature_dict:
            feature_dict[symbol] = 1.0*ctr
            ctr += 1

    # Encode operations as negative numbers
    ctr = -1
    operations = ['add', 'pow', 'mul', 'sqrt']
    for operation in operations:
        feature_dict[operation] = 1.0*ctr
        ctr -= 1

    return feature_dict


def integer_encoding_1d(lhs, rhs, feature_dict, max_length):
    """
    Convert symbolic expressions into integer-encoded vectors with padding.

    Args:
        lhs (sympy.Expr): Left-hand side expression.
        rhs (sympy.Expr): Right-hand side expression.
        feature_dict (dict): Dictionary mapping symbols and operations to integers.
        max_length (int): Maximum allowed length for each expression.

    Returns:
        np.ndarray: A fixed-length integer vector representing the encoded equation.

    Example:
        >>> from sympy import symbols
        >>> a, x = symbols('a x')
        >>> lhs, rhs = a*x + 2, 0
        >>> feature_dict = {'=': 0, a: 1, x: 2, 'add': -1, 'mul': -2, 2: 3}
        >>> integer_encoding(lhs, rhs, feature_dict, max_length=10)
        array([ 1, -2,  2, -1,  3,  0,  0,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7])
    """

    L = max_length
    PAD_ID = len(feature_dict) + 1  # Padding token

    vector = np.full(2 * L + 1, PAD_ID, dtype=np.float32)  # Preallocate with padding

    lhs_as_list, complexity_lhs = sympy_expression_to_list(lhs, feature_dict, L)
    rhs_as_list, complexity_rhs = sympy_expression_to_list(rhs, feature_dict, L)
    complexity = complexity_lhs + complexity_rhs

    # Fill array while ensuring the separator `=` is in the middle
    vector[:len(lhs_as_list)] = lhs_as_list
    vector[len(lhs_as_list)] = feature_dict["="]  # Middle separator
    vector[len(lhs_as_list) + 1 : len(lhs_as_list) + 1 + len(rhs_as_list)] = rhs_as_list

    return vector, complexity


def sympy_expression_to_list(expr, feature_dict, max_length):
    """
    Convert a SymPy expression into a fixed-length integer list using a feature dictionary.

    Args:
        expr (sympy.Expr): The symbolic expression to encode.
        feature_dict (dict): Dictionary mapping symbols and operations to integers.
        max_length (int): Maximum length of the output list.

    Returns:
        np.ndarray: A fixed-length list representing the expression, padded if necessary.

    Example:
        #>>> from sympy import symbols
        #>>> a, x = symbols('a x')
        #>>> expr = a*x + 2
        #>>> feature_dict = {'=': 0, a: 1, x: 2, 'add': -1, 'mul': -2, 2: 3}
        #>>> sympy_expression_to_list(expr, feature_dict, max_length=6)
        array([ 1, -2,  2, -1,  3,  7])  # 7 is the padding ID
    """

    PAD_ID = len(feature_dict) + 1  # Padding token
    elements = np.full(max_length, PAD_ID, dtype=np.int32)  # Preallocate with padding
    index = 0

    stack = [expr]  # Manual stack to avoid recursion overhead
    while stack and index < max_length:
        node = stack.pop()

        if isinstance(node, (int, float)):  # ✅ Handle numbers explicitly
            elements[index] = feature_dict.get(node, len(feature_dict))  # Assign new ID if missing
            feature_dict[node] = elements[index]  # Store it to avoid reassignments
            index += 1
        elif node.is_Symbol:
            elements[index] = feature_dict.get(node, 0)  # Default to 0 if missing
            index += 1
        elif node.is_Atom:
            continue  # Skip special cases like `pi` or `E` if needed
        else:
            elements[index] = feature_dict.get(node.func.__name__.lower(), 0)
            index += 1
            stack.extend(reversed(node.args))  # Push children to stack

    # Complexity measure: num of nodes
    complexity = index

    return elements, complexity  # Returns fixed-length padded array


def get_complexity_expression(expr):
    """
    Compute the complexity of a SymPy expression efficiently by counting the number of unique nodes 
    in its expression tree using an optimized traversal.

    Args:
        expr (sympy.Expr): The symbolic expression.

    Returns:
        int: Complexity score (total number of unique nodes in the expression tree).

    Example:
        >>> from sympy import symbols
        >>> a, x = symbols('a x')
        >>> expr = (a*x + 2) / (x + 1)
        >>> get_complexity_expression(expr)
        7
    """

    if expr == 0:
        return 1  # Complexity of constant zero is 1

    visited = set()
    stack = deque([expr])  # Use deque for fast pop/push operations
    complexity = 0

    while stack:
        node = stack.pop()

        if node in visited:
            continue  # Avoid redundant processing
        visited.add(node)

        complexity += 1  # Count this node

        # Add sub-expressions (children) efficiently
        if node.is_Atom:
            continue  # Skip atomic symbols and numbers
        stack.extend(node.args)  # Push children

    return complexity



def make_actions(lhs, rhs, actions_fixed, action_dim):
    """Generates a list of possible actions and a valid action mask."""
    
    # Dynamic actions (operation, term pairs)
    operations = [add, sub, mul, truediv]
    terms = get_ordered_sub_expressions(lhs)  # Extract meaningful terms
    actions_dynamic = list(product(operations, terms))

    # Combine fixed + dynamic actions
    actions = actions_fixed + actions_dynamic
    num_valid_actions = min(len(actions), action_dim)

    # Pre-allocate full-size mask (False by default)
    valid_action_mask = np.zeros(action_dim, dtype=bool)
    valid_action_mask[:num_valid_actions] = True  # Mark valid actions

    # Trim or pad with identity ops if needed
    actions = (actions[:action_dim] + [(custom_identity, None)] * action_dim)[:action_dim]

    return actions, valid_action_mask


@lru_cache(maxsize=1000)
def get_cached_terms(lhs):
    """Caches the sub-expressions extracted from lhs."""
    return get_ordered_sub_expressions(lhs)


def make_actions_cache(lhs, rhs, actions_fixed, action_dim, cache):
    """Generates a list of possible actions and a valid action mask with caching."""
    
    # Check cache first
    key = (lhs, rhs)
    if key in cache:
        return cache[key]

    # Dynamic actions (operation, term pairs)
    operations = [add, sub, mul, truediv]
    terms = get_cached_terms(lhs)  # Use cached sub-expressions
    actions_dynamic = list(product(operations, terms))

    # Combine fixed + dynamic actions
    actions = actions_fixed + actions_dynamic
    num_valid_actions = min(len(actions), action_dim)

    # Pre-allocate full-size mask (False by default)
    valid_action_mask = np.zeros(action_dim, dtype=bool)
    valid_action_mask[:num_valid_actions] = True  # Mark valid actions

    # Trim or pad with identity ops if needed
    actions = (actions[:action_dim] + [(custom_identity, None)] * action_dim)[:action_dim]

    # Store in cache
    cache[key] = (actions, valid_action_mask)

    return actions, valid_action_mask


def get_ordered_sub_expressions(expr):
    """Extracts all unique sub-expressions from a given symbolic expression, 
    excluding the full expression itself, and returns them in a fixed order.
    """
    
    if expr == 0:
        return []
    
    sub_expressions = set()
    queue = [expr]  # Use a queue for BFS traversal to ensure order consistency
    
    while queue:
        e = queue.pop(0)  # BFS: Process oldest element
        
#        if e not in sub_expressions and 1/e not in sub_expressions and e not in [-1,0,1]:
        if e not in sub_expressions and 1/e not in sub_expressions and not e.is_Integer and e not in [-1, 0, 1]:

            sub_expressions.add(e)

            # Only expand non-atomic expressions
            if not e.is_Atom:
                queue.extend(e.args)  

    # Exclude full expression and sort
    sub_expressions.discard(expr)  
    
    # Sorting rule:
    # - Prioritize smallest-length expressions first
    # - Tie-break by lexicographic order of string representations (stable mapping)
    #return sorted(sub_expressions, key=lambda x: x.sort_key())   #UNCOMMENT THIS ONE
    return sorted(sub_expressions, key=lambda x: (len(str(x)), str(x)))


def check_valid_eqn(lhs, rhs):
    """
    Validates and modifies an equation (lhs = rhs) for solving.
    
    Checks:
    1. Ensures the equation contains x. If not, returns invalid.
    2. If x is on the rhs but not lhs, it swaps them.

    Args:
        lhs (sympy.Expr or int): Left-hand side of the equation.
        rhs (sympy.Expr or int): Right-hand side of the equation.

    Returns:
        tuple: (bool, sympy.Expr, sympy.Expr) - Whether the equation is valid, and the modified (lhs, rhs).

    Example:
        >>> from sympy import symbols
        >>> x, a = symbols('x a')
        >>> is_valid_eqn(3, x + 1)
        (True, x + 1, 3)

        >>> is_valid_eqn(5, 2)
        (False, 5, 2)
    """

    x = symbols('x')

    # Check if lhs or rhs contains x
    lhs_has_x = getattr(lhs, 'has', lambda x: False)(x)
    rhs_has_x = getattr(rhs, 'has', lambda x: False)(x)

    if not lhs_has_x and not rhs_has_x:
        return False, lhs, rhs  # No x in the equation

    if not lhs_has_x and rhs_has_x:
        lhs, rhs = rhs, lhs  # Swap to ensure x is on lhs

    return True, lhs, rhs


def check_eqn_solved(lhs, rhs, main_eqn):
    """Checks if the equation is solved efficiently."""
    
    x = symbols('x')
    
    # Solution must have form (lhs, rhs) = (x, something without x)
    if lhs != x:
        return False

    # Check if rhs is a constant or does not contain x
    if isinstance(rhs, (int, float, Number)) or x not in rhs.free_symbols:

        # Substitute x = rhs into the main equation; should be zero
        sol = main_eqn.subs(x, rhs)

        # Fast checks before calling simplify
        if sol == 0 or sol.is_zero:
            return True

        # Use a cheaper method before full simplify
        if sol.expand() == 0:
            return True

        # Check for sqrts
        if powdenest(sol, force=True) == 0:
            return True

        # Use cheaper method first
        if ratsimp(sol) == 0:
            return True

        return False  # Avoid full simplify unless necessary
    
    return False



######################################## 2D encoding #########################################

def make_feature_dict_integer_2d(main_eqn):
    """
    Generates a dictionary that maps variables, constants, numbers, and operations 
    in the given equation to a structured 2D encoding.

    Encoding scheme:
    - Variables: [0, index] (e.g., x → [0, 0])
    - Constants/Symbols: [1, index] (e.g., a → [1, 0], b → [1, 1])
    - Numbers/Special Constants: [2, index] (e.g., 0 → [2, 2], '-1' → [2, 0], 'I' → [2, 1])
    - Operations: [3, index] (e.g., 'add' → [3, 0], 'pow' → [3, 1])

    Example output:
    ```
    {
        x: [0, 0],
        b: [1, 0],
        a: [1, 1],
        '-1': [2, 0],
        'I': [2, 1],
        0: [2, 2],
        1: [2, 3],
        2: [2, 4],
        3: [2, 5],
        'add': [3, 0],
        'pow': [3, 1],
        'mul': [3, 2],
        'sqrt': [3, 3]
    }
    ```

    Args:
        main_eqn (sympy expression): The mathematical equation to extract features from.

    Returns:
        dict: A dictionary mapping symbols and operations to their respective 2D encodings.
    """

    feature_dict = {}

    # 0: relation, = is special
    feature_dict['='] = [0,0]

    # 1: Operations
    operations = ['add', 'pow', 'mul', 'sqrt']
    feature_dict.update({op: [1, idx] for idx, op in enumerate(operations)})

    # 2: Variables
    x = symbols('x')
    variables = [x]
    feature_dict.update({var: [2, idx] for idx, var in enumerate(variables)})

    # 3: Constants/Symbols (excluding x)
    symbols_const = sorted(main_eqn.free_symbols - {x}, key=str)  # Sort for consistency
    feature_dict.update({sym: [3, idx] for idx, sym in enumerate(symbols_const)})

    # 4: Numbers & Special Constants
    special_constants = [I,E,pi,zoo]
    feature_dict.update({num: [4, idx] for idx, num in enumerate(special_constants)})

    # 5: Real numbers: added dynamically, so dont need in feature dict

    return feature_dict


def sympy_expression_to_list_2d(expr, feature_dict):
    """
    Convert a SymPy expression into a list of 2D encodings.
    
    Args:
        expr (sympy.Expr): The symbolic expression to encode.
        feature_dict (dict): Dictionary mapping symbols and operations to 2D encodings.

    Returns:
        List[List[int]]: A list of 2D encodings (without padding).
    """

    stack = [expr]
    encoded_list = []

    while stack:
        node = stack.pop()

        if node in feature_dict:
            encoded_list.append(feature_dict[node])
        elif isinstance(node, (int, float)):  # ✅ Handle numbers explicitly
            category_real_numbers = 5
            encoded_list.append([category_real_numbers, node])
        elif node.is_Symbol:
            encoded_list.append(feature_dict.get(node, [99, 99]))  # Use feature_dict encoding
        else:
            encoded_list.append(feature_dict.get(node.func.__name__.lower(), [99, 99]))  # Get encoding
            stack.extend(reversed(node.args))  # Push children to stack

    return encoded_list  # ✅ Returns list of valid encodings only (NO padding!)


def integer_encoding_2d(lhs, rhs, feature_dict, max_length):
    """
    Convert symbolic expressions into 2D-encoded vectors with fixed-length padding.

    Args:
        lhs (sympy.Expr): Left-hand side expression.
        rhs (sympy.Expr): Right-hand side expression.
        feature_dict (dict): Dictionary mapping symbols and operations to 2D encodings.
        max_length (int): Maximum allowed length for each expression.

    Returns:
        np.ndarray: A fixed-length 2D integer vector representing the encoded equation.
    """

    PAD_ID = [99, 99]  # Padding token

    # ✅ Get valid encodings without padding
    lhs_encoded = sympy_expression_to_list_2d(lhs, feature_dict)
    rhs_encoded = sympy_expression_to_list_2d(rhs, feature_dict)

    # ✅ Compute lengths
    lhs_len, rhs_len = len(lhs_encoded), len(rhs_encoded)

    # ✅ Fixed-size preallocation
    vector = np.full((2 * max_length + 1, 2), PAD_ID, dtype=np.int32)

    # ✅ Ensure lhs fits
    lhs_trimmed = lhs_encoded[:max_length]  # Truncate if needed
    lhs_actual_len = len(lhs_trimmed)

    # ✅ Ensure rhs fits
    rhs_trimmed = rhs_encoded[:max_length]  # Truncate if needed
    rhs_actual_len = len(rhs_trimmed)

    # ✅ Place values in the vector
    vector[:lhs_actual_len] = lhs_trimmed
    vector[lhs_actual_len] = feature_dict["="]  # Middle separator
    vector[lhs_actual_len + 1 : lhs_actual_len + 1 + rhs_actual_len] = rhs_trimmed

    complexity = 0
    return vector, complexity


###################################### Graph encoding #######################################


def sympy_expression_to_graph(expr, feature_dict):
    """
    Convert a SymPy expression into a graph representation for Torch Geometric.
    
    Args:
        expr (sympy.Expr): The symbolic expression to encode.
        feature_dict (dict): Dictionary mapping symbols and operations to features.

    Returns:
        torch_geometric.data.Data: Graph representation of the expression.
    """
    
    nodes = []  # List of node indices
    edges = []  # List of (parent, child) edges
    node_features = []  # Feature vector for each node
    node_map = {}  # Maps SymPy nodes to indices

    stack = [(expr, None)]  # (node, parent_index)
    node_idx = 0

    while stack:
        node, parent_idx = stack.pop()

        # Assign unique ID to each node
        if node not in node_map:
            node_map[node] = node_idx
            node_idx += 1

        cur_idx = node_map[node]
        nodes.append(cur_idx)

        # Get feature vector
        if node in feature_dict:
            node_features.append(feature_dict[node])
        elif isinstance(node, (int, float, Integer, Float)):  # Handle numbers
            node_features.append([5, node])  # Category 5 = real numbers
        elif node.is_Symbol:
            node_features.append(feature_dict.get(node, [99, 99]))  # Use feature_dict encoding
        else:
            node_features.append(feature_dict.get(node.func.__name__.lower(), [99, 99]))  # Get encoding

        # Connect to parent (if applicable)
        if parent_idx is not None:
            edges.append((parent_idx, cur_idx))

        # Push children to stack
        if not type(node) in [int,float] and node.args:  
            for child in reversed(node.args):
                stack.append((child, cur_idx))

    # Convert to Torch Tensors
    edge_index = torch.tensor(edges, dtype=torch.long).T if edges else torch.empty((2, 0), dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index)



def graph_encoding(lhs, rhs, feature_dict, max_length):
    """
    Convert symbolic equations into graph representations with fixed-size encoding
    and safely truncate overflow nodes/edges.
    """

    MAX_NODES, MAX_EDGES = 2*max_length + 1, 2*max_length + 1
    MAX_EDGES = 2*MAX_NODES

    # 1) Build raw graphs (LHS and RHS)
    lhs_graph = sympy_expression_to_graph(lhs, feature_dict)
    rhs_graph = sympy_expression_to_graph(rhs, feature_dict)

    # 2) Insert "=" node between LHS and RHS
    lhs_offset = lhs_graph.x.shape[0]
    eq_node_feature = torch.tensor([[0, 0]], dtype=torch.float)  # "=" symbol
    eq_node_idx = lhs_offset

    # Shift RHS node indices by (lhs_offset + 1)
    rhs_graph.edge_index += (lhs_offset + 1)

    # Merge node features
    x = torch.cat([lhs_graph.x, eq_node_feature, rhs_graph.x], dim=0)
    
    # Merge edges
    edge_index = torch.cat([lhs_graph.edge_index, rhs_graph.edge_index], dim=1)

    # Add edges connecting LHS-last-node -> "=" -> RHS-first-node
    lhs_last_idx = lhs_offset - 1
    rhs_start_idx = lhs_offset + 1
    eq_edges = torch.tensor([
        [lhs_last_idx, eq_node_idx],
        [eq_node_idx,  rhs_start_idx]
    ], dtype=torch.long)  # Shape is (2,2)

    # Just concatenate them directly, both 2D:
    edge_index = torch.cat([edge_index, eq_edges], dim=1)

    # 3) Inspect raw node/edge count
    raw_num_nodes = x.shape[0]
    raw_num_edges = edge_index.shape[1]
    #print("[graph_encoding] raw num_nodes:", raw_num_nodes)
    #print("[graph_encoding] raw num_edges:", raw_num_edges)
    #if edge_index.numel() > 0:
        #print("[graph_encoding] raw edge_index:\n", edge_index)
        #print("[graph_encoding] raw max node index in edges:", edge_index.max().item())

    # 4) Remove edges that reference nodes >= MAX_NODES
    #    (since we'll truncate node_features to the first MAX_NODES).
    valid_edge_mask = (edge_index[0] < MAX_NODES) & (edge_index[1] < MAX_NODES)
    edge_index = edge_index[:, valid_edge_mask]
    truncated_num_edges = edge_index.shape[1]

    # 5) Truncate node features if we have more than MAX_NODES
    if raw_num_nodes > MAX_NODES:
        x = x[:MAX_NODES]  # keep only the first MAX_NODES
        truncated_num_nodes = MAX_NODES
    else:
        truncated_num_nodes = raw_num_nodes

    # 6) Truncate edges if we have more than MAX_EDGES
    if truncated_num_edges > MAX_EDGES:
        edge_index = edge_index[:, :MAX_EDGES]
        truncated_num_edges = MAX_EDGES

    # 7) Now we pad up to MAX_NODES / MAX_EDGES
    padded_x = torch.full((MAX_NODES, 2), 99, dtype=torch.float)
    padded_x[:truncated_num_nodes] = x

    padded_edge_index = torch.full((2, MAX_EDGES), 0, dtype=torch.long)
    padded_edge_index[:, :truncated_num_edges] = edge_index

    # 8) Create boolean masks indicating which entries are "real"
    node_mask = torch.zeros(MAX_NODES, dtype=torch.bool)
    node_mask[:truncated_num_nodes] = True

    edge_mask = torch.zeros(MAX_EDGES, dtype=torch.bool)
    edge_mask[:truncated_num_edges] = True

    # Optional debug: final shapes
    #print("[graph_encoding] final truncated_num_nodes:", truncated_num_nodes)
    #print("[graph_encoding] final truncated_num_edges:", truncated_num_edges)
    #if truncated_num_edges > 0:
    #    print("[graph_encoding] final max node index in edges:", padded_edge_index[:,:truncated_num_edges].max().item())

    # 9) Wrap into a dictionary
    vec_dict = {
        "node_features": padded_x.numpy(),
        "edge_index": padded_edge_index.numpy(),
        "node_mask": node_mask.numpy(),
        "edge_mask": edge_mask.numpy(),
    }

    # Example complexity measure (not strictly needed)
    complexity = 0

    return vec_dict, complexity





##############################################################################################################
# Multi-Equation Feature Dictionary
##############################################################################################################


def make_feature_dict_multi(train_eqns, test_eqns, state_rep):
    """Chooses the correct feature dictionary function based on `state_rep`."""
    
    if state_rep in {'integer_1d', 'graph_integer_1d'}:
        return make_feature_dict_integer_1d_multi(train_eqns, test_eqns)
    elif state_rep in {'integer_2d', 'graph_integer_2d'}:
        return make_feature_dict_integer_2d_multi(train_eqns, test_eqns)
    else:
        raise ValueError(f"Unsupported encoding type: {state_rep}")


def make_feature_dict_integer_1d_multi(train_eqns, test_eqns):
    """
    Generate a feature dictionary mapping symbols, numbers, and operations to unique integers.

    Args:
        train_eqns (list): List of training equations (SymPy expressions).
        test_eqns (list): List of testing equations (SymPy expressions).

    Returns:
        dict: A dictionary mapping elements to integer encodings.
    """

    # Aggregate free symbols from ALL equations in train and test sets
    free_symbols = set()
    for eqn in train_eqns + test_eqns:
        free_symbols.update(eqn.free_symbols)

    # Special constants & numbers
    special_constants = [-1, I, E, pi, zoo]
    small_integers = list(range(4))  # e.g., 0,1,2,3
    all_symbols = list(free_symbols) + special_constants + small_integers

    # Initialize feature dictionary
    x = symbols('x')
    feature_dict = {'=': 0, x: 1}
    ctr = 2

    # Add all symbols
    for symbol in all_symbols:
        if symbol not in feature_dict:
            feature_dict[symbol] = ctr
            ctr += 1

    # Add operations
    operations = ['add', 'pow', 'mul', 'sqrt']
    for idx, operation in enumerate(operations, start=-1):
        feature_dict[operation] = idx

    return feature_dict



def make_feature_dict_integer_2d_multi(train_eqns, test_eqns):
    """
    Generates a dictionary that maps variables, constants, numbers, and operations 
    in the given set of equations to a structured 2D encoding.

    Encoding scheme:
    - Variables: [2, index] (e.g., x → [2, 0])
    - Constants/Symbols: [3, index] (e.g., a → [3, 0], b → [3, 1])
    - Numbers/Special Constants: [4, index] (e.g., 0 → [4, 2], '-1' → [4, 0], 'I' → [4, 1])
    - Operations: [1, index] (e.g., 'add' → [1, 0], 'pow' → [1, 1])
    """

    feature_dict = {}

    # 0: Relation '='
    feature_dict['='] = [0, 0]

    # 1: Operations
    operations = ['add', 'pow', 'mul', 'sqrt']
    feature_dict.update({op: [1, idx] for idx, op in enumerate(operations)})

    # 2: Variables (Ensure 'x' is always present)
    x = symbols('x')
    feature_dict[x] = [2, 0]

    # 3: Constants/Symbols (excluding x)
    symbols_const = set()
    for eqn in train_eqns + test_eqns:
        symbols_const.update(eqn.free_symbols)

    # Remove 'x' since it's categorized separately
    symbols_const.discard(x)
    feature_dict.update({sym: [3, idx] for idx, sym in enumerate(sorted(symbols_const, key=str))})

    # 4: Numbers & Special Constants
    special_constants = [-1, I, E, pi, zoo]
    feature_dict.update({num: [4, idx] for idx, num in enumerate(special_constants)})

    return feature_dict




################################# MULTIEQNS ###############################################


# def load_train_test_equations(dirn, level):
#     """
#     Loads train and test equations for a given level from the stored text files.
#     Returns lists of sympified equations.
#     """
#     level_dir = os.path.join(dirn, f"level{level}")

#     train_eqns_path = os.path.join(level_dir, "train_eqns.txt")
#     test_eqns_path = os.path.join(level_dir, "test_eqns.txt")

#     if not os.path.exists(train_eqns_path) or not os.path.exists(test_eqns_path):
#         raise FileNotFoundError(f"Train or test equation file not found for level {level} in {level_dir}")

#     # Read and sympify each equation
#     with open(train_eqns_path, "r") as f:
#         train_eqns = [sympify(line.strip()) for line in f.readlines()]

#     with open(test_eqns_path, "r") as f:
#         test_eqns = [sympify(line.strip()) for line in f.readlines()]

#     return train_eqns, test_eqns



def load_train_test_equations(dirn, level, generalization="shallow"):
    """
    Loads train and test equations for a given level from the stored text files.
    Supports both "shallow" and "deep" generalization.
    
    Parameters:
        dirn (str): Base directory where equation templates are stored.
        level (int): The level of equations to load.
        generalization (str): Either "shallow" or "deep" to load respective datasets.

    Returns:
        train_eqns (list): List of sympified train equations.
        test_eqns (list): List of sympified test equations.
    """
    if generalization not in ["shallow", "lexical", "structural", "deep", "random"]:
        raise ValueError(f"Invalid generalization type '{generalization}'. Choose 'shallow' or 'deep'.")

    # Construct path based on generalization type
    level_dir = os.path.join(dirn, generalization, f"level{level}")

    train_eqns_path = os.path.join(level_dir, "train_eqns.txt")
    test_eqns_path = os.path.join(level_dir, "test_eqns.txt")

    if not os.path.exists(train_eqns_path) or not os.path.exists(test_eqns_path):
        raise FileNotFoundError(f"Train or test equation file not found for level {level} in {level_dir}")

    # Read and sympify each equation
    with open(train_eqns_path, "r") as f:
        train_eqns = [sympify(line.strip()) for line in f.readlines()]

    with open(test_eqns_path, "r") as f:
        test_eqns = [sympify(line.strip()) for line in f.readlines()]

    return train_eqns, test_eqns

