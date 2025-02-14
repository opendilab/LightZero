import sys
from pathlib import Path

import numpy as np
import pytest
from easydict import EasyDict


def find_project_root(current_path: Path, target_dir="lzero"):
    """
    Recursively search upwards from the current path to locate the project root
    that contains the target directory.

    Args:
        current_path (Path): The starting path for the search.
        target_dir (str): The name of the target directory to locate.

    Returns:
        Path: The project root directory containing the target directory.

    Raises:
        FileNotFoundError: If the target directory is not found.
    """
    for parent in current_path.parents:
        if (parent / target_dir).is_dir():
            return parent
    raise FileNotFoundError(f"Could not find the project root containing the directory '{target_dir}'")


def find_and_add_to_sys_path(target_subdir: str, target_dir="lzero"):
    """
    Dynamically locate a specific directory under the project root and add it to `sys.path`.

    Args:
        target_subdir (str): The relative path to the target subdirectory (from the project root).
        target_dir (str): The name of the main project directory to locate.

    Raises:
        FileNotFoundError: If the target directory or subdirectory is not found.
    """
    # Find the project root dynamically
    project_root = find_project_root(Path(__file__).resolve(), target_dir)

    # Build the full path to the target subdirectory
    target_path = project_root / target_subdir

    # Check if the directory exists and add it to sys.path
    if target_path.is_dir():
        sys.path.append(str(target_path))
    else:
        raise FileNotFoundError(f"Target path does not exist: {target_path}")


# Use the function to add the desired path to sys.path
find_and_add_to_sys_path("lzero/mcts/ctree/ctree_alphazero/build")

import mcts_alphazero


class MockEnv:
    """
    A simple mock environment class that includes the necessary attributes and methods.
    Used to replace a real environment for unit testing.
    """

    def __init__(self):
        # Define the set of legal actions
        self.legal_actions = [0, 1, 2]
        # MCTS mode: self-play mode
        self.battle_mode_in_simulation_env = "self_play_mode"
        # Current player index
        self.current_player = 1
        # Current timestep
        self.timestep = 0
        # Action space, defining the number of actions
        self.action_space = type('action_space', (), {'n': 3})()

    def reset(self, start_player_index, init_state, katago_policy_init, katago_game_state):
        """
        Mock environment reset method.
        Initializes the environment state.
        """
        self.current_player = 1
        self.timestep = 0

    def step(self, action):
        """
        Mock environment step method.
        Executes the action and switches players.
        """
        self.current_player = 2 if self.current_player == 1 else 1
        self.timestep += 1

    def get_done_winner(self):
        """
        Mock environment get_done_winner method.
        Returns whether the game is over and the winner.
        The first return value indicates if the game is done.
        The second return value indicates the winner: 1 for player 1, 2 for player 2, -1 for a draw or ongoing game.
        (False, -1) indicates the game is not yet finished.
        """
        return (False, -1)


def mock_policy_value_func(env):
    """
    A mock policy_value_func function.
    Returns the action probability distribution and the value of the leaf node.
    """
    return ({0: 0.4, 1: 0.4, 2: 0.2}, 0.9)


@pytest.fixture
def mcts_fixture():
    """
    Initializes MCTS object and test environment using a pytest fixture.
    Provides a standardized testing environment to avoid repetitive code.
    """
    mcts = mcts_alphazero.MCTS(
        max_moves=100,  # Maximum number of moves
        num_simulations=100,  # Number of simulations per search
        pb_c_base=19652,  # Parameter for UCB score calculation
        pb_c_init=1.25,  # Initial value for UCB score calculation
        root_dirichlet_alpha=0.3,  # Dirichlet noise parameter for the root node
        root_noise_weight=0.25,  # Root node noise weight
        simulate_env=None  # The environment will be set during testing
    )
    root = mcts_alphazero.Node()  # Create the root node
    mock_env = MockEnv()  # Create the mock environment
    policy_value_func = mock_policy_value_func  # Use the mock policy-value function
    legal_actions = [0, 1, 2]  # Define the set of legal actions
    return mcts, root, mock_env, policy_value_func, legal_actions


def test_node_initialization():
    """
    Test if the Node class initializes correctly.
    Checks the default values and basic attributes of the node.
    """
    root = mcts_alphazero.Node()

    assert root.parent is None, "The parent of the root node should be None"
    assert root.prior_p == 1.0, "The prior_p of the root node should default to 1.0"
    assert root.visit_count == 0, "The visit_count of the root node should default to 0"
    assert root.value == 0.0, "The initial value of the root node should be 0.0"
    assert root.is_leaf(), "A newly created root node should be a leaf node"
    assert root.is_root(), "A newly created root node should be a root node"


def test_node_update():
    """
    Test if the update method of the Node class correctly updates visit_count and value_sum.
    Verify the logic for updating a node's visit count and value.
    """
    node = mcts_alphazero.Node()
    node.update(5.0)

    assert node.visit_count == 1, "After one update, visit_count should be 1"
    assert node.value == 5.0, "After one update, value should be 5.0"

    node.update(3.0)
    assert node.visit_count == 2, "After two updates, visit_count should be 2"
    assert node.value == 4.0, "After two updates, value should be (5.0 + 3.0) / 2 = 4.0"


def test_node_recursive_update_self_play_mode():
    """
    Test recursive updates in self-play mode for the Node class.
    In self-play mode, the parent's value is negated.
    """
    parent = mcts_alphazero.Node()
    child = mcts_alphazero.Node(parent=parent, prior_p=0.5)
    parent.add_child(1, child)

    child.update_recursive(1.0, "self_play_mode")

    assert child.visit_count == 1, "The visit_count of the child node should be 1"
    assert child.value == 1.0, "The value of the child node should be 1.0"
    assert parent.visit_count == 1, "The visit_count of the parent node should be 1"
    assert parent.value == -1.0, "The value of the parent node should be -1.0"


def test_node_recursive_update_play_with_bot_mode():
    """
    Test recursive updates in play-with-bot mode for the Node class.
    In play-with-bot mode, the parent's value is not negated.
    """
    parent = mcts_alphazero.Node()
    child = mcts_alphazero.Node(parent=parent, prior_p=0.5)
    parent.add_child(2, child)

    child.update_recursive(1.0, "play_with_bot_mode")

    assert child.visit_count == 1, "The visit_count of the child node should be 1"
    assert child.value == 1.0, "The value of the child node should be 1.0"
    assert parent.visit_count == 1, "The visit_count of the parent node should be 1"
    assert parent.value == 1.0, "The value of the parent node should be 1.0"


def test_node_add_child():
    """
    Test if the add_child method of the Node class correctly adds child nodes.
    Verify the logic for adding child nodes and the parent-child relationship.
    """
    parent = mcts_alphazero.Node()
    child = mcts_alphazero.Node(parent=parent, prior_p=0.7)
    parent.add_child(3, child)

    assert 3 in parent.children, "A child node with action 3 should be added to the parent's children"
    assert parent.children[3] is child, "The added child node should match the passed child"
    assert not parent.is_leaf(), "After adding a child, the parent node should not be a leaf node"


def test_ucb_score(mcts_fixture):
    """
    Test if the _ucb_score method of MCTS correctly calculates UCB scores.
    Verify if the UCB formula is implemented as expected.
    """
    mcts, root, _, _, _ = mcts_fixture

    parent = root
    child = mcts_alphazero.Node(parent=parent, prior_p=0.5)
    parent.add_child(0, child)

    for _ in range(10):
        parent.update(1.0)
    for _ in range(2):
        child.update(1.0)

    ucb = mcts._ucb_score(parent, child)

    expected_pb_c = np.log(
        (parent.visit_count + mcts.pb_c_base + 1) / mcts.pb_c_base) + mcts.pb_c_init
    expected_pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)
    expected_score = expected_pb_c * child.prior_p + child.value

    assert ucb == expected_score, "UCB score calculation is incorrect"


def test_add_exploration_noise(mcts_fixture):
    """
    Test if the _add_exploration_noise method of MCTS correctly adds exploration noise.
    Verify if the prior_p of root node's children is reasonable after adding noise.
    """
    mcts, root, _, _, _ = mcts_fixture

    root.add_child(0, mcts_alphazero.Node(parent=root, prior_p=0.4))
    root.add_child(1, mcts_alphazero.Node(parent=root, prior_p=0.6))

    mcts._add_exploration_noise(root)

    for action in root.children:
        child = root.children[action]
        assert 0.0 <= child.prior_p <= 1.0, "After adding exploration noise, prior_p is out of range [0, 1]"


def test_get_next_action(mcts_fixture):
    """
    Test if the get_next_action method of MCTS correctly returns the action and probability distribution.
    Verify if the simulation process and final action selection meet expectations.
    """
    mcts, root, mock_env, policy_value_func, legal_actions = mcts_fixture
    mcts.simulate_env = mock_env

    state_config_for_simulation_env_reset = EasyDict({
        'start_player_index': 0,
        'init_state': None,
        'katago_policy_init': False,
        'katago_game_state': None
    })

    # Execute get_next_action
    action, action_probs, root = mcts.get_next_action(
        state_config_for_env_reset=state_config_for_simulation_env_reset,
        policy_value_func=policy_value_func,
        temperature=1.0,
        sample=False,
    )

    # Check if the root node's visit count is 100 (num_simulations)
    assert root.visit_count == 100, f"Root node's visit count should be 100, but got {root.visit_count}"

    # Ensure the returned action is the one with the highest visit count
    max_visits_action = np.argmax(action_probs)
    assert action == max_visits_action, f"The returned action should be the one with the highest visit count ({max_visits_action})"

    # Verify the action probability distribution
    assert len(action_probs) == len(mock_env.legal_actions), "The length of the action probability distribution should match the number of legal actions"
    assert sum(action_probs) == pytest.approx(1.0), "The sum of the action probability distribution should be 1.0"

    # Print results (only for debugging purposes)
    print(f"Action: {action}, Action Probabilities: {action_probs}")