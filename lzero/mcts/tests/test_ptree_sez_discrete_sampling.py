import numpy as np
import pytest
import torch

from lzero.mcts.ptree.ptree_sez import Action, Node


def _action_value(action):
    return int(np.asarray(action.value).item())


@pytest.mark.unittest
def test_ptree_sez_samples_only_legal_actions_and_uses_action_indexed_logits():
    torch.manual_seed(0)
    node = Node(0, legal_actions=[1, 4], action_space_size=5, num_of_sampled_actions=2)

    node.expand(-1, 0, 0, 0.0, [100.0, -10.0, 90.0, 80.0, 10.0])

    sampled_actions = [_action_value(action) for action in node.legal_actions]
    assert set(sampled_actions) == {1, 4}
    assert len(node.children) == 2

    action_1_prior = float(node.get_child(Action(np.array(1))).prior)
    action_4_prior = float(node.get_child(Action(np.array(4))).prior)
    assert action_4_prior > action_1_prior


@pytest.mark.unittest
def test_ptree_sez_pads_sampled_actions_and_distributions_when_legal_count_is_smaller_than_k():
    torch.manual_seed(0)
    node = Node(0, legal_actions=[2], action_space_size=5, num_of_sampled_actions=4)

    node.expand(-1, 0, 0, 0.0, [0.0, 1.0, 2.0, 3.0, 4.0])

    assert [_action_value(action) for action in node.legal_actions] == [2]
    assert [_action_value(action) for action in node.sampled_actions] == [2, 2, 2, 2]

    node.add_exploration_noise(0.25, [0.25, 0.25, 0.25, 0.25])
    node.get_child(node.legal_actions[0]).visit_count = 3
    assert node.get_children_distribution() == [3, 0, 0, 0]


@pytest.mark.unittest
def test_ctree_sampled_efficientzero_discrete_sampling_respects_legal_actions():
    ezs_tree = pytest.importorskip("lzero.mcts.ctree.ctree_sampled_efficientzero.ezs_tree")

    roots = ezs_tree.Roots(2, [[1, 4], [2]], 5, 4, False)
    roots.prepare(
        0.25,
        [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]],
        [0.0, 0.0],
        [
            [100.0, -10.0, 90.0, 80.0, 10.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        ],
        [-1, -1],
    )

    sampled_actions = roots.get_sampled_actions()
    distributions = roots.get_distributions()

    assert len(sampled_actions[0]) == 4
    assert len(distributions[0]) == 4
    assert set(int(action[0]) for action in sampled_actions[0]) == {1, 4}
    assert len(set(int(action[0]) for action in sampled_actions[0][:2])) == 2

    assert len(sampled_actions[1]) == 4
    assert len(distributions[1]) == 4
    assert [int(action[0]) for action in sampled_actions[1]] == [2, 2, 2, 2]
