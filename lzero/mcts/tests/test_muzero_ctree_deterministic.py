from lzero.mcts.ctree.ctree_muzero import mz_tree


def test_deterministic_traverse_uses_stable_first_action_for_equal_ucb_scores():
    roots = mz_tree.Roots(1, [[0, 1, 2]])
    roots.prepare_no_noise([0.0], [[0.0, 0.0, 0.0]], [-1])
    min_max_stats = mz_tree.MinMaxStatsList(1)
    min_max_stats.set_delta(0.01)

    selected_actions = []
    for _ in range(5):
        results = mz_tree.ResultsWrapper(1)
        traversal = mz_tree.batch_traverse(
            roots,
            19652,
            1.25,
            0.997,
            min_max_stats,
            results,
            [-1],
            deterministic=True,
        )
        selected_actions.append(traversal[2][0])

    assert selected_actions == [0] * 5


def test_default_traverse_preserves_stochastic_tie_breaking():
    roots = mz_tree.Roots(1, [[0, 1, 2]])
    roots.prepare_no_noise([0.0], [[0.0, 0.0, 0.0]], [-1])
    min_max_stats = mz_tree.MinMaxStatsList(1)
    min_max_stats.set_delta(0.01)

    selected_actions = []
    for _ in range(30):
        results = mz_tree.ResultsWrapper(1)
        traversal = mz_tree.batch_traverse(
            roots,
            19652,
            1.25,
            0.997,
            min_max_stats,
            results,
            [-1],
        )
        selected_actions.append(traversal[2][0])

    assert len(set(selected_actions)) > 1
