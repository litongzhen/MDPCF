import numpy as np
import pytest

from scripts.probe_pareto_selection import select_pareto_subset


def test_swap_improves_diversity_without_cost_or_probability_regression():
    distances = np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]])
    costs = np.array([[1, 1], [2, 2], [1, 1]])
    probabilities = np.array([.8, .8, .9])
    selected, swaps = select_pareto_subset(distances, costs, probabilities, [0, 1])
    assert selected == [0, 2]
    assert swaps == 1
    assert (costs[selected].sum(axis=0) <= costs[[0, 1]].sum(axis=0)).all()
    assert probabilities[selected].sum() >= probabilities[[0, 1]].sum()


def test_infeasible_improvement_keeps_reference():
    distances = np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]])
    selected, swaps = select_pareto_subset(distances, [[1, 1], [1, 1], [2, 1]],
                                         [.9, .9, .8], [0, 1])
    assert selected == [0, 1]
    assert swaps == 0


def test_rejects_nonfinite_metrics():
    with pytest.raises(ValueError, match="finite"):
        select_pareto_subset([[0, np.nan], [np.nan, 0]], [[1], [1]], [.8, .9], [0])
