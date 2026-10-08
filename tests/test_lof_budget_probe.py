import numpy as np
import pytest

from scripts.probe_lof_budget_selection import select_budget_subset


def test_budget_preserves_feasible_completion_and_improves_spread():
    points = np.array([0.0, 0.01, 0.02, 0.8, 1.0])
    distance = abs(points[:, None] - points[None, :])
    scores = np.array([1.0, 1.01, 1.02, 1.04, 9.0])
    chosen = select_budget_subset(distance, scores, np.ones(5), 3, 0.1)
    assert len(set(chosen)) == 3
    assert 3 in chosen
    assert 4 not in chosen
    assert scores[chosen].mean() <= np.sort(scores)[:3].mean() * 1.1


def test_zero_budget_returns_lowest_lof_subset_and_is_deterministic():
    distances = np.ones((5, 5)) - np.eye(5)
    scores = np.array([1.4, 1.2, 1.0, 1.3, 1.1])
    chosen = select_budget_subset(distances, scores, np.ones(5), 3, 0.0)
    assert set(chosen) == {1, 2, 4}
    assert chosen == select_budget_subset(distances, scores, np.ones(5), 3, 0.0)


def test_reject_invalid_scores():
    with pytest.raises(ValueError):
        select_budget_subset(np.zeros((2, 2)), [1.0, float("nan")], [0.6, 0.7], 1)
