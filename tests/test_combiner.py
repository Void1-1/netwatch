"""Score combination: percentiles against history, weighted max, merged reasons"""

from __future__ import annotations

import numpy as np
import pytest

from netwatch.scoring.combiner import ScoreCombiner, merge_reasons


def fitted(history: dict[str, list[float]], processes: list[str] | None = None, **kwargs):
    n = len(next(iter(history.values())))
    names = processes or ["p"] * n
    combiner = ScoreCombiner(**{"min_process_history": 1, **kwargs})
    return combiner.fit(names, {k: np.asarray(v) for k, v in history.items()})


def test_percentile_counts_history_strictly_below() -> None:
    combiner = fitted({"d": [0, 0, 0, 1, 2]})
    # a tie with the most common score ranks at the bottom, not the middle
    assert combiner.percentile("d", "p", 0.0) == 0.0
    assert combiner.percentile("d", "p", 1.0) == pytest.approx(0.6)
    assert combiner.percentile("d", "p", 1.5) == pytest.approx(0.8)
    assert combiner.percentile("d", "p", 99.0) == 1.0


def test_per_process_history_with_global_fallback() -> None:
    processes = ["quiet"] * 50 + ["busy"] * 50 + ["rare"] * 2
    scores = [0.0] * 50 + list(np.linspace(0, 10, 50)) + [5.0, 5.0]
    combiner = fitted({"d": scores}, processes, min_process_history=10)
    # 5.0 is extreme for the quiet process but ordinary for the busy one
    assert combiner.percentile("d", "quiet", 5.0) == 1.0
    assert 0.4 < combiner.percentile("d", "busy", 5.0) < 0.6
    # too little history of its own: ranked against everyone
    assert combiner.history_for("d", "rare").size == len(scores)
    assert combiner.history_for("d", "never-seen").size == len(scores)


def test_weighted_max_and_contributors() -> None:
    history = {"a": list(range(100)), "b": list(range(100))}
    combiner = fitted(history, weights={"a": 1.0, "b": 0.5}, contribution_threshold=0.9)
    (result,) = combiner.combine(["p"], {"a": np.array([50.0]), "b": np.array([1000.0])})
    assert result.percentiles == {"a": 0.5, "b": 1.0}
    # b is above all history but only weighted 0.5, so the max is 0.5
    assert result.combined == pytest.approx(0.5)
    assert result.contributors == ()

    (result,) = combiner.combine(["p"], {"a": np.array([99.5]), "b": np.array([95.0])})
    assert result.combined == pytest.approx(1.0)
    assert result.contributors == ("a",)


def test_contributors_are_sorted_strongest_first() -> None:
    combiner = fitted({"a": list(range(100)), "b": list(range(100))}, contribution_threshold=0.9)
    (result,) = combiner.combine(["p"], {"a": np.array([92.0]), "b": np.array([99.5])})
    assert result.contributors == ("b", "a")


def test_merge_reasons_dedupes_in_order() -> None:
    merged = merge_reasons([["x", "y"], ["y", "z"], [], ["x"]])
    assert merged == ["x", "y", "z"]


def test_combine_before_fit_and_misaligned_history_raise() -> None:
    with pytest.raises(RuntimeError):
        ScoreCombiner().combine(["p"], {"d": np.array([1.0])})
    with pytest.raises(ValueError):
        ScoreCombiner().fit(["p", "q"], {"d": np.array([1.0])})
