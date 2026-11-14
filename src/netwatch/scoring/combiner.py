"""Combine raw detector scores into one alert score, and merge their reasons

The problem: detectors use different scales. The novelty score is a surprise in -log10 units (0
to ~10), the isolation forest's is roughly 0.3 to 0.8. Averaging them directly means nothing.

How:
    1. Turn each raw score into a *percentile* against that detector's own training history:
       "this window is stranger than 99.9% of the windows this detector saw in training".
       History is per process where there is enough of it (a busy browser's normal noise should
       not set the bar for a quiet daemon), machine-wide otherwise.
    2. Combine with a *weighted max*: ``max(weight_i * percentile_i)``. A max, not a mean: the
       layers look for different things, and an attack only one layer sees must not be averaged
       away by the others.

       A weight is a switch with three positions, not a dial, because a percentile tops out at
       1.0. At 1.0 the detector counts in full. Below the alert threshold it can no longer
       raise an alert by itself (``0.99 * 1.0 < 0.999``), but while it is still at or above
       ``contribution_threshold`` it adds its reasons to alerts other detectors raise. Below
       that it is silent. Weights above 1 are rejected by the config: they would lift
       percentiles under the threshold over it.
    3. Detectors whose weighted percentile reaches ``contribution_threshold`` are
       *contributors*. Only their reasons are shown in the alert
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Self

import numpy as np


@dataclass(frozen=True, slots=True)
class CombinedScore:
    """the combiner's result for one window

    Attributes:
        combined: weighted-max percentile in [0, 1]
        percentiles: unweighted percentile per detector
        contributors: detectors that pushed the score up, strongest first
    """

    combined: float
    percentiles: dict[str, float]
    contributors: tuple[str, ...] = field(default=())


def merge_reasons(reason_lists: Iterable[Iterable[str]]) -> list[str]:
    """join reasons from several detectors, in order, without duplicates"""
    seen: set[str] = set()
    merged: list[str] = []
    for reasons in reason_lists:
        for reason in reasons:
            if reason not in seen:
                seen.add(reason)
                merged.append(reason)
    return merged


class ScoreCombiner:
    """turns raw scores into percentiles and takes their weighted max

    Arguments:
        weights: weight per detector name (1.0 if missing)
        contribution_threshold: weighted percentile at which a detector counts as a
            contributor to an alert
        min_process_history: training windows a process needs for its own percentile history.
            Below this the machine-wide history is used
    """

    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        contribution_threshold: float = 0.95,
        min_process_history: int = 50,
    ) -> None:
        self.weights = dict(weights or {})
        self.contribution_threshold = contribution_threshold
        self.min_process_history = min_process_history
        self._global: dict[str, np.ndarray] = {}
        self._per_process: dict[str, dict[str, np.ndarray]] = {}

    def fit(self, process_names: Sequence[str], raw_scores: Mapping[str, np.ndarray]) -> Self:
        """record each detector's score history from its training windows

        Arguments:
            process_names: process of each training window
            raw_scores: detector name -> raw scores, in the same order as ``process_names``
        """
        processes = np.asarray(process_names, dtype=object)
        self._global, self._per_process = {}, {}
        for name, scores in raw_scores.items():
            scores = np.asarray(scores, dtype=float)
            if scores.shape != processes.shape:
                raise ValueError(
                    f"{name}: got {scores.shape[0]} scores for {len(processes)} windows"
                )
            self._global[name] = np.sort(scores)
            self._per_process[name] = {
                str(process): np.sort(scores[processes == process])
                for process in set(process_names)
                if np.count_nonzero(processes == process) >= self.min_process_history
            }
        return self

    @property
    def detector_names(self) -> list[str]:
        return list(self._global)

    def history_for(self, detector: str, process_name: str) -> np.ndarray:
        """sorted training scores that ``process_name`` is ranked against"""
        own = self._per_process.get(detector, {}).get(process_name)
        return own if own is not None else self._global[detector]

    def percentile(self, detector: str, process_name: str, raw: float) -> float:
        """fraction of historical scores *strictly below* ``raw``

        Strictly: most windows have a novelty score of exactly 0, and a window that ties the most
        common value should rank at the bottom, not in the middle
        """
        history = self.history_for(detector, process_name)
        if history.size == 0:
            return 0.0
        return float(np.searchsorted(history, raw, side="left") / history.size)

    def combine(
        self, process_names: Sequence[str], raw_scores: Mapping[str, np.ndarray]
    ) -> list[CombinedScore]:
        """one :class:`CombinedScore` per window, from raw scores in window order"""
        if not self._global:
            raise RuntimeError("ScoreCombiner must be fit() before combine()")
        results: list[CombinedScore] = []
        for i, process in enumerate(process_names):
            percentiles = {
                name: self.percentile(name, process, float(scores[i]))
                for name, scores in raw_scores.items()
            }
            weighted = {name: self.weights.get(name, 1.0) * p for name, p in percentiles.items()}
            contributors = tuple(
                sorted(
                    (n for n, v in weighted.items() if v >= self.contribution_threshold),
                    key=lambda n: -weighted[n],
                )
            )
            results.append(
                CombinedScore(
                    combined=max(weighted.values(), default=0.0),
                    percentiles=percentiles,
                    contributors=contributors,
                )
            )
        return results
