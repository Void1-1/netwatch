"""Layer 2: isolation forest over window features

Catches windows that are unusual as a *combination* (normal destinations and volume, but at
3 a.m. and on a new port), not because one feature is extreme. A single extreme feature is the
baseline detector's job: a forest cannot tell how far outside its training range a value is.

Each process with enough training windows gets its own forest. The rest share one machine-wide
forest.

A forest gives no reasons, so ``explain`` uses the same per-process baselines as the baseline
detector, adds a time-of-day check, and otherwise says it is the combination that is unusual
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Self

import numpy as np
from sklearn.ensemble import IsolationForest

from netwatch.detectors.base import Detector
from netwatch.detectors.baseline import ProcessBaseline, deviation_reasons
from netwatch.features.encoders import hour_of_day
from netwatch.features.windowing import windows_to_frame
from netwatch.models import FeatureWindow

# An hour with less than this share of a process's windows is unusual
RARE_HOUR_SHARE = 0.005
# History needed before any hour can be unusual
MIN_WINDOWS_FOR_HOURS = 100


def feature_matrix(windows: Sequence[FeatureWindow]) -> np.ndarray:
    """model inputs"""
    rows = [
        (
            math.log1p(f.connection_count),
            math.log1p(f.unique_remote_ips),
            math.log1p(f.unique_remote_ports),
            math.log1p(f.new_ip_count),
            math.log1p(f.new_port_count),
            f.hour_sin,
            f.hour_cos,
            math.log10(f.interval_variance + 1e-3),
            float(f.exe_in_suspicious_path),
        )
        for f in (w.features for w in windows)
    ]
    return np.asarray(rows, dtype=float).reshape(len(rows), 9)


class IsolationForestDetector(Detector):
    """Per-process isolation forests with machine-wide fallback

    Arguments:
        n_estimators: trees/forest
        min_windows_per_process: training windows a process needs to get a forest
        random_state: seed, so training is reproducible
    """

    name = "isolation_forest"

    def __init__(
        self,
        n_estimators: int = 100,
        min_windows_per_process: int = 50,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.min_windows_per_process = min_windows_per_process
        self.random_state = random_state
        self.models: dict[str, IsolationForest] = {}
        self.baselines: dict[str, ProcessBaseline] = {}
        self.global_model: IsolationForest | None = None
        self.global_baseline: ProcessBaseline | None = None

    def _new_forest(self) -> IsolationForest:
        return IsolationForest(n_estimators=self.n_estimators, random_state=self.random_state)

    def fit(self, windows: Sequence[FeatureWindow]) -> Self:
        if len(windows) < 2:
            raise ValueError("the isolation forest needs at least 2 training windows")
        X = feature_matrix(windows)
        frame = windows_to_frame(windows)

        self.global_model = self._new_forest().fit(X)
        self.global_baseline = ProcessBaseline.from_frame(frame)
        self.models, self.baselines = {}, {}
        for process, idx in frame.groupby("process_name").indices.items():
            if len(idx) < self.min_windows_per_process:
                continue  # too little history -> shared forest
            self.models[str(process)] = self._new_forest().fit(X[idx])
            self.baselines[str(process)] = ProcessBaseline.from_frame(frame.iloc[idx])
        self._fitted = True
        return self

    def score(self, windows: Sequence[FeatureWindow]) -> np.ndarray:
        self._require_fitted()
        assert self.global_model is not None
        X = feature_matrix(windows)
        scores = np.zeros(len(windows), dtype=float)
        by_process: dict[str, list[int]] = defaultdict(list)
        for i, w in enumerate(windows):
            by_process[w.process_name].append(i)
        for process, idx in by_process.items():
            model = self.models.get(process, self.global_model)
            # sklearn: lower score_samples = more abnormal. Flip the sign so higher is worse
            scores[idx] = -model.score_samples(X[idx])
        return scores

    def _baseline_for(self, process_name: str) -> tuple[ProcessBaseline, bool]:
        """baseline to explain against"""
        own = self.baselines.get(process_name)
        if own is not None:
            return own, True
        assert self.global_baseline is not None
        return self.global_baseline, False

    def explain(self, window: FeatureWindow) -> list[str]:
        self._require_fitted()
        baseline, own = self._baseline_for(window.process_name)
        reasons = deviation_reasons(window, baseline)

        hour = int(hour_of_day((window.window_start + window.window_end) / 2)) % 24
        if (
            own
            and baseline.windows >= MIN_WINDOWS_FOR_HOURS
            and baseline.hour_share[hour] < RARE_HOUR_SHARE
        ):
            span = f"{hour:02d}:00-{(hour + 1) % 24:02d}:00"
            reasons.append(f"active at an unusual hour for this process ({span})")
        if not reasons:
            reasons.append("unusual combination of activity for this process")
        if not own:
            reasons.append("little or no history for this process; compared against all processes")
        return reasons
