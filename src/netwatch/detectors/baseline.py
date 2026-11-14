"""Layer 1b: baseline detector

Novelty asks "is this process talking to somewhere new?". This detector asks "is it making more
connections, reaching more destinations, or keeping steadier timing than it ever does?".

Each feature gets a robust z-score against the process's own training history:
``(value - median) / (1.4826 * MAD)`` on a log scale. Only the suspicious direction counts (more
connections, more IPs, more ports, more regular timing). The spread has a minimum, so a feature
that never varied in training does not turn a tiny change into a huge score. The window's score
is the largest z, plus a fixed amount for running from a suspicious path the process never used.

The isolation forest cannot do this job: far outside its training range it scores "wildly out of
character" about the same as "a bit unusual".

The forest reuses :class:`ProcessBaseline` for its explanations, so both layers describe
deviations in the same words
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd

from netwatch.detectors.base import Detector
from netwatch.features.windowing import windows_to_frame
from netwatch.models import FeatureWindow

# MAD -> standard deviation for normally distributed data
MAD_TO_STD = 1.4826
# Minimum robust spread for scored features
MIN_SCALE = 0.25
# Robust z where a deviation is worth explaining
EXPLAIN_Z = 3.0
# Score addition when executable is in a suspicious directory
SUSPICIOUS_PATH_SCORE = 6.0
# amount of a process's training windows from a suspicious path, below this counts as new behavior
RARE_SHARE = 0.01
PORT_SCAN_MIN_PORTS = 20
# timing is periodic if below this gap regularity (squared coefficient of variation)
PERIODIC_MAX_VARIANCE = 0.1

# Count features scored by log1p scale, high side only
COUNT_FEATURES = ("connection_count", "unique_remote_ips", "unique_remote_ports")


def log_interval_variance(value: float | np.ndarray) -> float | np.ndarray:
    """timing regularity on a log scale"""
    return np.log10(np.asarray(value, dtype=float) + 1e-3)


def _robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, max(MAD_TO_STD * mad, MIN_SCALE)


@dataclass(frozen=True, slots=True)
class ProcessBaseline:
    """Normal behavior of a process (or entire machine)

    Attributes:
        windows: training windows used for the baseline
        typical: median raw value per count feature (for explanations)
        center: median per scored feature, on scoring (log) scale
        scale: robust spread per scored feature, on scoring scale
        suspicious_share: fraction of windows running from a suspicious path
        hour_share: fraction of windows in each local hour
    """

    windows: int
    typical: dict[str, float]
    center: dict[str, float]
    scale: dict[str, float]
    suspicious_share: float
    hour_share: tuple[float, ...]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> ProcessBaseline:
        center: dict[str, float] = {}
        scale: dict[str, float] = {}
        typical: dict[str, float] = {}
        for feature in COUNT_FEATURES:
            raw = frame[feature].to_numpy(dtype=float)
            typical[feature] = float(np.median(raw))
            center[feature], scale[feature] = _robust_center_scale(np.log1p(raw))
        regularity = log_interval_variance(frame["interval_variance"].to_numpy(dtype=float))
        center["interval_variance"], scale["interval_variance"] = _robust_center_scale(
            np.asarray(regularity)
        )
        hours = np.bincount(frame["hour"].to_numpy(dtype=int), minlength=24)[:24]
        return cls(
            windows=len(frame),
            typical=typical,
            center=center,
            scale=scale,
            suspicious_share=float(frame["exe_in_suspicious_path"].astype(float).mean()),
            hour_share=tuple(float(h) for h in hours / max(len(frame), 1)),
        )

    def deviations(self, window: FeatureWindow) -> dict[str, float]:
        """one-sided robust z per feature"""
        f = window.features
        z = {
            feature: (math.log1p(getattr(f, feature)) - self.center[feature]) / self.scale[feature]
            for feature in COUNT_FEATURES
        }
        z["interval_variance"] = (
            self.center["interval_variance"] - float(log_interval_variance(f.interval_variance))
        ) / self.scale["interval_variance"]
        return {name: max(0.0, value) for name, value in z.items()}

    def new_suspicious_path(self, window: FeatureWindow) -> bool:
        return window.features.exe_in_suspicious_path and self.suspicious_share < RARE_SHARE


def _ratio_text(ratio: float) -> str:
    return f"{ratio:.0f}x" if ratio >= 10 else f"{ratio:.1f}x"


def describe_deviation(feature: str, window: FeatureWindow, baseline: ProcessBaseline) -> str:
    """plain-language explanation for a deviating feature"""
    f = window.features
    if feature == "connection_count":
        typical = baseline.typical["connection_count"]
        ratio = f.connection_count / max(typical, 1.0)
        return (
            f"connection count {_ratio_text(ratio)} above this process's typical window "
            f"({f.connection_count} vs typical {typical:g})"
        )
    if feature == "unique_remote_ips":
        return (
            f"contacted {f.unique_remote_ips} distinct IPs "
            f"(typical: {baseline.typical['unique_remote_ips']:g})"
        )
    if feature == "unique_remote_ports":
        scan = " - possible port scan" if f.unique_remote_ports >= PORT_SCAN_MIN_PORTS else ""
        return (
            f"contacted {f.unique_remote_ports} distinct remote ports "
            f"(typical: {baseline.typical['unique_remote_ports']:g}){scan}"
        )
    if feature == "interval_variance":
        if f.interval_variance < PERIODIC_MAX_VARIANCE:
            return "connections are highly periodic (possible beaconing)"
        return "connection timing is unusually regular for this process"
    raise ValueError(f"no description for feature {feature!r}")


def deviation_reasons(
    window: FeatureWindow, baseline: ProcessBaseline, min_z: float = EXPLAIN_Z
) -> list[str]:
    reasons: list[str] = []
    if baseline.new_suspicious_path(window):
        where = os.path.dirname(window.exe_path) if window.exe_path else "a suspicious path"
        reasons.append(f"executable is running from {where}")
    z = baseline.deviations(window)
    for feature in sorted(z, key=lambda name: -z[name]):
        if z[feature] >= min_z:
            reasons.append(describe_deviation(feature, window, baseline))
    return reasons


class BaselineDetector(Detector):
    """per-process deviation scores on volume, fan-out, timing

    Arguments:
        min_windows_per_process: training windows a process needs for a baseline
    """

    name = "baseline"

    def __init__(self, min_windows_per_process: int = 50) -> None:
        super().__init__()
        self.min_windows_per_process = min_windows_per_process
        self.baselines: dict[str, ProcessBaseline] = {}
        self.global_baseline: ProcessBaseline | None = None

    def fit(self, windows: Sequence[FeatureWindow]) -> Self:
        if not windows:
            raise ValueError("cannot fit the baseline detector on zero windows")
        frame = windows_to_frame(windows)
        self.global_baseline = ProcessBaseline.from_frame(frame)
        self.baselines = {
            str(process): ProcessBaseline.from_frame(frame.iloc[idx])
            for process, idx in frame.groupby("process_name").indices.items()
            if len(idx) >= self.min_windows_per_process
        }
        self._fitted = True
        return self

    def baseline_for(self, process_name: str) -> tuple[ProcessBaseline, bool]:
        """baseline to judge ``process_name`` with"""
        self._require_fitted()
        own = self.baselines.get(process_name)
        if own is not None:
            return own, True
        assert self.global_baseline is not None
        return self.global_baseline, False

    def score_window(self, window: FeatureWindow) -> float:
        baseline, _ = self.baseline_for(window.process_name)
        score = max(baseline.deviations(window).values(), default=0.0)
        if baseline.new_suspicious_path(window):
            score += SUSPICIOUS_PATH_SCORE
        return score

    def score(self, windows: Sequence[FeatureWindow]) -> np.ndarray:
        self._require_fitted()
        return np.array([self.score_window(w) for w in windows], dtype=float)

    def explain(self, window: FeatureWindow) -> list[str]:
        baseline, own = self.baseline_for(window.process_name)
        reasons = deviation_reasons(window, baseline)
        if not reasons:
            z = baseline.deviations(window)
            strongest = max(z, key=lambda name: z[name])
            if z[strongest] > 0:
                reasons.append(describe_deviation(strongest, window, baseline))
            else:
                reasons.append("activity is slightly out of character for this process")
        if not own:
            reasons.append("little or no history for this process; compared against all processes")
        return reasons
