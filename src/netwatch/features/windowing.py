"""Group events into fixed time windows per process and compute their features

Why windows: one connection is too small to judge (a browser contacting a new IP is normal). Five
minutes of one process's activity has a shape (how many connections, to how many places, how
regular), and an alert still points at a specific moment.

Why per process: each window is compared only with that process's own history. Machine-wide
windows would let a busy browser hide a quiet implant.

Windows are aligned to multiples of ``window_seconds`` since the epoch, so an event always lands
in the same window no matter when windowing runs. Only window *closed* time slots: a slot that is
still filling would be undercounted
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd

from netwatch.features.encoders import (
    DEFAULT_SUSPICIOUS_PATHS,
    NEUTRAL_INTERVAL_VARIANCE,
    cyclic_hour,
    hour_of_day,
    normalized_interval_variance,
    suspicious_path_match,
)
from netwatch.features.novelty import NoveltyTracker, PairKind
from netwatch.models import ConnectionEvent, FeatureWindow, WindowFeatures


def window_start_for(timestamp: float, window_seconds: int) -> float:
    """start of the aligned window that contains ``timestamp``"""
    return float(math.floor(timestamp / window_seconds) * window_seconds)


def group_events(
    events: Iterable[ConnectionEvent], window_seconds: int
) -> dict[tuple[float, str], list[ConnectionEvent]]:
    """group events by ``(window_start, process_name)``, each group sorted by time"""
    groups: dict[tuple[float, str], list[ConnectionEvent]] = defaultdict(list)
    for event in events:
        groups[(window_start_for(event.timestamp, window_seconds), event.process_name)].append(
            event
        )
    for bucket in groups.values():
        bucket.sort(key=lambda e: e.timestamp)
    return dict(groups)


def periodicity(events: Sequence[ConnectionEvent], min_events: int) -> float:
    """lowest normalised interval variance among the window's destinations

    Measured per remote IP, not over all events: a compromised app still makes its normal,
    irregular connections, and those would hide a beacon's steady rhythm. The minimum asks "is
    *any* destination contacted on a timer?". Neutral value when no destination repeats enough
    """
    by_destination: dict[str, list[float]] = defaultdict(list)
    for event in events:
        by_destination[event.remote_ip].append(event.timestamp)
    scores = [
        score
        for stamps in by_destination.values()
        if (score := normalized_interval_variance(stamps, min_events)) is not None
    ]
    return min(scores, default=NEUTRAL_INTERVAL_VARIANCE)


def _representative_exe(events: Sequence[ConnectionEvent], patterns: Sequence[str]) -> str | None:
    """exe path to report for the window

    One process name can cover several binaries (and malware likes to borrow legitimate names),
    so a suspicious path wins over the most common one
    """
    paths = [e.exe_path for e in events if e.exe_path]
    for path in paths:
        if suspicious_path_match(path, patterns):
            return path
    return Counter(paths).most_common(1)[0][0] if paths else None


@dataclass(frozen=True, slots=True)
class WindowBuilder:
    """turns events into :class:`FeatureWindow` records

    Attributes:
        window_seconds: window length
        suspicious_paths: directories that make an executable suspicious
        min_periodic_events: repeats needed before timing regularity is judged
    """

    window_seconds: int = 300
    suspicious_paths: tuple[str, ...] = DEFAULT_SUSPICIOUS_PATHS
    min_periodic_events: int = 4

    def compute_window(
        self,
        process_name: str,
        window_start: float,
        events: Sequence[ConnectionEvent],
        tracker: NoveltyTracker,
    ) -> FeatureWindow:
        """compute one window's features. Novelty is checked but not committed"""
        remote_ips = sorted({e.remote_ip for e in events})
        remote_ports = sorted({e.remote_port for e in events})
        new_ips = tracker.novel(process_name, PairKind.IP, remote_ips, window_start)
        new_ports = tracker.novel(process_name, PairKind.PORT, remote_ports, window_start)
        exe_path = _representative_exe(events, self.suspicious_paths)
        # time of day at the window's midpoint, not at its start
        hour_sin, hour_cos = cyclic_hour(window_start + self.window_seconds / 2)

        features = WindowFeatures(
            connection_count=len(events),
            unique_remote_ips=len(remote_ips),
            unique_remote_ports=len(remote_ports),
            new_ip_count=len(new_ips),
            new_port_count=len(new_ports),
            hour_sin=hour_sin,
            hour_cos=hour_cos,
            interval_variance=periodicity(events, self.min_periodic_events),
            exe_in_suspicious_path=suspicious_path_match(exe_path, self.suspicious_paths)
            is not None,
        )
        return FeatureWindow(
            process_name=process_name,
            window_start=window_start,
            window_end=window_start + self.window_seconds,
            features=features,
            exe_path=exe_path,
            remote_ips=tuple(remote_ips),
            remote_ports=tuple(remote_ports),
            new_remote_ips=tuple(new_ips),
            new_remote_ports=tuple(new_ports),
        )

    def build(
        self,
        events: Iterable[ConnectionEvent],
        tracker: NoveltyTracker,
        *,
        update_tracker: bool = True,
    ) -> list[FeatureWindow]:
        """build windows for ``events``, oldest first

        Arguments:
            events: events from closed time slots, in any order
            tracker: novelty memory that decides what counts as new
            update_tracker: commit each window's destinations to ``tracker`` right after
                computing it. Batch building (before a model exists) does this. The live monitor
                passes ``False`` and commits only alert-free windows itself, so an alert's
                destinations stay new unless the user labels the alert expected

        Order matters: what counts as new depends on everything committed before
        """
        groups = group_events(events, self.window_seconds)
        windows: list[FeatureWindow] = []
        for window_start, process_name in sorted(groups):
            window = self.compute_window(
                process_name, window_start, groups[(window_start, process_name)], tracker
            )
            if update_tracker:
                tracker.observe_window(window)
            windows.append(window)
        return windows


def windows_to_frame(windows: Sequence[FeatureWindow]) -> pd.DataFrame:
    """windows as a table: one row each, with process, timing and every feature

    Adds an integer ``hour`` column (local hour of the window's midpoint), which is easier to
    use for per-hour baselines than sin/cos
    """
    columns = ["process_name", "window_start", "hour", *WindowFeatures.names()]
    rows = [
        {
            "process_name": w.process_name,
            "window_start": w.window_start,
            "hour": int(hour_of_day((w.window_start + w.window_end) / 2)) % 24,
            **w.features.to_dict(),
        }
        for w in windows
    ]
    return pd.DataFrame(rows, columns=columns)
