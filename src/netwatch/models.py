"""core data records between stages of netwatch's pipeline

Stages only talk to each other through dataclasses defined here.

conventions:
    * Timestamps are POSIX epoch seconds (``float``)
    * Behaviour is keyed by ``process_name``
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from typing import Any, NamedTuple


@dataclass(frozen=True, slots=True)
class ConnectionEvent:
    """a network connection observed on this machine, with its process

    process attribution (``pid``, ``process_name``, ``exe_path``, ``parent_name``) is key

    attributes:
        timestamp: when the connection was first observed
        pid: process id at observation time, or ``None``
        process_name: short executable name
        exe_path: full executable path if readable
        parent_name: name of parent process (``launchd``, ``zsh``, ...)
        local_ip: this machine's side of the socket
        local_port: this machine's port (usually ephemeral for outbound traffic)
        remote_ip: peer being talked to
        remote_port: peer's port; for outbound traffic, the service port
        protocol: ``"tcp"`` or ``"udp"``
        status: Socket state from psutil (``ESTABLISHED``, ``SYN_SENT``, ``NONE``)
    """

    timestamp: float
    pid: int | None
    process_name: str
    exe_path: str | None
    parent_name: str | None
    local_ip: str
    local_port: int
    remote_ip: str
    remote_port: int
    protocol: str
    status: str


@dataclass(frozen=True, slots=True)
class WindowFeatures:
    """summary of a process's network behaviour in a time window

    attributes:
        connection_count: new connections opened in the window
        unique_remote_ips: distinct peers contacted
        unique_remote_ports: distinct peer ports contacted
        new_ip_count: peers this process has not contacted before
        new_port_count: peer ports this process has not used before
        hour_sin: sine of time-of-day angle (see ``hour_cos``)
        hour_cos: cosine of time-of-day angle (used with sin for circular relationship)
        interval_variance: normalised variance of the gaps between repeated connections to the
            same peer (near 0 = clockwork)
        exe_in_suspicious_path: whether the executable runs from a suspicious directory, such
            as ``/tmp``
    """

    connection_count: int
    unique_remote_ips: int
    unique_remote_ports: int
    new_ip_count: int
    new_port_count: int
    hour_sin: float
    hour_cos: float
    interval_variance: float
    exe_in_suspicious_path: bool

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """feature names in canonical (vector) order"""
        return tuple(f.name for f in fields(cls))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WindowFeatures:
        return cls(**{name: data[name] for name in cls.names()})


@dataclass(frozen=True, slots=True)
class FeatureWindow:
    """a process's activity over a fixed time window

    ``features`` scored by detectors. Remaining fields are *context*

    attributes:
        process_name: process belonging to the window
        window_start: inclusive start (epoch seconds), aligned to window size
        window_end: exclusive end (epoch seconds)
        features: numeric feature vector
        exe_path: representative executable path
        remote_ips: every distinct peer IP contacted, sorted
        remote_ports: every distinct peer port contacted, sorted
        new_remote_ips: subset of ``remote_ips`` that was new
        new_remote_ports: subset of ``remote_ports`` that was new
    """

    process_name: str
    window_start: float
    window_end: float
    features: WindowFeatures
    exe_path: str | None = None
    remote_ips: tuple[str, ...] = ()
    remote_ports: tuple[int, ...] = ()
    new_remote_ips: tuple[str, ...] = ()
    new_remote_ports: tuple[int, ...] = ()

    @property
    def key(self) -> tuple[str, float]:
        """identity of a window: ``(process_name, window_start)``"""
        return (self.process_name, self.window_start)

    def context_dict(self) -> dict[str, Any]:
        """context fields as a JSON-serialisable dict"""
        return {
            "exe_path": self.exe_path,
            "remote_ips": list(self.remote_ips),
            "remote_ports": list(self.remote_ports),
            "new_remote_ips": list(self.new_remote_ips),
            "new_remote_ports": list(self.new_remote_ports),
        }


class AlertStatus(StrEnum):
    """alert lifecycle. ``OPEN`` until labelled"""

    OPEN = "open"
    EXPECTED = "expected"
    SUSPICIOUS = "suspicious"


class Verdict(StrEnum):
    """user's judgement of an alert"""

    EXPECTED = "expected"
    SUSPICIOUS = "suspicious"


@dataclass(slots=True)
class Alert:
    """An explained claim that one process window looked anomalous.

    An alert can cover more than one window: later windows of the same process that only
    repeat its cause are attached to it

    attributes:
        timestamp: when the alert was raised (epoch seconds)
        process_name: process whose window alerted
        window_start: start of the window that raised the alert
        combined_score: final score in [0, 1] that crossed the threshold
        per_detector_scores: each detector's percentile vs. history, by name
        reasons: plain explanations, most important first
        status: open, or latest user verdict
        id: database id, ``None`` until stored
        repeat_window_starts: starts of the later windows attached to this alert as repeats,
            oldest first
    """

    timestamp: float
    process_name: str
    window_start: float
    combined_score: float
    per_detector_scores: dict[str, float]
    reasons: list[str]
    status: AlertStatus = AlertStatus.OPEN
    id: int | None = None
    repeat_window_starts: list[float] = field(default_factory=list)

    @property
    def window_keys(self) -> list[tuple[str, float]]:
        """every window covered by this alert, as ``(process_name, window_start)``"""
        starts = [self.window_start, *self.repeat_window_starts]
        return [(self.process_name, start) for start in starts]

    @property
    def occurrences(self) -> int:
        """how many windows this alert covers: the first and repeats"""
        return 1 + len(self.repeat_window_starts)

    @property
    def last_window_start(self) -> float:
        """start of the most recent window this alert covers"""
        return max([self.window_start, *self.repeat_window_starts])


@dataclass(frozen=True, slots=True)
class Label:
    """a user's verdict on an alert

    append-only: re-labelling an alert adds a new row and the most recent label wins
    """

    alert_id: int
    verdict: Verdict
    timestamp: float
    note: str | None = None


class NoveltyRecord(NamedTuple):
    """a remembered (process, kind, value) pair from novelty memory"""

    process_name: str
    kind: str
    value: str
    familiarity: float
    last_seen: float
    first_seen: float
