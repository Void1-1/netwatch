"""Small stateless helpers shared by windowing and explanations

Each one turns a raw value into something a model, or a person reading an alert, can use
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Sequence
from datetime import datetime

import numpy as np

# Time of day


def hour_of_day(timestamp: float) -> float:
    """local hour as a fraction in [0, 24), e.g. 13.5 for 13:30

    Local time, not UTC: people's habits follow the local clock
    """
    dt = datetime.fromtimestamp(timestamp)
    return dt.hour + dt.minute / 60 + dt.second / 3600


def cyclic_hour(timestamp: float) -> tuple[float, float]:
    """time of day as a point on a circle: ``(sin, cos)``

    A plain 0-23 hour makes 23:00 and 00:00 look far apart. On a circle they are neighbours
    """
    angle = 2 * math.pi * hour_of_day(timestamp) / 24
    return math.sin(angle), math.cos(angle)


# Ports

# top of the IANA well-known and registered port ranges
WELL_KNOWN_MAX = 1023
REGISTERED_MAX = 49151

# a few common services, only used to make explanations readable
COMMON_SERVICES: dict[int, str] = {
    22: "ssh",
    25: "smtp",
    53: "dns",
    80: "http",
    123: "ntp",
    143: "imap",
    443: "https",
    465: "smtps",
    587: "smtp submission",
    853: "dns over tls",
    993: "imaps",
    3478: "stun",
    5223: "push",
    8080: "http alt",
    8443: "https alt",
}


def port_bucket(port: int) -> str:
    """IANA range of a port: well-known, registered or dynamic"""
    if port <= WELL_KNOWN_MAX:
        return "well-known"
    if port <= REGISTERED_MAX:
        return "registered"
    return "dynamic"


def describe_port(port: int) -> str:
    """readable port, e.g. ``"443 (https)"`` or ``"4444 (registered range)"``"""
    service = COMMON_SERVICES.get(port)
    return f"{port} ({service})" if service else f"{port} ({port_bucket(port)} range)"


# Executable location

DEFAULT_SUSPICIOUS_PATHS: tuple[str, ...] = (
    "/tmp",
    "/private/tmp",
    "/var/tmp",
    "/private/var/tmp",
    "/dev/shm",
    "~/Downloads",
)
# macOS per-user temp dirs under /var/folders are left out on purpose: installers and updaters
# run from there all the time, so flagging them would mostly be noise


def suspicious_path_match(
    exe_path: str | None, patterns: Iterable[str] = DEFAULT_SUSPICIOUS_PATHS
) -> str | None:
    """first suspicious directory ``exe_path`` is under, or ``None``

    Whole path components only: ``/tmpfoo/bin`` does not match ``/tmp``. ``~`` in a pattern means
    the current user's home
    """
    if not exe_path:
        return None
    path = os.path.normpath(exe_path)
    for pattern in patterns:
        prefix = os.path.normpath(os.path.expanduser(pattern))
        if path == prefix or path.startswith(prefix.rstrip("/") + "/"):
            return pattern
    return None


# Timing regularity

# neutral value: what fully random timing (a Poisson process) scores
NEUTRAL_INTERVAL_VARIANCE = 1.0


def normalized_interval_variance(timestamps: Sequence[float], min_events: int = 4) -> float | None:
    """how uneven the gaps between ``timestamps`` are: ``var(gaps) / mean(gaps)^2``

    The length of the gaps does not matter: a 30-second beacon and a 10-minute beacon both score
    near 0, and bursty human-driven traffic scores around 1 or higher. ``None`` when there is
    nothing to judge: too few events, or all of them at the same instant
    """
    if len(timestamps) < min_events:
        return None
    gaps = np.diff(np.sort(np.asarray(timestamps, dtype=float)))
    mean = float(gaps.mean())
    if mean <= 0:
        return None
    return float(gaps.var() / mean**2)
