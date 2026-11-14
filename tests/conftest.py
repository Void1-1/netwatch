"""Shared test fixtures

Two things hold for the whole suite:

* **No network.** An autouse fixture makes every attempt to open an outbound socket fail loudly,
  so "netwatch never phones home" is checked on every test run, not just promised.
* **No root, no real data.** Tests build records by hand or with the synthetic generator, and
  databases are in memory or under ``tmp_path``
"""

from __future__ import annotations

import socket
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

from netwatch.models import ConnectionEvent, FeatureWindow, WindowFeatures
from netwatch.synthetic.generator import default_start

# a fixed Monday 12:00 in local time, far from DST changes. Built from the local clock so
# time-of-day features behave the same in every timezone
NOON = default_start() + 12 * 3600


@pytest.fixture(autouse=True, scope="session")
def _no_network() -> Iterator[None]:
    """fail every outbound connection attempt, for the whole session

    Session-scoped, so it is already active when module-scoped fixtures (such as the
    end-to-end demo run) execute
    """

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("network access is disabled in the netwatch test suite")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket.socket, "connect", refuse)
        patch.setattr(socket.socket, "connect_ex", refuse)
        patch.setattr(socket, "create_connection", refuse)
        yield


@pytest.fixture
def noon() -> float:
    return NOON


@pytest.fixture
def make_event() -> Callable[..., ConnectionEvent]:
    """factory for ConnectionEvents with sensible defaults"""

    def factory(
        timestamp: float = NOON,
        process_name: str = "app",
        remote_ip: str = "192.0.2.10",
        remote_port: int = 443,
        **overrides: Any,
    ) -> ConnectionEvent:
        fields: dict[str, Any] = {
            "timestamp": timestamp,
            "pid": 100,
            "process_name": process_name,
            "exe_path": f"/usr/bin/{process_name}",
            "parent_name": "launchd",
            "local_ip": "192.168.1.23",
            "local_port": 50000,
            "remote_ip": remote_ip,
            "remote_port": remote_port,
            "protocol": "tcp",
            "status": "ESTABLISHED",
        }
        fields.update(overrides)
        return ConnectionEvent(**fields)

    return factory


@pytest.fixture
def make_window() -> Callable[..., FeatureWindow]:
    """factory for FeatureWindows: pass any feature or context field by name"""

    def factory(
        process_name: str = "app",
        window_start: float = NOON,
        **values: Any,
    ) -> FeatureWindow:
        features: dict[str, Any] = {
            "connection_count": 3,
            "unique_remote_ips": 1,
            "unique_remote_ports": 1,
            "new_ip_count": 0,
            "new_port_count": 0,
            "hour_sin": 0.0,
            "hour_cos": -1.0,
            "interval_variance": 1.0,
            "exe_in_suspicious_path": False,
        }
        context: dict[str, Any] = {"exe_path": f"/usr/bin/{process_name}"}
        for key, value in values.items():
            if key in features:
                features[key] = value
            else:
                context[key] = value
        return FeatureWindow(
            process_name=process_name,
            window_start=window_start,
            window_end=window_start + 300,
            features=WindowFeatures(**features),
            **context,
        )

    return factory


@pytest.fixture
def normal_windows(make_window: Callable[..., FeatureWindow]) -> list[FeatureWindow]:
    """plausible training history: a quiet daemon and a busy browser that keeps meeting new IPs

    300 windows each, spread over the day, with small random variation
    """
    rng = np.random.default_rng(0)
    windows: list[FeatureWindow] = []
    for i in range(300):
        start = NOON + i * 300
        angle = 2 * np.pi * (i % 288) / 288
        sin, cos = float(np.sin(angle)), float(np.cos(angle))
        windows.append(
            make_window(
                "daemon",
                start,
                connection_count=int(rng.integers(1, 4)),
                unique_remote_ips=1,
                unique_remote_ports=1,
                new_ip_count=int(rng.random() < 0.01),
                hour_sin=sin,
                hour_cos=cos,
            )
        )
        ips = int(rng.integers(2, 8))
        windows.append(
            make_window(
                "browser",
                start,
                connection_count=ips * int(rng.integers(2, 5)),
                unique_remote_ips=ips,
                unique_remote_ports=int(rng.integers(1, 3)),
                new_ip_count=int(rng.poisson(1.5)),
                hour_sin=sin,
                hour_cos=cos,
                interval_variance=float(rng.uniform(0.6, 3.0)),
            )
        )
    return windows
