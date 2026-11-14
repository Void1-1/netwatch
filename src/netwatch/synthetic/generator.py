"""Generate realistic "normal" network traffic for several fake processes

The aim is not perfect realism but the *things that make detection hard*, so the demo and the
end-to-end test mean something:

* Processes differ a lot: a browser is busy and keeps meeting new CDN addresses, a sync daemon
  polls the same host every minute around the clock, an updater wakes up at 3 a.m., a dev tool
  only runs on some days.
* Activity follows a daily rhythm and is lighter at weekends.
* Connections come in bursts (a page load opens several at once), so timing is irregular for
  most processes but truly periodic for some.
* New destinations keep appearing at a rate that depends on the process, and recent ones are
  revisited, like real CDN and API churn.
* Timestamps are rounded down to the collector's snapshot interval, like the ones
  :class:`~netwatch.collector.psutil_collector.PsutilCollector` records.

Addresses (all reserved, never routed on the internet):
    * 192.0.2.0/24 and 198.51.100.0/24 (RFC 5737): each process's stable "home" destinations
    * 198.18.0.0/15 (RFC 2544): an endless supply of fresh addresses for churn
    * 203.0.113.0/24 (RFC 5737): injected attackers (see :mod:`netwatch.synthetic.attacks`)

Everything is driven by a seeded numpy ``Generator``, so the same seed always gives the same
traffic
"""

from __future__ import annotations

import ipaddress
import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from netwatch.models import ConnectionEvent

SECONDS_PER_DAY = 86_400
LOCAL_IP = "192.168.1.23"
CHURN_BASE = ipaddress.IPv4Address("198.18.0.1")
CHURN_SIZE = 2**17 - 2  # usable addresses in 198.18.0.0/15
EPHEMERAL_PORTS = (49152, 65535)
# a Monday, in local time. Fixed so runs are reproducible. In January, so a run of several
# weeks does not cross a daylight-saving change
DEFAULT_START = datetime(2026, 1, 5)


def default_start() -> float:
    """epoch seconds of local midnight on :data:`DEFAULT_START`"""
    return DEFAULT_START.timestamp()


def activity_curve(
    start_hour: int, end_hour: int, day_level: float = 1.0, night_level: float = 0.03
) -> tuple[float, ...]:
    """24 hourly activity levels: ``day_level`` in [start, end), half-way at the edges"""
    levels = []
    for hour in range(24):
        if start_hour <= hour < end_hour:
            levels.append(day_level)
        elif hour in (start_hour - 1, end_hour):
            levels.append((day_level + night_level) / 2)
        else:
            levels.append(night_level)
    return tuple(levels)


def _ips(prefix: str, first: int, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}.{i}" for i in range(first, first + count))


@dataclass(frozen=True, slots=True)
class PeriodicTask:
    """a connection a process makes at a fixed interval (polling, heartbeats)"""

    remote_ip: str
    remote_port: int
    interval_seconds: float
    jitter_seconds: float


@dataclass(frozen=True, slots=True)
class ProcessProfile:
    """how one fake process behaves

    Attributes:
        name: process name
        exe_path: executable path (all look legitimate)
        parent_name: parent process name
        hourly_activity: 24 multipliers on ``sessions_per_window``, by local hour
        sessions_per_window: mean bursts of activity per fully active window
        mean_burst: mean connections per burst
        home_ips: stable destinations. Earlier ones are contacted more often (Zipf-like)
        ports: ``(port, weight)`` pairs a burst's port is drawn from
        new_destination_rate: chance a burst goes to a never-seen address
        revisit_rate: chance a burst revisits a recently new address
        weekend_factor: activity multiplier on Saturday and Sunday
        active_day_probability: chance the process runs at all on a given day
        udp_ports: ports that use UDP instead of TCP
        periodic: fixed-interval tasks that run around the clock
        pid: fixed fake PID
    """

    name: str
    exe_path: str
    parent_name: str
    hourly_activity: tuple[float, ...]
    sessions_per_window: float
    mean_burst: float
    home_ips: tuple[str, ...]
    ports: tuple[tuple[int, float], ...]
    new_destination_rate: float
    revisit_rate: float = 0.15
    weekend_factor: float = 1.0
    active_day_probability: float = 1.0
    udp_ports: frozenset[int] = frozenset()
    periodic: tuple[PeriodicTask, ...] = ()
    pid: int = 1000


DEFAULT_PROFILES: tuple[ProcessProfile, ...] = (
    ProcessProfile(
        name="browser",
        exe_path="/Applications/Browser.app/Contents/MacOS/Browser",
        parent_name="launchd",
        hourly_activity=activity_curve(8, 23),
        sessions_per_window=4.0,
        mean_burst=3.5,
        home_ips=_ips("192.0.2", 10, 25),
        ports=((443, 0.95), (80, 0.05)),
        new_destination_rate=0.12,
        weekend_factor=1.2,
        pid=812,
    ),
    ProcessProfile(
        name="mailclient",
        exe_path="/Applications/MailClient.app/Contents/MacOS/MailClient",
        parent_name="launchd",
        hourly_activity=activity_curve(7, 22, night_level=0.05),
        sessions_per_window=0.4,
        mean_burst=2.0,
        home_ips=_ips("192.0.2", 60, 4),
        ports=((993, 0.6), (587, 0.2), (443, 0.2)),
        new_destination_rate=0.03,
        weekend_factor=0.5,
        periodic=(PeriodicTask("192.0.2.60", 993, interval_seconds=300, jitter_seconds=10),),
        pid=845,
    ),
    ProcessProfile(
        name="chatapp",
        exe_path="/Applications/ChatApp.app/Contents/MacOS/ChatApp",
        parent_name="launchd",
        hourly_activity=activity_curve(9, 19),
        sessions_per_window=1.0,
        mean_burst=2.0,
        home_ips=_ips("198.51.100", 20, 6),
        ports=((443, 0.9), (3478, 0.1)),
        new_destination_rate=0.02,
        weekend_factor=0.2,
        udp_ports=frozenset({3478}),
        pid=903,
    ),
    ProcessProfile(
        name="cloudsync",
        exe_path="/Applications/CloudSync.app/Contents/MacOS/cloudsyncd",
        parent_name="launchd",
        hourly_activity=activity_curve(8, 20, day_level=1.0, night_level=0.2),
        sessions_per_window=0.4,
        mean_burst=3.0,
        home_ips=_ips("198.51.100", 50, 5),
        ports=((443, 1.0),),
        new_destination_rate=0.01,
        # a sync client polls its API every minute: truly periodic, which is exactly what a
        # naive "periodic = beacon" rule gets wrong
        periodic=(PeriodicTask("198.51.100.50", 443, interval_seconds=60, jitter_seconds=2),),
        pid=377,
    ),
    ProcessProfile(
        name="softwareupdated",
        exe_path="/usr/libexec/softwareupdated",
        parent_name="launchd",
        # mostly idle, with a scheduled check around 03:00 every night
        hourly_activity=tuple(1.0 if h == 3 else 0.02 for h in range(24)),
        sessions_per_window=1.5,
        mean_burst=4.0,
        home_ips=_ips("192.0.2", 200, 3),
        ports=((443, 1.0),),
        new_destination_rate=0.05,
        pid=221,
    ),
    ProcessProfile(
        name="python3",
        exe_path="/usr/local/bin/python3",
        parent_name="zsh",
        hourly_activity=activity_curve(10, 18, night_level=0.0),
        sessions_per_window=0.8,
        mean_burst=3.0,
        home_ips=_ips("198.51.100", 100, 5),
        ports=((443, 0.95), (22, 0.05)),
        new_destination_rate=0.1,
        weekend_factor=0.1,
        active_day_probability=0.7,
        pid=5120,
    ),
)


def profile_by_name(
    name: str, profiles: tuple[ProcessProfile, ...] = DEFAULT_PROFILES
) -> ProcessProfile:
    for profile in profiles:
        if profile.name == name:
            return profile
    raise KeyError(f"no synthetic process profile named {name!r}")


def quantize(timestamp: float, collector_interval: float) -> float:
    """round a timestamp down to the collector's snapshot grid"""
    if collector_interval <= 0:
        return timestamp
    return math.floor(timestamp / collector_interval) * collector_interval


def make_event(
    profile_or_name: ProcessProfile | str,
    timestamp: float,
    remote_ip: str,
    remote_port: int,
    rng: np.random.Generator,
    *,
    exe_path: str | None = None,
    parent_name: str | None = None,
    pid: int | None = None,
    protocol: str = "tcp",
    status: str | None = None,
) -> ConnectionEvent:
    """one synthetic :class:`ConnectionEvent` with a random ephemeral local port"""
    if isinstance(profile_or_name, ProcessProfile):
        name = profile_or_name.name
        exe_path = exe_path or profile_or_name.exe_path
        parent_name = parent_name or profile_or_name.parent_name
        pid = pid if pid is not None else profile_or_name.pid
    else:
        name = profile_or_name
    return ConnectionEvent(
        timestamp=timestamp,
        pid=pid,
        process_name=name,
        exe_path=exe_path,
        parent_name=parent_name,
        local_ip=LOCAL_IP,
        local_port=int(rng.integers(EPHEMERAL_PORTS[0], EPHEMERAL_PORTS[1] + 1)),
        remote_ip=remote_ip,
        remote_port=remote_port,
        protocol=protocol,
        status=status or ("ESTABLISHED" if protocol == "tcp" else "NONE"),
    )


class TrafficGenerator:
    """seeded generator of normal traffic for a set of process profiles"""

    def __init__(
        self,
        profiles: tuple[ProcessProfile, ...] = DEFAULT_PROFILES,
        seed: int = 0,
        window_seconds: int = 300,
        collector_interval: float = 5.0,
    ) -> None:
        self.profiles = profiles
        self.rng = np.random.default_rng(seed)
        self.window_seconds = window_seconds
        self.collector_interval = collector_interval
        self._fresh_counter = 0
        self._recent_new: dict[str, list[str]] = {p.name: [] for p in profiles}
        # Zipf-like popularity: the k-th home destination gets weight 1/k
        self._home_weights = {
            p.name: (w := 1.0 / np.arange(1, len(p.home_ips) + 1)) / w.sum() for p in profiles
        }

    # destinations

    def _fresh_ip(self) -> str:
        """next unused churn address from 198.18.0.0/15"""
        address = CHURN_BASE + (self._fresh_counter % CHURN_SIZE)
        self._fresh_counter += 1
        return str(address)

    def _pick_destination(self, profile: ProcessProfile) -> str:
        recent = self._recent_new[profile.name]
        roll = self.rng.random()
        if roll < profile.new_destination_rate:
            ip = self._fresh_ip()
            recent.append(ip)
            del recent[:-50]  # keep only the 50 most recent new addresses
            return ip
        if roll < profile.new_destination_rate + profile.revisit_rate and recent:
            return recent[int(self.rng.integers(len(recent)))]
        index = int(self.rng.choice(len(profile.home_ips), p=self._home_weights[profile.name]))
        return profile.home_ips[index]

    def _pick_port(self, profile: ProcessProfile) -> int:
        ports = [p for p, _ in profile.ports]
        weights = np.array([w for _, w in profile.ports], dtype=float)
        return int(ports[int(self.rng.choice(len(ports), p=weights / weights.sum()))])

    # generation

    def _event(
        self, profile: ProcessProfile, timestamp: float, ip: str, port: int
    ) -> ConnectionEvent:
        protocol = "udp" if port in profile.udp_ports else "tcp"
        return make_event(
            profile,
            quantize(timestamp, self.collector_interval),
            ip,
            port,
            self.rng,
            protocol=protocol,
        )

    def _sessions(self, profile: ProcessProfile, start: float, end: float) -> list[ConnectionEvent]:
        """bursty, human-driven activity for one profile between start and end"""
        events: list[ConnectionEvent] = []
        windows_per_day = SECONDS_PER_DAY // self.window_seconds
        days = math.ceil((end - start) / SECONDS_PER_DAY)
        for day in range(days):
            day_start = start + day * SECONDS_PER_DAY
            if self.rng.random() >= profile.active_day_probability:
                continue  # this process does not run today
            weekend = datetime.fromtimestamp(day_start).weekday() >= 5
            day_factor = profile.weekend_factor if weekend else 1.0
            for slot in range(windows_per_day):
                window_start = day_start + slot * self.window_seconds
                hour = (slot * self.window_seconds) // 3600
                expected = profile.sessions_per_window * profile.hourly_activity[hour] * day_factor
                for _ in range(int(self.rng.poisson(expected))):
                    t = window_start + self.rng.uniform(0, self.window_seconds)
                    ip, port = self._pick_destination(profile), self._pick_port(profile)
                    for _ in range(int(self.rng.geometric(1.0 / profile.mean_burst))):
                        if t < end:
                            events.append(self._event(profile, t, ip, port))
                        t += self.rng.exponential(1.5)  # seconds between connections in a burst
        return events

    def _periodic(self, profile: ProcessProfile, start: float, end: float) -> list[ConnectionEvent]:
        """fixed-interval connections (polls, heartbeats) that run around the clock"""
        events: list[ConnectionEvent] = []
        for task in profile.periodic:
            t = start + self.rng.uniform(0, task.interval_seconds)
            while t < end:
                jittered = t + self.rng.normal(0, task.jitter_seconds)
                events.append(self._event(profile, jittered, task.remote_ip, task.remote_port))
                t += task.interval_seconds
        return events

    def generate(self, start: float, days: int) -> list[ConnectionEvent]:
        """normal traffic for every profile over ``days`` days from ``start``

        ``start`` should be a local midnight (like :func:`default_start`), so hourly activity
        lines up with the local clock
        """
        end = start + days * SECONDS_PER_DAY
        events: list[ConnectionEvent] = []
        for profile in self.profiles:
            events.extend(self._sessions(profile, start, end))
            events.extend(self._periodic(profile, start, end))
        events = [e for e in events if start <= e.timestamp < end]
        events.sort(key=lambda e: (e.timestamp, e.process_name))
        return events


def generate_normal_traffic(
    start: float,
    days: int,
    seed: int = 0,
    profiles: tuple[ProcessProfile, ...] = DEFAULT_PROFILES,
    collector_interval: float = 5.0,
) -> list[ConnectionEvent]:
    """shortcut for :class:`TrafficGenerator` when only the events are needed"""
    generator = TrafficGenerator(profiles, seed=seed, collector_interval=collector_interval)
    return generator.generate(start, days)
