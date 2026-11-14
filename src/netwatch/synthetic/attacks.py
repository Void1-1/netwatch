"""Labelled attack scenarios to inject into synthetic traffic

Each injector returns the attack's events plus an :class:`InjectedAttack` record of the ground
truth (which process, from when to when). The demo and the end-to-end test use it to check that
the attack was flagged, and count every other alert as a false alarm.

Each scenario is tied to the features that should expose it:

==================  ==================================  ==========================
Scenario            What it looks like                  Should be caught by
==================  ==================================  ==========================
periodic beacon     an existing app phones a new host   periodic timing, new IP
                    every ~30 s
large upload        a burst of parallel connections     connection-count spike,
                    to a new host (exfiltration)        new IP
port scan           one process touches 150 ports on    distinct-port spike, new
                    a LAN host in 90 s                  ports
/tmp dropper        a never-seen binary in /tmp calls   unseen process, suspicious
                    home on an odd port                 path, new IP and port
==================  ==================================  ==========================

Byte counts are not collected yet, so the "large upload" is modelled as what it looks like at the
connection level: many parallel connections. Attacker addresses use 203.0.113.0/24 (RFC 5737
TEST-NET-3)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from netwatch.models import ConnectionEvent
from netwatch.synthetic.generator import make_event, profile_by_name, quantize

HOUR = 3600
DAY = 86_400


@dataclass(frozen=True, slots=True)
class InjectedAttack:
    """ground truth for one injected attack

    Attributes:
        kind: short scenario name, e.g. ``"periodic beacon"``
        process_name: process the attack's connections belong to
        start: first attack connection (epoch seconds)
        end: just after the last attack connection
        description: one line for reports
    """

    kind: str
    process_name: str
    start: float
    end: float
    description: str

    def overlaps(self, process_name: str, window_start: float, window_end: float) -> bool:
        """if a window of ``process_name`` contains part of this attack"""
        return (
            process_name == self.process_name
            and window_start < self.end
            and window_end > self.start
        )


@dataclass(frozen=True, slots=True)
class AttackInjection:
    """an attack's events plus its ground-truth record"""

    events: list[ConnectionEvent]
    attack: InjectedAttack


def _finish(
    kind: str, process: str, events: list[ConnectionEvent], description: str
) -> AttackInjection:
    events.sort(key=lambda e: e.timestamp)
    start, end = events[0].timestamp, events[-1].timestamp + 1
    return AttackInjection(events, InjectedAttack(kind, process, start, end, description))


def periodic_beacon(
    start: float,
    *,
    process_name: str = "chatapp",
    c2_ip: str = "203.0.113.7",
    port: int = 443,
    interval_seconds: float = 30.0,
    jitter_seconds: float = 1.5,
    duration_seconds: float = 2 * HOUR,
    seed: int = 0,
    collector_interval: float = 5.0,
) -> AttackInjection:
    """a compromised app checking in with its command-and-control server on a timer"""
    rng = np.random.default_rng(seed)
    profile = profile_by_name(process_name)
    events: list[ConnectionEvent] = []
    t = start
    while t < start + duration_seconds:
        stamp = quantize(t + rng.normal(0, jitter_seconds), collector_interval)
        events.append(make_event(profile, stamp, c2_ip, port, rng))
        t += interval_seconds
    return _finish(
        "periodic beacon",
        process_name,
        events,
        f"{process_name} beacons to {c2_ip}:{port} every ~{interval_seconds:g}s",
    )


def large_upload(
    start: float,
    *,
    process_name: str = "mailclient",
    dest_ip: str = "203.0.113.42",
    port: int = 443,
    connections: int = 60,
    duration_seconds: float = 8 * 60,
    seed: int = 1,
    collector_interval: float = 5.0,
) -> AttackInjection:
    """exfiltration: many parallel connections to a new host within a few minutes"""
    rng = np.random.default_rng(seed)
    profile = profile_by_name(process_name)
    stamps = np.sort(rng.uniform(start, start + duration_seconds, size=connections))
    events = [
        make_event(profile, quantize(float(t), collector_interval), dest_ip, port, rng)
        for t in stamps
    ]
    return _finish(
        "large upload",
        process_name,
        events,
        f"{process_name} opens {connections} connections to {dest_ip}:{port}",
    )


def port_scan(
    start: float,
    *,
    process_name: str = "python3",
    target_ip: str = "192.168.1.10",
    port_count: int = 150,
    duration_seconds: float = 90.0,
    seed: int = 2,
    collector_interval: float = 5.0,
) -> AttackInjection:
    """reconnaissance: one process probing many ports on a LAN host"""
    rng = np.random.default_rng(seed)
    profile = profile_by_name(process_name)
    ports = rng.choice(np.arange(1, 10_000), size=port_count, replace=False)
    stamps = np.sort(rng.uniform(start, start + duration_seconds, size=port_count))
    events = [
        # most probes hit closed ports and never finish the handshake
        make_event(
            profile,
            quantize(float(t), collector_interval),
            target_ip,
            int(p),
            rng,
            status="SYN_SENT",
        )
        for t, p in zip(stamps, ports, strict=True)
    ]
    return _finish(
        "port scan",
        process_name,
        events,
        f"{process_name} probes {port_count} ports on {target_ip}",
    )


def tmp_dropper(
    start: float,
    *,
    process_name: str = "updater_helper",
    exe_path: str = "/tmp/.cache/updater_helper",
    parent_name: str = "bash",
    c2_ip: str = "203.0.113.66",
    port: int = 8443,
    interval_seconds: float = 90.0,
    jitter_seconds: float = 15.0,
    duration_seconds: float = 3 * HOUR,
    seed: int = 3,
    collector_interval: float = 5.0,
) -> AttackInjection:
    """a newly dropped binary running from /tmp and phoning home"""
    rng = np.random.default_rng(seed)
    events: list[ConnectionEvent] = []
    t = start
    while t < start + duration_seconds:
        stamp = quantize(t + rng.normal(0, jitter_seconds), collector_interval)
        events.append(
            make_event(
                process_name,
                stamp,
                c2_ip,
                port,
                rng,
                exe_path=exe_path,
                parent_name=parent_name,
                pid=31337,
            )
        )
        t += interval_seconds
    return _finish(
        "/tmp dropper",
        process_name,
        events,
        f"{exe_path} (never seen before) phones home to {c2_ip}:{port}",
    )


def standard_attack_suite(test_start: float, seed: int = 0) -> list[AttackInjection]:
    """the four reference attacks, spread over the first two test days

    Times are local and vary the context: one at night, one in the middle of a busy workday,
    and so on. Needs a test period of at least two days that starts at local midnight
    """
    return [
        large_upload(test_start + 3 * HOUR + 20 * 60, seed=seed + 1),
        periodic_beacon(test_start + 13 * HOUR + 10 * 60, seed=seed),
        tmp_dropper(test_start + DAY + 1 * HOUR + 30 * 60, seed=seed + 3),
        port_scan(test_start + DAY + 11 * HOUR + 5 * 60, seed=seed + 2),
    ]


def inject(
    events: Iterable[ConnectionEvent], injections: Iterable[AttackInjection]
) -> list[ConnectionEvent]:
    """merge attack events into normal traffic, keeping time order"""
    merged = list(events)
    for injection in injections:
        merged.extend(injection.events)
    merged.sort(key=lambda e: (e.timestamp, e.process_name))
    return merged
