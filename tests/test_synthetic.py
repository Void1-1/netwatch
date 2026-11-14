"""Synthetic traffic: reproducible, privacy-safe, and shaped like real behaviour"""

from __future__ import annotations

import ipaddress

from netwatch.features.novelty import NoveltyTracker
from netwatch.features.windowing import WindowBuilder
from netwatch.synthetic.attacks import inject, standard_attack_suite
from netwatch.synthetic.generator import DEFAULT_PROFILES, default_start, generate_normal_traffic

DAY = 86_400
RESERVED = [
    ipaddress.ip_network(n)
    for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24", "198.18.0.0/15")
]
LAN = ipaddress.ip_network("192.168.0.0/16")


def test_same_seed_same_traffic() -> None:
    a = generate_normal_traffic(default_start(), days=1, seed=3)
    b = generate_normal_traffic(default_start(), days=1, seed=3)
    c = generate_normal_traffic(default_start(), days=1, seed=4)
    assert a == b
    assert a != c


def test_traffic_is_ordered_quantised_and_bounded() -> None:
    start = default_start()
    events = generate_normal_traffic(start, days=1, seed=0)
    assert {e.process_name for e in events} <= {p.name for p in DEFAULT_PROFILES}
    assert [e.timestamp for e in events] == sorted(e.timestamp for e in events)
    assert all(start <= e.timestamp < start + DAY for e in events)
    assert all(e.timestamp % 5 == 0 for e in events)  # collector snapshot grid


def test_every_address_is_reserved_and_never_routable() -> None:
    start = default_start()
    events = inject(
        generate_normal_traffic(start, days=2, seed=0), standard_attack_suite(start, seed=0)
    )
    for event in events:
        address = ipaddress.ip_address(event.remote_ip)
        assert any(address in net for net in RESERVED) or address in LAN, event.remote_ip


def test_attack_suite_ground_truth_matches_its_events() -> None:
    start = default_start()
    injections = standard_attack_suite(start)
    assert len(injections) == 4
    for injection in injections:
        attack = injection.attack
        assert injection.events
        assert all(e.process_name == attack.process_name for e in injection.events)
        assert all(attack.start <= e.timestamp < attack.end for e in injection.events)
    dropper = next(i for i in injections if i.attack.kind == "/tmp dropper")
    assert all(e.exe_path.startswith("/tmp/") for e in dropper.events)


def test_sync_daemon_is_genuinely_periodic() -> None:
    """a legitimate process is periodic too, so periodic timing alone must not mean beacon"""
    events = generate_normal_traffic(default_start(), days=1, seed=0)
    windows = WindowBuilder().build(
        [e for e in events if e.process_name == "cloudsync"], NoveltyTracker()
    )
    periodic = [w for w in windows if w.features.interval_variance < 0.1]
    assert len(periodic) > 0.8 * len(windows)
