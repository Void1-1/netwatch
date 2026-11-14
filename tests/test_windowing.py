"""Windowing and feature encoders: events in, per-process feature windows out"""

from __future__ import annotations

import math

import pytest

from netwatch.features.encoders import (
    NEUTRAL_INTERVAL_VARIANCE,
    cyclic_hour,
    describe_port,
    normalized_interval_variance,
    port_bucket,
    suspicious_path_match,
)
from netwatch.features.novelty import NoveltyTracker
from netwatch.features.windowing import (
    WindowBuilder,
    group_events,
    window_start_for,
    windows_to_frame,
)
from netwatch.models import WindowFeatures

# encoders


def test_cyclic_hour_is_on_the_unit_circle_and_wraps_at_midnight(noon: float) -> None:
    sin, cos = cyclic_hour(noon)
    assert sin**2 + cos**2 == pytest.approx(1.0)
    assert cos == pytest.approx(-1.0)  # noon is opposite midnight
    # 23:59 and 00:01 should be neighbours, not 24 hours apart
    midnight = noon - 12 * 3600
    before = cyclic_hour(midnight - 60)
    after = cyclic_hour(midnight + 60)
    assert math.dist(before, after) < 0.01


def test_port_buckets_and_descriptions() -> None:
    assert port_bucket(443) == "well-known"
    assert port_bucket(8443) == "registered"
    assert port_bucket(60000) == "dynamic"
    assert describe_port(443) == "443 (https)"
    assert describe_port(4444) == "4444 (registered range)"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/tmp/x/payload", "/tmp"),
        ("/private/tmp/payload", "/private/tmp"),
        ("/tmpfoo/bin/app", None),  # whole path components only
        ("/usr/bin/python3", None),
        (None, None),
    ],
)
def test_suspicious_path_match(path: str | None, expected: str | None) -> None:
    assert suspicious_path_match(path) == expected


def test_interval_variance_separates_clockwork_from_bursts() -> None:
    clockwork = [i * 30.0 for i in range(10)]
    bursty = [0.0, 1.0, 2.0, 60.0, 61.0, 200.0]
    assert normalized_interval_variance(clockwork) == pytest.approx(0.0)
    assert normalized_interval_variance(bursty) > 1.0
    # too few events, or all at one instant, is not a rhythm
    assert normalized_interval_variance([0.0, 30.0]) is None
    assert normalized_interval_variance([5.0] * 6) is None


# grouping


def test_window_start_alignment() -> None:
    assert window_start_for(1_000_000.0, 300) == 999_900.0
    assert window_start_for(999_900.0, 300) == 999_900.0


def test_events_are_grouped_per_process_and_window(make_event, noon) -> None:
    start = window_start_for(noon, 300)
    events = [
        make_event(start + 10, "a"),
        make_event(start + 20, "b"),
        make_event(start + 299, "a"),
        make_event(start + 300, "a"),  # next window
    ]
    groups = group_events(events, 300)
    assert {key: len(v) for key, v in groups.items()} == {
        (start, "a"): 2,
        (start, "b"): 1,
        (start + 300, "a"): 1,
    }


# features


def test_window_features_count_connections_peers_and_ports(make_event, noon) -> None:
    start = window_start_for(noon, 300)
    events = [
        make_event(start + 1, remote_ip="192.0.2.1", remote_port=443),
        make_event(start + 2, remote_ip="192.0.2.1", remote_port=443),
        make_event(start + 3, remote_ip="192.0.2.2", remote_port=80),
    ]
    (window,) = WindowBuilder().build(events, NoveltyTracker())
    f = window.features
    assert (f.connection_count, f.unique_remote_ips, f.unique_remote_ports) == (3, 2, 2)
    assert window.remote_ips == ("192.0.2.1", "192.0.2.2")
    assert window.remote_ports == (80, 443)
    assert window.window_end - window.window_start == 300
    # not enough repeats to judge timing: neutral value
    assert f.interval_variance == NEUTRAL_INTERVAL_VARIANCE


def test_periodicity_is_measured_per_destination(make_event, noon) -> None:
    """a beacon running alongside normal, irregular traffic must still show up"""
    start = window_start_for(noon, 300)
    beacon = [make_event(start + 5 + 30 * i, remote_ip="203.0.113.7") for i in range(9)]
    noise = [make_event(start + t, remote_ip="192.0.2.9") for t in (2, 3, 4, 150, 151)]
    (window,) = WindowBuilder().build(beacon + noise, NoveltyTracker())
    assert window.features.interval_variance < 0.05


def test_first_window_is_all_new_then_familiar_once_committed(make_event, noon) -> None:
    start = window_start_for(noon, 300)
    tracker = NoveltyTracker()
    builder = WindowBuilder()
    first = builder.build([make_event(start + 1)], tracker)
    second = builder.build([make_event(start + 301)], tracker)
    assert first[0].features.new_ip_count == 1
    assert first[0].new_remote_ips == ("192.0.2.10",)
    assert first[0].features.new_port_count == 1
    assert second[0].features.new_ip_count == 0


def test_update_tracker_false_leaves_novelty_memory_untouched(make_event, noon) -> None:
    start = window_start_for(noon, 300)
    tracker = NoveltyTracker()
    builder = WindowBuilder()
    builder.build([make_event(start + 1)], tracker, update_tracker=False)
    assert len(tracker) == 0
    again = builder.build([make_event(start + 301)], tracker, update_tracker=False)
    assert again[0].features.new_ip_count == 1


def test_suspicious_executable_wins_as_the_window_exe(make_event, noon) -> None:
    """a binary that borrows a legitimate name but runs from /tmp must not go unnoticed"""
    start = window_start_for(noon, 300)
    events = [
        make_event(start + 1, "updater", exe_path="/usr/libexec/updater"),
        make_event(start + 2, "updater", exe_path="/usr/libexec/updater"),
        make_event(start + 3, "updater", exe_path="/tmp/.x/updater"),
    ]
    (window,) = WindowBuilder().build(events, NoveltyTracker())
    assert window.features.exe_in_suspicious_path is True
    assert window.exe_path == "/tmp/.x/updater"


def test_windows_to_frame_has_every_feature(make_window) -> None:
    frame = windows_to_frame([make_window("a"), make_window("b")])
    assert list(frame["process_name"]) == ["a", "b"]
    assert set(WindowFeatures.names()) <= set(frame.columns)
    assert frame["hour"].between(0, 23).all()
