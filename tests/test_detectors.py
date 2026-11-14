"""Detectors: the shared interface, each implemented layer, and the stubs"""

from __future__ import annotations

import math

import numpy as np
import pytest

from netwatch.config import DetectorsConfig
from netwatch.detectors import (
    AutoencoderDetector,
    BaselineDetector,
    Detector,
    IsolationForestDetector,
    NotFittedError,
    NoveltyDetector,
    SequenceDetector,
    build_detectors,
)
from netwatch.detectors.novelty_detector import poisson_tail_surprise

IMPLEMENTED = [NoveltyDetector, BaselineDetector, IsolationForestDetector]


# shared interface


@pytest.mark.parametrize("cls", IMPLEMENTED)
def test_scoring_before_fit_raises(cls: type[Detector], make_window) -> None:
    with pytest.raises(NotFittedError):
        cls().score([make_window()])


@pytest.mark.parametrize("cls", IMPLEMENTED)
def test_score_shape_and_explain_type(cls: type[Detector], normal_windows, make_window) -> None:
    detector = cls().fit(normal_windows)
    scores = detector.score(normal_windows[:10])
    assert scores.shape == (10,)
    assert np.isfinite(scores).all()
    odd = make_window(
        "daemon", connection_count=80, new_ip_count=1, new_remote_ips=("203.0.113.7",)
    )
    reasons = detector.explain(odd)
    assert reasons
    assert all(isinstance(r, str) and r for r in reasons)


@pytest.mark.parametrize("cls", [AutoencoderDetector, SequenceDetector])
def test_planned_layers_are_explicit_stubs(cls: type[Detector], make_window) -> None:
    detector = cls()
    with pytest.raises(NotImplementedError, match="planned"):
        detector.fit([make_window()])
    with pytest.raises(NotImplementedError):
        detector.score([make_window()])
    with pytest.raises(NotImplementedError):
        detector.explain(make_window())


def test_build_detectors_follows_config() -> None:
    config = DetectorsConfig()
    assert [d.name for d in build_detectors(config)] == [
        "novelty",
        "baseline",
        "isolation_forest",
    ]
    config.baseline.enabled = False
    assert [d.name for d in build_detectors(config)] == ["novelty", "isolation_forest"]


# Layer 1: novelty


def test_poisson_tail_surprise() -> None:
    assert poisson_tail_surprise(0, 0.5) == 0.0
    # P(X >= 1) = 1 - e^-rate
    assert poisson_tail_surprise(1, 0.5) == pytest.approx(-math.log10(1 - math.exp(-0.5)))
    # more new destinations is always more surprising. Huge counts stay finite
    values = [poisson_tail_surprise(k, 0.2) for k in range(1, 8)]
    assert values == sorted(values)
    assert math.isfinite(poisson_tail_surprise(500, 0.01))
    assert poisson_tail_surprise(500, 0.01) > 100


def test_a_new_ip_is_more_surprising_for_a_quiet_process(normal_windows, make_window) -> None:
    detector = NoveltyDetector().fit(normal_windows)
    quiet, busy = detector.score(
        [make_window("daemon", new_ip_count=1), make_window("browser", new_ip_count=1)]
    )
    assert quiet > busy
    assert busy < 1.0  # one new IP is routine for the browser


def test_unseen_processes_get_extra_surprise(normal_windows, make_window) -> None:
    detector = NoveltyDetector(unseen_process_surprise=4.0).fit(normal_windows)
    known, unseen = detector.score(
        [make_window("daemon", new_ip_count=1), make_window("dropper", new_ip_count=1)]
    )
    assert unseen > known
    assert "never seen during training" in detector.explain(make_window("dropper"))[0]


def test_novelty_explain_names_new_destinations_and_summarises_many(
    normal_windows, make_window
) -> None:
    detector = NoveltyDetector().fit(normal_windows)
    few = detector.explain(
        make_window(
            "daemon",
            new_ip_count=1,
            new_port_count=1,
            new_remote_ips=("203.0.113.7",),
            new_remote_ports=(8443,),
        )
    )
    assert "first connection from this process to 203.0.113.7" in few
    assert "first connection from this process on port 8443 (https alt)" in few

    ports = tuple(range(1, 151))
    many = detector.explain(make_window("daemon", new_port_count=150, new_remote_ports=ports))
    assert "first connections on 150 new ports (e.g. 1, 2, 3)" in many


# Layer 1b: baseline


def test_baseline_flags_volume_spikes_relative_to_the_process(normal_windows, make_window) -> None:
    detector = BaselineDetector().fit(normal_windows)
    typical, spike = detector.score(
        [make_window("daemon", connection_count=2), make_window("daemon", connection_count=60)]
    )
    assert spike > typical + 3
    reasons = detector.explain(make_window("daemon", connection_count=60))
    assert any(r.startswith("connection count") and "typical window" in r for r in reasons)


def test_baseline_flags_beacon_timing_and_port_scans(normal_windows, make_window) -> None:
    detector = BaselineDetector().fit(normal_windows)
    beacon = make_window("daemon", connection_count=3, interval_variance=0.01)
    assert "connections are highly periodic (possible beaconing)" in detector.explain(beacon)
    scan = make_window("daemon", connection_count=150, unique_remote_ports=150)
    assert any("possible port scan" in r for r in detector.explain(scan))


def test_baseline_flags_a_new_suspicious_path(normal_windows, make_window) -> None:
    detector = BaselineDetector().fit(normal_windows)
    window = make_window("daemon", exe_in_suspicious_path=True, exe_path="/tmp/.x/daemon")
    assert detector.score([window])[0] >= 6.0
    assert "executable is running from /tmp/.x" in detector.explain(window)


def test_baseline_uses_machine_wide_history_for_unknown_processes(
    normal_windows, make_window
) -> None:
    detector = BaselineDetector().fit(normal_windows)
    _, own = detector.baseline_for("dropper")
    assert own is False
    assert any("compared against all processes" in r for r in detector.explain(make_window("x")))


# Layer 2: isolation forest


def test_isolation_forest_scores_odd_windows_higher(normal_windows, make_window) -> None:
    detector = IsolationForestDetector(n_estimators=50).fit(normal_windows)
    normal = detector.score(normal_windows[:200:2]).mean()  # daemon windows
    odd = detector.score(
        [make_window("daemon", connection_count=40, unique_remote_ips=12, new_ip_count=6)]
    )[0]
    assert odd > normal


def test_isolation_forest_builds_per_process_models_with_fallback(
    normal_windows, make_window
) -> None:
    detector = IsolationForestDetector(n_estimators=20, min_windows_per_process=50)
    detector.fit([*normal_windows, make_window("rare", connection_count=1)])
    assert set(detector.models) == {"daemon", "browser"}
    # "rare" has one window: scored by the shared forest, and says so
    assert detector.score([make_window("rare")]).shape == (1,)
    assert any("compared against all processes" in r for r in detector.explain(make_window("rare")))
