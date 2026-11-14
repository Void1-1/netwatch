"""End to end: synthetic normal traffic + injected attacks -> the attacks get flagged

Runs the same code path as ``netwatch demo`` (and so as ``collect`` -> ``build-windows`` ->
``train`` -> ``watch``) on a shorter training period, then checks detection, explanations,
false-alarm rate and the poisoning rule
"""

from __future__ import annotations

import socket
from collections.abc import Iterator

import pytest

from netwatch.config import Config
from netwatch.demo import DemoReport, run_demo
from netwatch.pipeline import select_training_windows
from netwatch.storage.db import Database


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[DemoReport, Database]]:
    """one shortened demo run (4 training days), shared by every test here"""
    db_path = tmp_path_factory.mktemp("e2e") / "demo.db"
    report = run_demo(Config(), train_days=4, test_days=2, seed=7, db_path=db_path)
    with Database.open(db_path) as db:
        yield report, db


def outcome(report: DemoReport, kind: str):
    return next(o for o in report.outcomes if o.attack.kind == kind)


def test_every_injected_attack_is_detected(demo) -> None:
    report, _ = demo
    assert [o.attack.kind for o in report.outcomes] == [
        "large upload",
        "periodic beacon",
        "/tmp dropper",
        "port scan",
    ]
    missed = [o.attack.kind for o in report.outcomes if not o.detected]
    assert not missed, f"missed attacks: {missed}"


def test_long_running_attacks_alert_in_most_of_their_windows(demo) -> None:
    report, _ = demo
    for kind in ("periodic beacon", "/tmp dropper"):
        result = outcome(report, kind)
        assert result.flagged >= 0.9 * result.windows, kind


@pytest.mark.parametrize(
    ("kind", "expected_reason"),
    [
        ("periodic beacon", "connections are highly periodic (possible beaconing)"),
        ("periodic beacon", "first connection from this process to 203.0.113.7"),
        ("large upload", "first connection from this process to 203.0.113.42"),
        ("large upload", "connection count"),
        ("/tmp dropper", "was never seen during training"),
        ("/tmp dropper", "executable is running from /tmp/.cache"),
        ("port scan", "possible port scan"),
    ],
)
def test_alerts_explain_why_they_fired(demo, kind: str, expected_reason: str) -> None:
    report, _ = demo
    alert = outcome(report, kind).first_alert
    assert alert is not None
    assert any(expected_reason in reason for reason in alert.reasons), alert.reasons


def test_false_alerts_are_rare_and_reported(demo) -> None:
    report, _ = demo
    assert report.normal_test_windows > 1000
    assert report.false_alert_rate < 0.01
    assert all(alert.reasons for alert in report.false_alerts)


def test_poisoning_rule_keeps_attack_windows_out_of_retraining(demo) -> None:
    """retraining on the live days must leave out every attack window

    The simulated reviewer labelled attack alerts suspicious and false alarms expected, so the
    attack windows, and only those, stay out, and the false alarms are re-admitted
    """
    report, db = demo
    selection = select_training_windows(db, days=report.test_days, warmup_hours=24)
    assert selection.excluded_alerted == report.labeled_suspicious
    assert selection.excluded_alerted == sum(o.flagged for o in report.outcomes)
    assert selection.readmitted_expected == report.labeled_expected == len(report.false_alerts)


@pytest.fixture(scope="module")
def unattended(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[DemoReport, Database]]:
    """the same run with nobody labelling alerts (``netwatch demo --no-review``)"""
    db_path = tmp_path_factory.mktemp("e2e-unattended") / "demo.db"
    report = run_demo(Config(), train_days=4, test_days=2, seed=7, review=False, db_path=db_path)
    with Database.open(db_path) as db:
        yield report, db


def test_unattended_repeats_are_attached_not_raised(unattended) -> None:
    """without labels, alerts stay open and repeats of their cause attach to them

    Long-running attacks show it best: each keeps alerting in nearly every window, but only a
    few of those windows raise a new alert
    """
    report, db = unattended
    assert report.all_detected
    assert report.repeats > 0
    for kind in ("periodic beacon", "/tmp dropper"):
        result = outcome(report, kind)
        assert result.flagged >= 0.9 * result.windows, kind
        assert result.alerts < result.flagged, kind
    stored = db.query_alerts()
    assert len(stored) == len(report.false_alerts) + sum(o.alerts for o in report.outcomes)
    assert sum(a.occurrences for a in stored) - len(stored) == report.repeats


def test_unattended_repeats_stay_out_of_retraining(unattended) -> None:
    report, db = unattended
    selection = select_training_windows(db, days=report.test_days, warmup_hours=24)
    alerting = len(report.false_alerts) + report.false_repeats
    assert selection.excluded_alerted == alerting + sum(o.flagged for o in report.outcomes)
    assert selection.readmitted_expected == 0


def test_the_suite_really_has_no_network() -> None:
    with pytest.raises(RuntimeError, match="network access is disabled"):
        socket.create_connection(("192.0.2.1", 443), timeout=0.1)
