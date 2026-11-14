"""Alert deduplication: the rule on its own, and repeats inside the live Monitor

The Monitor scenarios use two tiny test detectors so the cause of every alert is obvious: ``new``
fires on any new destination (and, like the real novelty detector, only explains destinations),
``busy`` fires on a burst of connections and gives a reason of its own
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest

from netwatch.alerts.base import CollectingSink
from netwatch.alerts.dedup import OpenAlert, covers, find_repeated, is_repeat, reason_kind
from netwatch.detectors.base import Detector
from netwatch.features.novelty import NoveltyTracker, PairKind
from netwatch.features.windowing import WindowBuilder, window_start_for
from netwatch.feedback.labels import build_training_set, record_label
from netwatch.models import Alert, AlertStatus, ConnectionEvent, FeatureWindow, Verdict
from netwatch.pipeline import Monitor, TrainedModel
from netwatch.scoring.combiner import ScoreCombiner
from netwatch.storage.db import Database

C2 = "203.0.113.7"
OTHER = "203.0.113.8"
KNOWN = "192.0.2.1"
BUSY = 5


# the rule on its own


def open_alert(window: FeatureWindow, reasons: list[str], **overrides) -> OpenAlert:
    alert = Alert(
        timestamp=window.window_end,
        process_name=window.process_name,
        window_start=window.window_start,
        combined_score=1.0,
        per_detector_scores={},
        reasons=reasons,
        **overrides,
    )
    return OpenAlert(alert, window)


@pytest.fixture
def parent(make_window, noon) -> OpenAlert:
    window = make_window("cloudsync", noon, new_remote_ips=(C2, OTHER), new_remote_ports=(8443,))
    return open_alert(window, [f"first connection to {C2}", "unusual combination"])


def test_same_destinations_and_no_new_reason_is_a_repeat(make_window, noon, parent) -> None:
    later = make_window("cloudsync", noon + 3600, new_remote_ips=(C2,))
    assert is_repeat(later, [], parent)
    assert is_repeat(later, ["unusual combination"], parent)
    # a window with nothing new at all is a repeat too, if its reasons are known
    assert is_repeat(make_window("cloudsync", noon + 3600), ["unusual combination"], parent)


@pytest.mark.parametrize(
    ("change", "reasons"),
    [
        ({"new_remote_ips": (C2, "198.51.100.1")}, []),  # another new IP
        ({"new_remote_ports": (8443, 22)}, []),  # another new port
        ({"new_remote_ips": (C2,)}, ["connection count 9x above typical"]),  # new reason
        ({"process_name": "browser", "new_remote_ips": (C2,)}, []),  # other process
    ],
)
def test_anything_new_is_not_a_repeat(make_window, noon, parent, change, reasons) -> None:
    later = make_window(window_start=noon + 3600, **{"process_name": "cloudsync", **change})
    assert not is_repeat(later, reasons, parent)


SPIKE = "connection count 47x above this process's typical window (47 vs typical 1)"


def spike(count: int) -> str:
    return SPIKE.replace("47", str(count))


def test_a_reason_of_similar_size_is_not_news(make_window, noon) -> None:
    """reasons quote counts: a lasting spike must not raise an alert per window"""
    first = open_alert(make_window("app", noon), [spike(47), "unusual combination"])
    later = make_window("app", noon + 300)
    assert is_repeat(later, [spike(52)], first)
    assert is_repeat(later, [spike(9)], first)  # smaller is never news
    assert is_repeat(later, [spike(94), "unusual combination"], first)  # exactly double


def test_a_reason_that_grows_several_times_over_is_news(make_window, noon) -> None:
    """an open alert must not swallow a much larger version of its own cause"""
    first = open_alert(make_window("app", noon), [spike(47)])
    later = make_window("app", noon + 300)
    assert not is_repeat(later, [spike(95)], first)
    assert not is_repeat(later, [spike(4700)], first)
    # once the larger alert exists, windows of that size repeat it, not the first
    bigger = open_alert(make_window("app", noon + 300), [spike(4700)])
    assert find_repeated(make_window("app", noon + 600), [spike(5000)], [first, bigger]) is (
        bigger.alert
    )


def test_only_amounts_are_compared_loosely() -> None:
    scan = "contacted 150 distinct remote ports (typical: 1) - possible port scan"
    assert covers(scan, scan.replace("150", "90"))
    assert reason_kind(scan) != reason_kind("contacted 12 distinct remote ports (typical: 1)")
    # digits that name something are not amounts: another hour or path is another reason
    hour = "active at an unusual hour for this process (03:00-04:00)"
    assert not covers(hour, hour.replace("03:00-04:00", "04:00-05:00"))
    assert not covers("executable is running from /tmp/a1", "executable is running from /tmp/a2")
    assert not covers(SPIKE, "connections are highly periodic (possible beaconing)")


def test_only_open_alerts_absorb_repeats(make_window, noon, parent) -> None:
    later = make_window("cloudsync", noon + 3600, new_remote_ips=(C2,))
    for status in (AlertStatus.EXPECTED, AlertStatus.SUSPICIOUS):
        parent.alert.status = status
        assert not is_repeat(later, [], parent)


def test_repeats_gather_on_the_oldest_covering_alert(make_window, noon) -> None:
    first = open_alert(make_window("app", noon, new_remote_ips=(C2,)), ["a"])
    wider = open_alert(make_window("app", noon + 300, new_remote_ips=(C2, OTHER)), ["a", "b"])
    later = make_window("app", noon + 600, new_remote_ips=(C2,))
    assert find_repeated(later, ["a"], [first, wider]) is first.alert
    assert find_repeated(later, ["b"], [first, wider]) is wider.alert
    assert find_repeated(later, ["c"], [first, wider]) is None
    assert find_repeated(later, [], []) is None


# repeats inside the live Monitor


class NewDestinations(Detector):
    """any new IP is as anomalous as it gets. Reasons only name the new IPs"""

    name = "new"
    explains_destinations = True

    def fit(self, windows: Sequence[FeatureWindow]) -> NewDestinations:
        self._fitted = True
        return self

    def score(self, windows: Sequence[FeatureWindow]) -> np.ndarray:
        return np.array([10.0 if w.new_remote_ips else 0.0 for w in windows])

    def explain(self, window: FeatureWindow) -> list[str]:
        return [f"first connection to {ip}" for ip in window.new_remote_ips]


class Busy(Detector):
    """more than ``BUSY`` connections in a window is as anomalous as it gets"""

    name = "busy"

    def fit(self, windows: Sequence[FeatureWindow]) -> Busy:
        self._fitted = True
        return self

    def score(self, windows: Sequence[FeatureWindow]) -> np.ndarray:
        return np.array([10.0 if w.features.connection_count > BUSY else 0.0 for w in windows])

    def explain(self, window: FeatureWindow) -> list[str]:
        return [f"burst of {window.features.connection_count} connections"]


def make_monitor(db: Database | None, sink: CollectingSink, tracker: NoveltyTracker) -> Monitor:
    combiner = ScoreCombiner(min_process_history=1).fit(
        ["app"] * 10, {"new": np.zeros(10), "busy": np.zeros(10)}
    )
    model = TrainedModel(
        [NewDestinations().fit([]), Busy().fit([])],
        combiner,
        trained_at=0,
        training_windows=10,
        processes=["app"],
    )
    return Monitor(model, tracker, WindowBuilder(), 0.99, sinks=[sink], db=db)


@pytest.fixture
def slot(make_event, noon) -> Callable[..., list[ConnectionEvent]]:
    """events for slot ``i``: one connection from ``app`` to each IP, ``times`` times"""
    start = window_start_for(noon, 300)

    def factory(i: int, *ips: str, times: int = 1) -> list[ConnectionEvent]:
        return [
            make_event(start + i * 300 + 1 + n, "app", remote_ip=ip)
            for ip in ips
            for n in range(times)
        ]

    return factory


def test_monitor_attaches_repeats_and_raises_anything_new(slot) -> None:
    sink, tracker = CollectingSink(), NoveltyTracker()
    with Database.open(":memory:") as db:
        monitor = make_monitor(db, sink, tracker)
        (first,) = monitor.process_events(slot(0, C2))
        (again,) = monitor.process_events(slot(1, C2))
        (wider,) = monitor.process_events(slot(2, C2, OTHER))
        (burst,) = monitor.process_events(slot(3, C2, times=BUSY + 1))
        (third,) = monitor.process_events(slot(4, C2))

        # the revisit only repeats alert #1: attached, nobody notified
        assert first.alert is not None and not first.repeat
        assert again.repeat and third.repeat
        assert again.alert.id == third.alert.id == first.alert.id
        # a new destination or a new reason is a new alert
        assert not wider.repeat and wider.alert.id != first.alert.id
        assert not burst.repeat and burst.alert.reasons == [
            f"first connection to {C2}",
            f"burst of {BUSY + 1} connections",
        ]
        assert sink.alerts == [first.alert, wider.alert, burst.alert]

        stored = db.get_alert(first.alert.id)
        assert stored.repeat_window_starts == [again.window.window_start, third.window.window_start]
        assert stored.occurrences == 3
        assert stored.last_window_start == third.window.window_start
        assert len(db.query_alerts()) == 3

        # repeats count as alerted: C2 never became familiar
        assert not tracker.is_known("app", PairKind.IP, C2, third.window.window_end)


def test_attached_windows_stay_out_of_training_until_the_parent_is_expected(slot, noon) -> None:
    sink, tracker = CollectingSink(), NoveltyTracker()
    tracker.observe("app", PairKind.IP, [KNOWN], at=noon)
    with Database.open(":memory:") as db:
        monitor = make_monitor(db, sink, tracker)
        results = monitor.process_events(
            [*slot(0, C2), *slot(1, C2), *slot(2, C2), *slot(3, KNOWN)]
        )
        parent, quiet = results[0].alert, results[3]
        assert [r.repeat for r in results] == [False, True, True, False]
        assert quiet.alert is None

        def training() -> list[FeatureWindow]:
            windows = db.query_windows()
            return build_training_set(windows, db.query_alerts(), db.list_labels()).windows

        # the alert and both of its repeats are out. Only the quiet window is in
        assert training() == [quiet.window]

        # labelling the parent expected re-admits every window it covers...
        record_label(db, parent.id, Verdict.EXPECTED, tracker=tracker, now=noon)
        assert training() == [r.window for r in results]
        # ...and commits each one's destinations: three sightings of C2
        last = results[2].window.window_end
        assert tracker.familiarity("app", PairKind.IP, C2, last) == pytest.approx(3, rel=0.01)

        # now familiar, the destination no longer causes an alert at all
        (after,) = monitor.process_events(slot(5, C2))
        assert after.alert is None


def test_a_recurrence_after_a_suspicious_label_alerts_again(slot) -> None:
    sink, tracker = CollectingSink(), NoveltyTracker()
    with Database.open(":memory:") as db:
        monitor = make_monitor(db, sink, tracker)
        (first,) = monitor.process_events(slot(0, C2))
        record_label(db, first.alert.id, Verdict.SUSPICIOUS, tracker=tracker, now=0)
        (again,) = monitor.process_events(slot(1, C2))
        assert again.alert is not None and not again.repeat
        assert sink.alerts == [first.alert, again.alert]


def test_open_alerts_from_an_earlier_run_absorb_repeats(slot) -> None:
    tracker = NoveltyTracker()
    with Database.open(":memory:") as db:
        (first,) = make_monitor(db, CollectingSink(), tracker).process_events(slot(0, C2))
        sink = CollectingSink()
        (again,) = make_monitor(db, sink, tracker).process_events(slot(1, C2))
        assert again.repeat and again.alert.id == first.alert.id
        assert sink.alerts == []


def test_monitor_without_a_database_deduplicates_in_memory(slot) -> None:
    sink = CollectingSink()
    monitor = make_monitor(None, sink, NoveltyTracker())
    first, again, burst = monitor.process_events(
        [*slot(0, C2), *slot(1, C2), *slot(2, C2, times=BUSY + 1)]
    )
    assert again.repeat and again.alert is first.alert
    assert first.alert.repeat_window_starts == [again.window.window_start]
    assert not burst.repeat
    assert sink.alerts == [first.alert, burst.alert]
