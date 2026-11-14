"""Labels and the poisoning rule, on their own and inside the Monitor"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from netwatch.alerts.base import CollectingSink
from netwatch.config import Config
from netwatch.detectors.base import Detector
from netwatch.features.novelty import NoveltyTracker, PairKind
from netwatch.features.windowing import WindowBuilder, window_start_for
from netwatch.feedback.labels import build_training_set, latest_verdicts, record_label
from netwatch.models import Alert, AlertStatus, FeatureWindow, Label, Verdict
from netwatch.pipeline import WINDOWED_UNTIL_KEY, Monitor, TrainedModel, load_tracker, sync_tracker
from netwatch.scoring.combiner import ScoreCombiner
from netwatch.storage.db import Database


def alert_for(window: FeatureWindow, alert_id: int | None) -> Alert:
    return Alert(
        timestamp=window.window_end,
        process_name=window.process_name,
        window_start=window.window_start,
        combined_score=1.0,
        per_detector_scores={"novelty": 1.0},
        reasons=["test"],
        id=alert_id,
    )


# the poisoning rule on its own


def test_alerted_windows_stay_out_unless_labelled_expected(make_window, noon) -> None:
    clean, unreviewed, expected, suspicious = (make_window("app", noon + i * 300) for i in range(4))
    alerts = [alert_for(unreviewed, 1), alert_for(expected, 2), alert_for(suspicious, 3)]
    labels = [Label(2, Verdict.EXPECTED, noon), Label(3, Verdict.SUSPICIOUS, noon)]

    result = build_training_set([clean, unreviewed, expected, suspicious], alerts, labels)

    assert result.windows == [clean, expected]
    assert result.excluded_alerted == 2
    assert result.readmitted_expected == 1


def test_repeats_share_the_fate_of_their_alert(make_window, noon) -> None:
    first, repeat, clean = (make_window("app", noon + i * 300) for i in range(3))
    alert = alert_for(first, 1)
    alert.repeat_window_starts.append(repeat.window_start)
    windows = [first, repeat, clean]

    unreviewed = build_training_set(windows, [alert], [])
    assert unreviewed.windows == [clean]
    assert unreviewed.excluded_alerted == 2

    expected = build_training_set(windows, [alert], [Label(1, Verdict.EXPECTED, noon)])
    assert expected.windows == windows
    assert expected.readmitted_expected == 2


def test_latest_label_wins(make_window, noon) -> None:
    window = make_window("app", noon)
    alerts = [alert_for(window, 1)]
    flip_to_suspicious = [
        Label(1, Verdict.EXPECTED, noon),
        Label(1, Verdict.SUSPICIOUS, noon + 10),
    ]
    flip_to_expected = [
        Label(1, Verdict.SUSPICIOUS, noon),
        Label(1, Verdict.EXPECTED, noon + 10),
    ]
    assert latest_verdicts(flip_to_suspicious) == {1: Verdict.SUSPICIOUS}
    assert build_training_set([window], alerts, flip_to_suspicious).windows == []
    assert build_training_set([window], alerts, flip_to_expected).windows == [window]


def test_unsaved_alerts_are_treated_as_unreviewed(make_window, noon) -> None:
    window = make_window("app", noon)
    assert build_training_set([window], [alert_for(window, None)], []).windows == []


# recording labels


def test_record_label_updates_status_and_commits_expected_destinations(make_window, noon) -> None:
    window = make_window("app", noon, remote_ips=("203.0.113.9",), remote_ports=(8443,))
    tracker = NoveltyTracker()
    with Database.open(":memory:") as db:
        db.upsert_windows([window])
        alert_id = db.insert_alert(alert_for(window, None))

        record_label(db, alert_id, Verdict.SUSPICIOUS, tracker=tracker, now=noon)
        assert db.get_alert(alert_id).status == AlertStatus.SUSPICIOUS
        assert len(tracker) == 0  # suspicious: nothing becomes familiar

        record_label(db, alert_id, Verdict.EXPECTED, note="mine", tracker=tracker, now=noon + 1)
        assert db.get_alert(alert_id).status == AlertStatus.EXPECTED
        assert tracker.is_known("app", PairKind.IP, "203.0.113.9", window.window_end)
        assert [label.verdict for label in db.list_labels()] == [
            Verdict.SUSPICIOUS,
            Verdict.EXPECTED,
        ]

        with pytest.raises(KeyError):
            record_label(db, 999, Verdict.EXPECTED)


def test_labelling_an_alert_expected_commits_its_repeats_too(make_window, noon) -> None:
    first = make_window("app", noon, remote_ips=("203.0.113.9",))
    repeat = make_window("app", noon + 300, remote_ips=("203.0.113.9",), remote_ports=(8443,))
    tracker = NoveltyTracker()
    with Database.open(":memory:") as db:
        db.upsert_windows([first, repeat])
        alert_id = db.insert_alert(alert_for(first, None))
        db.attach_repeat(alert_id, repeat.window_start)

        record_label(db, alert_id, Verdict.EXPECTED, tracker=tracker, now=noon)
        at = repeat.window_end
        # port 8443 only appeared in the repeat. The IP was seen in both windows
        assert tracker.is_known("app", PairKind.PORT, 8443, at)
        assert tracker.familiarity("app", PairKind.IP, "203.0.113.9", at) == pytest.approx(
            2, rel=0.01
        )


def test_a_change_of_mind_takes_the_destinations_back(make_window, noon) -> None:
    """expected -> suspicious must leave novelty memory as if the alert was never excused"""
    first = make_window(
        "app",
        noon,
        remote_ips=("192.0.2.1", "203.0.113.9"),
        remote_ports=(443, 8443),
        new_remote_ips=("203.0.113.9",),
        new_remote_ports=(8443,),
    )
    repeat = make_window(
        "app", noon + 300, remote_ips=("203.0.113.9",), new_remote_ips=("203.0.113.9",)
    )
    tracker = NoveltyTracker()
    tracker.observe("app", PairKind.IP, ["192.0.2.1"], at=noon - 300)
    tracker.observe("app", PairKind.PORT, [443], at=noon - 300)
    at = repeat.window_end
    with Database.open(":memory:") as db:
        db.upsert_windows([first, repeat])
        alert_id = db.insert_alert(alert_for(first, None))
        db.attach_repeat(alert_id, repeat.window_start)

        record_label(db, alert_id, Verdict.EXPECTED, tracker=tracker, now=at)
        assert tracker.is_known("app", PairKind.IP, "203.0.113.9", at)
        assert tracker.is_known("app", PairKind.PORT, 8443, at)

        record_label(db, alert_id, Verdict.SUSPICIOUS, tracker=tracker, now=at + 60)
        assert not tracker.is_known("app", PairKind.IP, "203.0.113.9", at + 60)
        assert not tracker.is_known("app", PairKind.PORT, 8443, at + 60)
        # what the process already knew before the alert is untouched
        assert tracker.is_known("app", PairKind.IP, "192.0.2.1", at + 60)
        assert tracker.is_known("app", PairKind.PORT, 443, at + 60)

        # and the mind can change back
        record_label(db, alert_id, Verdict.EXPECTED, tracker=tracker, now=at + 120)
        assert tracker.is_known("app", PairKind.IP, "203.0.113.9", at + 120)
        sync_tracker(db, tracker, now=at + 120)  # expected again: nothing left to forget
        assert tracker.is_known("app", PairKind.IP, "203.0.113.9", at + 120)


def test_a_sighting_after_the_suspicious_label_is_kept(make_window, noon) -> None:
    alerted = make_window("app", noon, remote_ips=("203.0.113.9",), new_remote_ips=("203.0.113.9",))
    tracker = NoveltyTracker()
    with Database.open(":memory:") as db:
        db.upsert_windows([alerted])
        alert_id = db.insert_alert(alert_for(alerted, None))
        record_label(db, alert_id, Verdict.SUSPICIOUS, tracker=tracker, now=noon + 400)

        # a later window that did not alert commits the address on its own merits
        tracker.observe("app", PairKind.IP, ["203.0.113.9"], at=noon + 900)
        sync_tracker(db, tracker, now=noon + 900)
        assert tracker.is_known("app", PairKind.IP, "203.0.113.9", noon + 900)


# labelling while `netwatch watch` runs
#
# Each command opens its own connection and loads its own tracker, as two terminals would.
# The monitor keeps its tracker in memory for hours


def test_expected_label_survives_the_running_monitor_saving_afterwards(
    make_window, noon, tmp_path
) -> None:
    path = tmp_path / "netwatch.db"
    first = make_window("app", noon, remote_ips=("203.0.113.9",), remote_ports=(8443,))
    repeat = make_window("app", noon + 300, remote_ips=("203.0.113.9",))
    clean = make_window("app", noon + 600, remote_ips=("192.0.2.1",))
    now = clean.window_end
    with Database.open(path) as watch_db, Database.open(path) as label_db:
        watch_tracker = load_tracker(watch_db, Config())
        watch_db.upsert_windows([first, repeat])
        alert_id = watch_db.insert_alert(alert_for(first, None))
        watch_db.attach_repeat(alert_id, repeat.window_start)

        # `netwatch label ID expected` in another terminal
        label_tracker = load_tracker(label_db, Config())
        record_label(label_db, alert_id, Verdict.EXPECTED, tracker=label_tracker, now=now)
        sync_tracker(label_db, label_tracker, now=now)

        # the monitor closes its next slot and saves, without knowing about the label
        watch_tracker.observe_window(clean)
        sync_tracker(watch_db, watch_tracker, now=now)

        # neither side's commit is lost, and the monitor now knows the labelled destinations
        # without a restart
        for tracker in (load_tracker(label_db, Config()), watch_tracker):
            assert tracker.is_known("app", PairKind.IP, "203.0.113.9", now)
            assert tracker.is_known("app", PairKind.PORT, 8443, now)
            assert tracker.is_known("app", PairKind.IP, "192.0.2.1", now)


def test_a_label_keeps_what_the_monitor_saved_while_it_ran(make_window, noon, tmp_path) -> None:
    path = tmp_path / "netwatch.db"
    alerted = make_window("app", noon, remote_ips=("203.0.113.9",))
    clean = make_window("app", noon + 300, remote_ips=("192.0.2.1",))
    now = clean.window_end
    with Database.open(path) as watch_db, Database.open(path) as label_db:
        watch_tracker = load_tracker(watch_db, Config())
        watch_db.upsert_windows([alerted])
        alert_id = watch_db.insert_alert(alert_for(alerted, None))

        label_tracker = load_tracker(label_db, Config())
        watch_tracker.observe_window(clean)
        sync_tracker(watch_db, watch_tracker, now=now)  # between the label's load and save
        record_label(label_db, alert_id, Verdict.EXPECTED, tracker=label_tracker, now=now)
        sync_tracker(label_db, label_tracker, now=now)

        stored = load_tracker(watch_db, Config())
        assert stored.is_known("app", PairKind.IP, "192.0.2.1", now)
        assert stored.is_known("app", PairKind.IP, "203.0.113.9", now)


# the poisoning rule inside the live Monitor


class FlagProcess(Detector):
    """test detector: one named process is always as anomalous as it gets"""

    name = "flag"

    def __init__(self, bad: str) -> None:
        super().__init__()
        self.bad = bad

    def fit(self, windows: Sequence[FeatureWindow]) -> FlagProcess:
        self._fitted = True
        return self

    def score(self, windows: Sequence[FeatureWindow]) -> np.ndarray:
        return np.array([10.0 if w.process_name == self.bad else 0.0 for w in windows])

    def explain(self, window: FeatureWindow) -> list[str]:
        return [f"{window.process_name} is flagged"]


def test_monitor_alerts_and_keeps_alerted_destinations_unfamiliar(make_event, noon) -> None:
    detector = FlagProcess("evil").fit([])
    combiner = ScoreCombiner(min_process_history=1).fit(["good"] * 10, {"flag": np.zeros(10)})
    model = TrainedModel([detector], combiner, trained_at=0, training_windows=10, processes=[])
    tracker = NoveltyTracker()
    sink = CollectingSink()
    start = window_start_for(noon, 300)
    events = [
        make_event(start + 1, "good", remote_ip="192.0.2.1"),
        make_event(start + 2, "evil", remote_ip="203.0.113.7"),
    ]

    with Database.open(":memory:") as db:
        monitor = Monitor(model, tracker, WindowBuilder(), 0.99, sinks=[sink], db=db)
        results = monitor.process_events(events)

        flagged = {r.window.process_name: r.alert for r in results}
        assert flagged["good"] is None
        assert flagged["evil"] is not None
        assert flagged["evil"].reasons == ["evil is flagged"]
        assert sink.alerts == [flagged["evil"]]
        assert flagged["evil"].id is not None  # stored before delivery

        # only the alert-free window's destination became familiar
        assert tracker.is_known("good", PairKind.IP, "192.0.2.1", start + 300)
        assert not tracker.is_known("evil", PairKind.IP, "203.0.113.7", start + 300)

        assert db.count_windows() == 2
        assert float(db.get_meta(WINDOWED_UNTIL_KEY)) == start + 300

        # and the alerted window is kept out of the next training run
        training = build_training_set(db.query_windows(), db.query_alerts(), db.list_labels())
        assert [w.process_name for w in training.windows] == ["good"]


def test_a_running_monitor_cannot_bring_back_what_a_suspicious_label_took(
    make_window, noon, tmp_path
) -> None:
    """the monitor's in-memory copy still holds the pair. Merging it back must not restore it"""
    path = tmp_path / "netwatch.db"
    alerted = make_window(
        "app",
        noon,
        remote_ips=("203.0.113.9",),
        remote_ports=(8443,),
        new_remote_ips=("203.0.113.9",),
        new_remote_ports=(8443,),
    )
    clean = make_window("app", noon + 300, remote_ips=("192.0.2.1",))
    now = clean.window_end
    with Database.open(path) as watch_db, Database.open(path) as label_db:
        watch_tracker = load_tracker(watch_db, Config())
        watch_db.upsert_windows([alerted])
        alert_id = watch_db.insert_alert(alert_for(alerted, None))

        # labelled expected by mistake, and the running monitor picks that up
        label_tracker = load_tracker(label_db, Config())
        record_label(label_db, alert_id, Verdict.EXPECTED, tracker=label_tracker, now=now)
        sync_tracker(label_db, label_tracker, now=now)
        sync_tracker(watch_db, watch_tracker, now=now)
        assert watch_tracker.is_known("app", PairKind.IP, "203.0.113.9", now)

        # `netwatch label ID suspicious` in the other terminal
        label_tracker = load_tracker(label_db, Config())
        record_label(label_db, alert_id, Verdict.SUSPICIOUS, tracker=label_tracker, now=now + 60)
        sync_tracker(label_db, label_tracker, now=now + 60)

        # the monitor closes its next slot and syncs, still holding the pair in memory
        watch_tracker.observe_window(clean)
        sync_tracker(watch_db, watch_tracker, now=now + 300)

        for tracker in (load_tracker(label_db, Config()), watch_tracker):
            assert not tracker.is_known("app", PairKind.IP, "203.0.113.9", now + 300)
            assert not tracker.is_known("app", PairKind.PORT, 8443, now + 300)
            assert tracker.is_known("app", PairKind.IP, "192.0.2.1", now + 300)
