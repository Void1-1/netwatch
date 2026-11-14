"""Pipeline pieces around the Monitor: catching up on stored events, atomic slots, model loading"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from netwatch.alerts.base import CollectingSink
from netwatch.detectors.base import Detector
from netwatch.features.novelty import NoveltyTracker, PairKind
from netwatch.features.windowing import WindowBuilder, window_start_for
from netwatch.models import FeatureWindow
from netwatch.pipeline import (
    SETTLE_SECONDS,
    WINDOWED_UNTIL_KEY,
    Monitor,
    PipelineError,
    TrainedModel,
    build_pending_windows,
    load_model,
    save_model,
    score_closed_slots,
    score_pending_events,
)
from netwatch.scoring.combiner import ScoreCombiner
from netwatch.storage.db import Database

BAD = "203.0.113.7"
GOOD = "192.0.2.1"


class FlagAddress(Detector):
    """test detector: any window that contacts ``BAD`` is as anomalous as it gets"""

    name = "flag"

    def fit(self, windows: Sequence[FeatureWindow]) -> FlagAddress:
        self._fitted = True
        return self

    def score(self, windows: Sequence[FeatureWindow]) -> np.ndarray:
        return np.array([10.0 if BAD in w.remote_ips else 0.0 for w in windows])

    def explain(self, window: FeatureWindow) -> list[str]:
        return [f"contacted {BAD}"]


def make_model() -> TrainedModel:
    combiner = ScoreCombiner(min_process_history=1).fit(["app"] * 10, {"flag": np.zeros(10)})
    return TrainedModel(
        [FlagAddress().fit([])], combiner, trained_at=0, training_windows=10, processes=["app"]
    )


def make_monitor(db: Database, tracker: NoveltyTracker, sink: CollectingSink) -> Monitor:
    return Monitor(make_model(), tracker, WindowBuilder(), 0.99, sinks=[sink], db=db)


# stored events that no window was built from yet


def test_stored_events_are_scored_before_live_ones_move_progress_past_them(
    make_event, noon
) -> None:
    """events collected between the last ``build-windows`` and ``watch`` must not be skipped"""
    start = window_start_for(noon, 300)
    sink, tracker = CollectingSink(), NoveltyTracker()
    with Database.open(":memory:") as db:
        # training data, windowed in batch before a model existed
        db.insert_events([make_event(start + 1, "app", remote_ip=GOOD)])
        assert len(build_pending_windows(db, tracker, WindowBuilder(), until=start + 300)) == 1

        # collected afterwards, while nothing was scoring: one quiet slot and one bad one
        db.insert_events(
            [
                make_event(start + 301, "app", remote_ip=GOOD),
                make_event(start + 601, "app", remote_ip=BAD),
            ]
        )
        monitor = make_monitor(db, tracker, sink)
        caught_up = score_pending_events(db, monitor, until=start + 900)

        assert [s.window.window_start for s in caught_up] == [start + 300, start + 600]
        assert [s.alert is not None for s in caught_up] == [False, True]
        assert len(sink.alerts) == 1  # the stored bad slot alerted like a live one
        assert not tracker.is_known("app", PairKind.IP, BAD, start + 900)
        assert float(db.get_meta(WINDOWED_UNTIL_KEY)) == start + 900
        assert db.count_windows() == 3

        # nothing is processed twice, and the open slot is left alone
        assert score_pending_events(db, monitor, until=start + 1199) == []


def test_watch_scores_what_another_collector_stored(make_event, noon, monkeypatch) -> None:
    """a ``collect`` left running next to ``watch`` may see processes ``watch`` cannot"""
    start = window_start_for(noon, 300)
    monkeypatch.setattr("netwatch.pipeline.time.time", lambda: start + 900)  # for pruning
    sink, tracker = CollectingSink(), NoveltyTracker()
    with Database.open(":memory:") as db:
        monitor = make_monitor(db, tracker, sink)
        db.insert_events([make_event(start + 1, "app", remote_ip=GOOD)])  # watch's own
        db.insert_events([make_event(start + 2, "daemon", remote_ip=BAD)])  # the other one's

        scored = score_closed_slots(db, monitor, tracker, until=start + 300)
        assert {s.window.process_name for s in scored} == {"app", "daemon"}
        assert [a.process_name for a in sink.alerts] == ["daemon"]
        # what the quiet window taught is stored, the alerting one's destination is not
        stored = {(r.process_name, r.value) for r in db.load_novelty() if r.kind == PairKind.IP}
        assert stored == {("app", GOOD)}

        # slots without events still count as done, so they are not read on every step
        assert score_closed_slots(db, monitor, tracker, until=start + 900) == []
        assert float(db.get_meta(WINDOWED_UNTIL_KEY)) == start + 900


def test_a_slot_that_only_just_ended_is_left_to_settle(make_event, noon, monkeypatch) -> None:
    """another command's last events of a slot are stored a moment after their timestamp"""
    start = window_start_for(noon, 300)
    with Database.open(":memory:") as db:
        tracker = NoveltyTracker()
        monitor = make_monitor(db, tracker, CollectingSink())
        db.insert_events([make_event(start + 299, "app", remote_ip=GOOD)])

        monkeypatch.setattr("netwatch.pipeline.time.time", lambda: start + 300 + 1)
        assert score_closed_slots(db, monitor, tracker) == []
        assert db.count_windows() == 0

        monkeypatch.setattr("netwatch.pipeline.time.time", lambda: start + 300 + SETTLE_SECONDS)
        assert len(score_closed_slots(db, monitor, tracker)) == 1


def test_a_slot_another_command_scored_is_not_scored_again(make_event, noon) -> None:
    """``build-windows`` can reach a slot just before a running ``watch`` closes it"""
    start = window_start_for(noon, 300)
    events = [make_event(start + 1, "app", remote_ip=BAD)]
    watch_sink, batch_sink = CollectingSink(), CollectingSink()
    with Database.open(":memory:") as db:
        watch = make_monitor(db, NoveltyTracker(), watch_sink)
        db.insert_events(events)  # watch stores what it collects, then waits for the slot

        batch = make_monitor(db, NoveltyTracker(), batch_sink)
        assert len(score_pending_events(db, batch, until=start + 300)) == 1
        assert watch.process_events(events) == []

        (alert,) = db.query_alerts()
        assert alert.occurrences == 1  # not attached to itself as a repeat
        assert (len(batch_sink.alerts), len(watch_sink.alerts)) == (1, 0)
        assert db.count_windows() == 1

        # the next slot is still the running watch's to score
        (later,) = watch.process_events([make_event(start + 301, "app", remote_ip=GOOD)])
        assert later.alert is None


def test_a_slot_is_stored_whole_or_not_at_all(make_event, noon, monkeypatch) -> None:
    """a crash while saving must not leave an alerting window stored without its alert

    Such a window would count as clean and be trained on
    """
    start = window_start_for(noon, 300)
    with Database.open(":memory:") as db:
        monitor = make_monitor(db, NoveltyTracker(), CollectingSink())

        def crash(alert: object) -> int:
            raise RuntimeError("power cut")

        monkeypatch.setattr(db, "insert_alert", crash)
        with pytest.raises(RuntimeError, match="power cut"):
            monitor.process_events([make_event(start + 1, "app", remote_ip=BAD)])

        assert db.count_windows() == 0
        assert db.query_alerts() == []
        assert db.get_meta(WINDOWED_UNTIL_KEY) is None


# loading a model as root


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    path = tmp_path / "models" / "netwatch-model.pkl"
    save_model(make_model(), path)
    return path


def test_models_load_normally_without_root(model_path: Path) -> None:
    assert load_model(model_path).processes == ["app"]
    with pytest.raises(PipelineError, match="netwatch train"):
        load_model(model_path.with_name("missing.pkl"))


@pytest.mark.skipif(os.name != "posix", reason="ownership checks are POSIX-only")
def test_root_only_loads_a_model_it_or_the_sudo_user_owns(model_path: Path, monkeypatch) -> None:
    """unpickling runs code from the file, and under sudo that code runs as root"""
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    owner = model_path.stat().st_uid

    # `sudo netwatch watch` by the user who trained the model
    monkeypatch.setenv("SUDO_UID", str(owner))
    assert load_model(model_path).processes == ["app"]

    # the same file, but anyone on the machine could have rewritten it
    model_path.chmod(0o666)
    with pytest.raises(PipelineError, match="refusing to load"):
        load_model(model_path)
    model_path.chmod(0o644)

    # a file that belongs to some third user
    monkeypatch.setenv("SUDO_UID", str(owner + 1))
    if owner != 0:
        with pytest.raises(PipelineError, match="refusing to load"):
            load_model(model_path)
