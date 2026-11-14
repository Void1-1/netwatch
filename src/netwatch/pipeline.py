"""run the stages in order, shared by CLI and demo

* :func:`store_new_connections` - collector -> storage, once per connection even with two
  collectors on one database.
* :func:`build_pending_windows` - storage -> features (batch, before a model).
* :func:`select_training_windows` - warm-up skip + the poisoning rule.
* :func:`train_model` - fit every detector, then the combiner on their scores.
* :class:`Monitor` - the live loop body: window -> score -> alert (or attach a repeat to an
  open alert) -> learn.
* :func:`score_pending_events` - the same loop body over stored events no window was built
  from yet (once a model exists).
* :func:`score_closed_slots` - one step of ``netwatch watch``: sync novelty memory, score what
  is pending, sync again.
"""

from __future__ import annotations

import logging
import os
import pickle
import stat
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from netwatch import __version__
from netwatch.alerts.base import AlertSink
from netwatch.alerts.dedup import OpenAlert, find_repeated
from netwatch.collector.psutil_collector import PsutilCollector
from netwatch.config import Config
from netwatch.detectors import Detector, build_detectors, detector_weights
from netwatch.features.novelty import NoveltyTracker
from netwatch.features.windowing import WindowBuilder, window_start_for
from netwatch.feedback.labels import build_training_set, forget_suspicious_destinations
from netwatch.models import Alert, AlertStatus, ConnectionEvent, FeatureWindow
from netwatch.scoring.combiner import CombinedScore, ScoreCombiner, merge_reasons
from netwatch.storage.db import Database

log = logging.getLogger(__name__)

HOUR = 3600.0
DAY = 86_400.0
# Meta key recording the end of the last windowed time slot
WINDOWED_UNTIL_KEY = "windowed_until"
# Refuse to train on less than this: the result would be meaningless
MIN_TRAINING_WINDOWS = 20
# A slot counts as closed this long after its end. A collector stamps its events when it takes
# the snapshot and stores them a moment later, so the last events of a slot from another
# running command can arrive just after the slot has ended
SETTLE_SECONDS = 5.0


class PipelineError(RuntimeError):
    """A user-facing issue: missing data, missing model, etc."""


# Construction helpers


def make_window_builder(config: Config) -> WindowBuilder:
    return WindowBuilder(
        window_seconds=config.features.window_seconds,
        suspicious_paths=tuple(config.features.suspicious_paths),
        min_periodic_events=config.features.min_periodic_events,
    )


def new_tracker(config: Config) -> NoveltyTracker:
    return NoveltyTracker(
        half_life_days=config.novelty.half_life_days,
        min_familiarity=config.novelty.min_familiarity,
    )


def load_tracker(db: Database, config: Config) -> NoveltyTracker:
    """novelty memory as last saved in ``db``"""
    return NoveltyTracker.from_records(
        db.load_novelty(),
        half_life_days=config.novelty.half_life_days,
        min_familiarity=config.novelty.min_familiarity,
    )


def sync_tracker(db: Database, tracker: NoveltyTracker, now: float | None = None) -> None:
    """Merge ``tracker`` with stored novelty memory both ways, then prune

    Between the merge and the save, destinations flagged by alerts labelled suspicious are
    forgotten again (:func:`netwatch.feedback.labels.forget_suspicious_destinations`), so a
    copy of the memory that predates the label cannot bring them back
    """
    with db.write_lock():
        tracker.merge_records(db.load_novelty())
        forget_suspicious_destinations(db, tracker)
        tracker.prune(time.time() if now is None else now)
        db.save_novelty(tracker.to_records())


# Collecting


def store_new_connections(db: Database, collector: PsutilCollector, interval: float) -> int:
    """take a snapshot and store the connections that are new. Returns how many were stored

    ``netwatch collect`` and ``netwatch watch`` can run on the same database, and both see
    most connections. A connection that the other one already stored is left out, so it is
    counted once.

    How a duplicate is recognised: the connections in this snapshot opened after the previous
    one, so the other collector cannot have seen them before then either. An event for the
    same socket stored since the previous snapshot is therefore the same connection. This
    collector's own earlier report of a socket it now reports again is always older than
    that: the socket was missing from the previous snapshot. The check starts a little before
    the previous snapshot (at most a second, and less than half an interval) because the
    other collector may have started its snapshot earlier and still caught the connection
    """
    events = collector.collect()
    previous = collector.previous_snapshot_at
    if previous is None:
        # a first snapshot without priming (`collect --once`) reports everything that is open
        return db.insert_events(events)
    return db.insert_events(events, seen_since=previous - min(1.0, interval / 2))


# Batch windowing


def _pending_range(
    db: Database, builder: WindowBuilder, until: float | None
) -> tuple[float | None, float] | None:
    """``(start, end)`` of the closed slots not windowed yet, or ``None`` if there are none

    ``start`` is ``None`` when nothing has been windowed so far. ``end`` is the start of the
    slot that contains ``until``: that slot is still filling. By default ``until`` is
    ``SETTLE_SECONDS`` ago, so a slot that has only just ended is left for the next call
    """
    done = db.get_meta(WINDOWED_UNTIL_KEY)
    start = float(done) if done is not None else None
    until = time.time() - SETTLE_SECONDS if until is None else until
    end = window_start_for(until, builder.window_seconds)
    return None if start is not None and start >= end else (start, end)


def build_pending_windows(
    db: Database,
    tracker: NoveltyTracker,
    builder: WindowBuilder,
    until: float | None = None,
) -> list[FeatureWindow]:
    """Window events not yet windowed in closed slots before ``until``.

    Used to build training data before a model exists: nothing is scored, and every window's
    destinations go straight into novelty memory. Once there is a model use
    :func:`score_pending_events`, or stored events would skip the poisoning rule. Progress is
    recorded in the ``meta`` table.
    """
    with db.write_lock():
        pending = _pending_range(db, builder, until)
        if pending is None or db.event_time_bounds() is None:
            return []
        start, end = pending
        windows = builder.build(db.query_events(start=start, end=end), tracker, update_tracker=True)
        db.upsert_windows(windows)
        db.set_meta(WINDOWED_UNTIL_KEY, repr(end))
    return windows


# Training


@dataclass(frozen=True, slots=True)
class TrainingSelection:
    """which windows were used by training; why ones were left out"""

    windows: list[FeatureWindow]
    considered: int
    excluded_warmup: int
    excluded_alerted: int
    readmitted_expected: int
    start: float
    end: float


def select_training_windows(db: Database, days: int, warmup_hours: float) -> TrainingSelection:
    """most recent ``days`` of windows, (not warm-up and alerted)

    "Most recent" measured from the newest stored window

    filters:
        1. warm-up: windows from ``warmup_hours`` of collection are skipped
        2. poisoning: see :func:`netwatch.feedback.labels.build_training_set`.
    """
    bounds = db.window_time_bounds()
    if bounds is None:
        raise PipelineError("no feature windows yet: run `netwatch collect`, then `build-windows`")
    first_start, last_end = bounds
    start = last_end - days * DAY
    candidates = db.query_windows(start=start, end=last_end)
    warmup_end = first_start + warmup_hours * HOUR
    after_warmup = [w for w in candidates if w.window_start >= warmup_end]
    training = build_training_set(after_warmup, db.query_alerts(), db.list_labels())
    return TrainingSelection(
        windows=training.windows,
        considered=len(candidates),
        excluded_warmup=len(candidates) - len(after_warmup),
        excluded_alerted=training.excluded_alerted,
        readmitted_expected=training.readmitted_expected,
        start=start,
        end=last_end,
    )


@dataclass
class TrainedModel:
    """Everything needed to score new windows: fitted detectors + combiner

    Saved with ``pickle``
    """

    detectors: list[Detector]
    combiner: ScoreCombiner
    trained_at: float
    training_windows: int
    processes: list[str]
    version: str = __version__

    def score(self, windows: Sequence[FeatureWindow]) -> list[CombinedScore]:
        if not windows:
            return []
        raw = {d.name: d.score(windows) for d in self.detectors}
        return self.combiner.combine([w.process_name for w in windows], raw)

    def _contributors(self, score: CombinedScore) -> list[Detector]:
        """detectors that contributed to ``score``"""
        by_name = {d.name: d for d in self.detectors}
        return [by_name[name] for name in score.contributors if name in by_name]

    def explain(self, window: FeatureWindow, score: CombinedScore) -> list[str]:
        """merged reasons from detectors that contributed to ``score``"""
        reasons = merge_reasons(d.explain(window) for d in self._contributors(score))
        return reasons or [f"combined anomaly score {score.combined:.3f} reached the threshold"]

    def explain_beyond_destinations(self, window: FeatureWindow, score: CombinedScore) -> list[str]:
        """reasons for ``score`` that do not follow from the window's new destinations

        This is what alert deduplication compares as well as destinations (see
        :mod:`netwatch.alerts.dedup`): reasons from contributing detectors other than those
        flagged ``explains_destinations``
        """
        return merge_reasons(
            d.explain(window) for d in self._contributors(score) if not d.explains_destinations
        )


def train_model(
    windows: Sequence[FeatureWindow], config: Config, now: float | None = None
) -> TrainedModel:
    """Fit every enabled detector on ``windows``, then the combiner

    The combiner learns each detector's score distribution on the same training windows
    """
    if len(windows) < MIN_TRAINING_WINDOWS:
        raise PipelineError(
            f"only {len(windows)} training windows available; need at least "
            f"{MIN_TRAINING_WINDOWS}. Collect more data."
        )
    detectors = build_detectors(config.detectors)
    raw_scores = {d.name: d.fit(windows).score(windows) for d in detectors}
    combiner = ScoreCombiner(
        weights=detector_weights(config.detectors),
        contribution_threshold=config.scoring.contribution_threshold,
        min_process_history=config.scoring.min_process_history,
    ).fit([w.process_name for w in windows], raw_scores)
    return TrainedModel(
        detectors=detectors,
        combiner=combiner,
        trained_at=time.time() if now is None else now,
        training_windows=len(windows),
        processes=sorted({w.process_name for w in windows}),
    )


def save_model(model: TrainedModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(model, fh)


def _check_model_owner(path: Path) -> None:
    """as root, refuse a model file that someone other than root or the sudo user controls

    Unpickling runs code from the file, and ``sudo netwatch watch`` would run it as root. The
    file normally belongs to the user who ran ``netwatch train``, which is as trusted as the
    ``netwatch`` they are about to run with sudo. A file owned by anyone else, or writable by
    everyone, is not. Does nothing when not running as root
    """
    if os.name != "posix" or os.geteuid() != 0:
        return
    trusted = {0}
    sudo_uid = os.environ.get("SUDO_UID", "")
    if sudo_uid.isdigit():
        trusted.add(int(sudo_uid))
    info = path.stat()
    if info.st_uid not in trusted or info.st_mode & stat.S_IWOTH:
        raise PipelineError(
            f"refusing to load {path} as root: it must belong to root or to the user who ran "
            "sudo, and must not be writable by everyone. Loading a model runs code from it"
        )


def load_model(path: Path) -> TrainedModel:
    if not path.is_file():
        raise PipelineError(f"no trained model at {path}: run `netwatch train`")
    _check_model_owner(path)
    with path.open("rb") as fh:
        model = pickle.load(fh)  # own local file, untrusted input should be avoided
    if not isinstance(model, TrainedModel):
        raise PipelineError(f"{path} is not a netwatch model")
    if model.version != __version__:
        log.warning("model was trained with netwatch %s; consider retraining", model.version)
    return model


# Live monitoring


@dataclass(frozen=True, slots=True)
class ScoredWindow:
    """a window, its score, and alert (if any)"""

    window: FeatureWindow
    score: CombinedScore
    alert: Alert | None = None
    repeat: bool = False


class Monitor:
    """window, score, alert, learn, one closed time slot at a time

    1. Build each process's window *without* committing novelty.
    2. Score every window and combine detector scores.
    3. Windows at or above ``alert_threshold`` raise an explained alert, unless they only
       repeat an open one (:mod:`netwatch.alerts.dedup`)
    4. Only alert-free windows have their destinations committed to novelty memory. Repeats
       count as alerted.

    With a database, open alerts are read on every alerting window, so alerts from earlier
    runs absorb repeats too. A slot's reads and writes are one transaction
    (:meth:`~netwatch.storage.db.Database.write_lock`): a label recorded in another terminal
    cannot land between reading the open alerts and attaching a repeat, and a crash cannot
    leave a stored window without its alert. Sinks are called after it has committed. A slot
    that is already windowed is skipped, so two commands sharing a database never score the
    same slot twice.

    The monitor never saves ``tracker``: ``netwatch watch`` syncs it with the database around
    every slot (:func:`sync_tracker`).
    """

    def __init__(
        self,
        model: TrainedModel,
        tracker: NoveltyTracker,
        builder: WindowBuilder,
        alert_threshold: float,
        sinks: Iterable[AlertSink] = (),
        db: Database | None = None,
    ) -> None:
        self.model = model
        self.tracker = tracker
        self.builder = builder
        self.alert_threshold = alert_threshold
        self.sinks = list(sinks)
        self.db = db
        self._raised: dict[str, list[OpenAlert]] = defaultdict(list)

    def process_events(self, events: Iterable[ConnectionEvent]) -> list[ScoredWindow]:
        """process events from closed slots"""
        by_slot: dict[float, list[ConnectionEvent]] = defaultdict(list)
        for event in events:
            by_slot[window_start_for(event.timestamp, self.builder.window_seconds)].append(event)
        results: list[ScoredWindow] = []
        for slot in sorted(by_slot):
            results.extend(self.process_slot(by_slot[slot]))
        return results

    def process_slot(self, events: Sequence[ConnectionEvent]) -> list[ScoredWindow]:
        """process the events of a closed time slot"""
        if self.db is None:
            results = self._score_slot(events)
        else:
            with self.db.write_lock():
                if self._already_windowed(events):
                    return []
                results = self._score_slot(events)
                self._store(results)
        for result in results:
            if result.alert is not None and not result.repeat:
                for sink in self.sinks:
                    sink.emit(result.alert)
        return results

    def _already_windowed(self, events: Sequence[ConnectionEvent]) -> bool:
        """whether another command got to this slot first

        ``netwatch build-windows`` run next to ``netwatch watch`` can score a slot just before
        ``watch`` closes it. Scoring it again would attach the window to its own alert as a
        repeat
        """
        done = self.db.get_meta(WINDOWED_UNTIL_KEY) if self.db is not None else None
        return bool(events) and done is not None and events[0].timestamp < float(done)

    def _score_slot(self, events: Sequence[ConnectionEvent]) -> list[ScoredWindow]:
        """window and score one slot, deciding per window: quiet, new alert or repeat"""
        windows = self.builder.build(events, self.tracker, update_tracker=False)
        results: list[ScoredWindow] = []
        for window, score in zip(windows, self.model.score(windows), strict=True):
            if score.combined < self.alert_threshold:
                self.tracker.observe_window(window)
                results.append(ScoredWindow(window, score))
                continue
            parent = self._repeated_alert(window, score)
            if parent is not None:
                parent.repeat_window_starts.append(window.window_start)
                results.append(ScoredWindow(window, score, parent, repeat=True))
                continue
            alert = Alert(
                timestamp=window.window_end,
                process_name=window.process_name,
                window_start=window.window_start,
                combined_score=score.combined,
                per_detector_scores=dict(score.percentiles),
                reasons=self.model.explain(window, score),
            )
            if self.db is None:
                self._raised[window.process_name].append(OpenAlert(alert, window))
            results.append(ScoredWindow(window, score, alert))
        return results

    def _store(self, results: Sequence[ScoredWindow]) -> None:
        """save a slot's windows, alerts and repeats, and record the slot as windowed"""
        if self.db is None or not results:
            return
        self.db.upsert_windows(r.window for r in results)
        for result in results:
            if result.alert is None:
                continue
            if not result.repeat:
                self.db.insert_alert(result.alert)
            elif result.alert.id is not None:
                self.db.attach_repeat(result.alert.id, result.window.window_start)
        self.db.set_meta(WINDOWED_UNTIL_KEY, repr(results[0].window.window_end))

    def _open_alerts(self, process_name: str) -> list[OpenAlert]:
        """open alerts for ``process_name`` with windows"""
        if self.db is None:
            return [c for c in self._raised[process_name] if c.alert.status == AlertStatus.OPEN]
        candidates: list[OpenAlert] = []
        for alert in self.db.open_alerts(process_name):
            window = self.db.get_window(alert.process_name, alert.window_start)
            # without its window an alert's new destinations are unknown, so nothing can
            # safely be called a repeat of it
            if window is not None:
                candidates.append(OpenAlert(alert, window))
        return candidates

    def _repeated_alert(self, window: FeatureWindow, score: CombinedScore) -> Alert | None:
        """the open alert whose cause alerting ``window`` only repeats"""
        candidates = self._open_alerts(window.process_name)
        if not candidates:
            return None
        reasons = self.model.explain_beyond_destinations(window, score)
        return find_repeated(window, reasons, candidates)


def score_pending_events(
    db: Database, monitor: Monitor, until: float | None = None
) -> list[ScoredWindow]:
    """run stored events no window was built from yet through ``monitor``, oldest slot first

    Every stored event in a closed slot, whichever command stored it: what ``netwatch watch``
    collected itself, what a ``netwatch collect`` running next to it stored (it may see more
    processes, if only it runs as root), and what was collected while nothing was scoring.
    They are scored, alert and stay out of novelty memory alike.
    :func:`build_pending_windows` would let them skip all of that. Only closed slots before
    ``until`` (see :func:`_pending_range`). Progress is recorded up to there even when the
    slots held no events, so the same range is not read again
    """
    pending = _pending_range(db, monitor.builder, until)
    if pending is None:
        return []
    start, end = pending
    scored = monitor.process_events(db.query_events(start=start, end=end))
    with db.write_lock():
        done = db.get_meta(WINDOWED_UNTIL_KEY)
        if done is None or float(done) < end:
            db.set_meta(WINDOWED_UNTIL_KEY, repr(end))
    return scored


def score_closed_slots(
    db: Database, monitor: Monitor, tracker: NoveltyTracker, until: float | None = None
) -> list[ScoredWindow]:
    """one step of ``netwatch watch``: score the slots that closed since the last step

    Novelty memory is synced before scoring, to take in what a label changed meanwhile, and
    after it, to store what the slots taught. Costs one small read while no slot has closed
    """
    if _pending_range(db, monitor.builder, until) is None:
        return []
    sync_tracker(db, tracker)
    scored = score_pending_events(db, monitor, until)
    sync_tracker(db, tracker)
    return scored
