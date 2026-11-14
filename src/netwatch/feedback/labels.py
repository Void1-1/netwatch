"""User labels, and the poisoning rule that decides what the models learn from

The problem:
    A detector retrained on recent data learns whatever happened recently, attacks included. If
    a beacon runs for a week and the model is retrained on that week, the beacon becomes
    "normal" and stops alerting. An attacker only has to be patient.

The rule:
    A window that alerted is left out of training unless the user labelled its alert
    ``expected`` ("yes, that was me running a backup to a new server"). Unlabelled and
    ``suspicious`` alerts stay out. Novelty memory works the same way: the live monitor does not
    commit an alerting window's destinations, and labelling the alert ``expected`` is what
    commits them (see :func:`record_label`).

    Repeats count as alerted too: windows attached to an open alert because they only repeated
    its cause (see :mod:`netwatch.alerts.dedup`). They are left out, and committed, together
    with their alert, so one label settles them all.

Labels are append-only and the latest one wins, so a change of mind is always possible and
always recorded. That holds for novelty memory too: an alert whose latest label is
``suspicious`` has the destinations it flagged as new forgotten again, up to the time of that
label (see :func:`forget_suspicious_destinations`). Relabelling it ``expected`` commits them
again
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from netwatch.features.novelty import NoveltyTracker
from netwatch.models import Alert, AlertStatus, FeatureWindow, Label, Verdict
from netwatch.storage.db import Database


@dataclass(frozen=True, slots=True)
class TrainingSet:
    """windows that passed the poisoning rule, plus counts for reports

    Attributes:
        windows: windows safe to train on
        excluded_alerted: windows left out because they alerted (raised an alert or repeated
            an open one) and were not labelled expected
        readmitted_expected: alerting windows kept because their alert was labelled expected
    """

    windows: list[FeatureWindow]
    excluded_alerted: int
    readmitted_expected: int


def latest_verdicts(labels: Iterable[Label]) -> dict[int, Verdict]:
    """latest verdict per alert id"""
    latest: dict[int, Label] = {}
    for label in labels:
        current = latest.get(label.alert_id)
        if current is None or label.timestamp >= current.timestamp:
            latest[label.alert_id] = label
    return {alert_id: label.verdict for alert_id, label in latest.items()}


def build_training_set(
    windows: Sequence[FeatureWindow],
    alerts: Iterable[Alert],
    labels: Iterable[Label],
) -> TrainingSet:
    """apply the poisoning rule to ``windows``

    A window is left out when an alert covers it (as the window that raised the alert or as one
    of its repeats) and that alert's latest verdict is not :attr:`Verdict.EXPECTED`
    """
    verdicts = latest_verdicts(labels)
    blocked: set[tuple[str, float]] = set()
    expected: set[tuple[str, float]] = set()
    for alert in alerts:
        keys = alert.window_keys
        if alert.id is not None and verdicts.get(alert.id) == Verdict.EXPECTED:
            expected.update(keys)
        else:
            blocked.update(keys)
    # a window with both an expected and an unreviewed alert stays out: the cautious choice
    expected -= blocked

    kept = [w for w in windows if w.key not in blocked]
    return TrainingSet(
        windows=kept,
        excluded_alerted=len(windows) - len(kept),
        readmitted_expected=sum(1 for w in kept if w.key in expected),
    )


def forget_suspicious_destinations(db: Database, tracker: NoveltyTracker) -> int:
    """make ``tracker`` forget what alerts labelled suspicious flagged as new. Returns how many

    For every window covered by an alert whose latest label is ``suspicious``, the destinations
    that window flagged as new are dropped if they were last seen at or before the label. This
    undoes an earlier ``expected`` label, and any quiet window that committed them while the
    alert was still open or wrongly settled. A sighting after the label is kept.

    Suspicious wins while it stands: a sighting from before the label cannot make such a
    destination familiar again, not even through another alert labelled ``expected``.

    It is derived from the stored labels every time, not remembered: every command that syncs
    novelty memory (:func:`netwatch.pipeline.sync_tracker`) reaches the same result, so a
    running ``netwatch watch`` cannot merge the pair back in, and relabelling the alert
    ``expected`` lifts it
    """
    return sum(
        tracker.forget_new_destinations(window, until=labelled_at)
        for window, labelled_at in db.suspicious_windows()
    )


def record_label(
    db: Database,
    alert_id: int,
    verdict: Verdict,
    note: str | None = None,
    tracker: NoveltyTracker | None = None,
    now: float | None = None,
) -> Label:
    """store a verdict for an alert and apply what follows from it

    * Adds a :class:`Label` and updates the alert's status, as one transaction. The verdict
      covers every window of the alert, repeats included.
    * ``expected`` with a ``tracker``: commits the destinations of every covered window to
      novelty memory, oldest first, one sighting per window, as if they had never alerted. The
      same harmless activity is then not reported as new next time.
    * ``suspicious`` with a ``tracker``: forgets the destinations those windows flagged as new,
      as last seen up to now. This is what takes back an earlier ``expected``.

    The caller saves the tracker with :func:`netwatch.pipeline.sync_tracker`, which merges, so
    a ``netwatch watch`` running at the same time neither loses these changes nor overwrites
    them.

    Raises:
        KeyError: if ``alert_id`` does not exist
    """
    with db.write_lock():
        alert = db.get_alert(alert_id)
        if alert is None:
            raise KeyError(f"no alert with id {alert_id}")

        label = Label(
            alert_id=alert_id,
            verdict=verdict,
            timestamp=time.time() if now is None else now,
            note=note,
        )
        db.insert_label(label)
        db.set_alert_status(alert_id, AlertStatus(verdict.value))
        windows = [
            window
            for process_name, window_start in alert.window_keys
            if (window := db.get_window(process_name, window_start)) is not None
        ]

    if tracker is not None:
        for window in windows:
            if verdict == Verdict.EXPECTED:
                tracker.observe_window(window)
            else:
                tracker.forget_new_destinations(window, until=label.timestamp)
    return label
