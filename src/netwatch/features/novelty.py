"""Memory of which (process, destination) pairs are familiar, fading over time

Answers "has this process talked to this IP / port before?". Malware has to talk to *somewhere*,
and that place is usually new for the process it hides in.

Decay:
    A pair seen once, months ago, should not stay familiar forever. Each pair has a *familiarity*
    that halves every ``half_life_days`` and grows by 1 on every sighting::

        familiarity(t) = familiarity(t0) * 0.5 ** ((t - t0) / half_life) + 1

    A pair is known while its familiarity is at least ``min_familiarity``. With the defaults
    (14 days, 0.5) a pair seen once is remembered for about two weeks, and a pair seen every day
    for a month is remembered for months.

Looking up vs. learning:
    Looking a pair up never changes it. Pairs are only learned through
    :meth:`NoveltyTracker.observe`, so the live pipeline can hold back the pairs of a window
    that alerted (the poisoning rule). An attacker's address must not become familiar just by
    being contacted again and again.

Several writers:
    ``netwatch watch`` keeps a tracker in memory while it runs, and ``netwatch label`` loads,
    changes and saves its own copy. So a stored memory is never overwritten, only *merged*
    (:meth:`NoveltyTracker.merge_records`, used by :func:`netwatch.pipeline.sync_tracker`).

    Per pair, the merge keeps the more familiar side, compared at the later ``last_seen``.
    Merging twice, or in a different order, gives the same result, so merging a stale copy back
    in changes nothing.

    The cost: if both sides saw the same pair since they last merged, only one side's sightings
    are kept. (Adding them would count the shared history twice on every save.) Counting too few
    is the safe direction: a pair may fade a little sooner, but it never becomes familiar
    without being seen

Forgetting:
    A merge can only add, so a pair one writer dropped would come straight back from another
    writer's copy. Pairs are therefore never forgotten by hand. The one exception follows from
    the labels: destinations an alert labelled ``suspicious`` flagged as new are forgotten up to
    the time of the label (:meth:`NoveltyTracker.forget`). Every writer reads the same labels and
    applies the same rule when it syncs, so the pair stays gone on all sides
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

from netwatch.models import FeatureWindow, NoveltyRecord

SECONDS_PER_DAY = 86_400.0

T = TypeVar("T", str, int)


class PairKind(StrEnum):
    """what kind of destination a remembered pair holds"""

    IP = "ip"
    PORT = "port"


@dataclass(slots=True)
class _Memory:
    familiarity: float
    last_seen: float
    first_seen: float


class NoveltyTracker:
    """remembers (process, ip) and (process, port) pairs, with time decay

    Arguments:
        half_life_days: time for a familiarity to halve when the pair is not seen again
        min_familiarity: a pair is known at or above this (decayed) familiarity
    """

    def __init__(self, half_life_days: float = 14.0, min_familiarity: float = 0.5) -> None:
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        self.half_life_days = half_life_days
        self.min_familiarity = min_familiarity
        self._half_life_seconds = half_life_days * SECONDS_PER_DAY
        self._memory: dict[tuple[str, str, str], _Memory] = {}

    def __len__(self) -> int:
        return len(self._memory)

    # lookups (never change anything)

    def _decayed(self, memory: _Memory, at: float) -> float:
        age = max(0.0, at - memory.last_seen)
        return memory.familiarity * 0.5 ** (age / self._half_life_seconds)

    def familiarity(self, process: str, kind: PairKind, value: str | int, at: float) -> float:
        """familiarity of a pair at time ``at``, after decay (0.0 if never seen)"""
        memory = self._memory.get((process, str(kind), str(value)))
        return 0.0 if memory is None else self._decayed(memory, at)

    def is_known(self, process: str, kind: PairKind, value: str | int, at: float) -> bool:
        return self.familiarity(process, kind, value, at) >= self.min_familiarity

    def novel(self, process: str, kind: PairKind, values: Iterable[T], at: float) -> list[T]:
        """distinct ``values`` that are *not* known for ``process`` at ``at``"""
        return sorted({v for v in values if not self.is_known(process, kind, v, at)})

    def known_processes(self) -> set[str]:
        return {process for process, _, _ in self._memory}

    # learning

    def observe(self, process: str, kind: PairKind, values: Iterable[str | int], at: float) -> None:
        """record one sighting of each distinct value at time ``at``"""
        for value in set(values):
            key = (process, str(kind), str(value))
            memory = self._memory.get(key)
            if memory is None:
                self._memory[key] = _Memory(familiarity=1.0, last_seen=at, first_seen=at)
            else:
                memory.familiarity = self._decayed(memory, at) + 1.0
                memory.last_seen = max(memory.last_seen, at)

    def observe_window(self, window: FeatureWindow) -> None:
        """commit every destination in ``window`` to memory (one sighting each)"""
        self.observe(window.process_name, PairKind.IP, window.remote_ips, window.window_end)
        self.observe(window.process_name, PairKind.PORT, window.remote_ports, window.window_end)

    def forget(
        self, process: str, kind: PairKind, values: Iterable[str | int], until: float
    ) -> int:
        """drop each pair last seen at or before ``until``. Returns how many

        A pair seen again after ``until`` is kept: that sighting was learned on its own merits.
        Repeating the call changes nothing
        """
        dropped = 0
        for value in set(values):
            key = (process, str(kind), str(value))
            memory = self._memory.get(key)
            if memory is not None and memory.last_seen <= until:
                del self._memory[key]
                dropped += 1
        return dropped

    def forget_new_destinations(self, window: FeatureWindow, until: float) -> int:
        """forget the destinations ``window`` flagged as new, as last seen up to ``until``

        Only the *new* ones: the window's other destinations were familiar before it and stay so
        """
        return self.forget(
            window.process_name, PairKind.IP, window.new_remote_ips, until
        ) + self.forget(window.process_name, PairKind.PORT, window.new_remote_ports, until)

    def prune(self, at: float, floor: float = 0.01) -> int:
        """drop pairs that have faded below ``floor``. Returns how many"""
        stale = [k for k, m in self._memory.items() if self._decayed(m, at) < floor]
        for key in stale:
            del self._memory[key]
        return len(stale)

    # saving and loading

    def to_records(self) -> list[NoveltyRecord]:
        return [
            NoveltyRecord(process, kind, value, m.familiarity, m.last_seen, m.first_seen)
            for (process, kind, value), m in self._memory.items()
        ]

    @classmethod
    def from_records(
        cls,
        records: Iterable[NoveltyRecord],
        half_life_days: float = 14.0,
        min_familiarity: float = 0.5,
    ) -> NoveltyTracker:
        tracker = cls(half_life_days=half_life_days, min_familiarity=min_familiarity)
        tracker.merge_records(records)
        return tracker

    def merge_records(self, records: Iterable[NoveltyRecord]) -> None:
        """merge another copy of the memory into this one, never losing a pair

        Pairs only in ``records`` are added as they are. For a pair on both sides: the larger
        decayed familiarity, the later ``last_seen``, the earlier ``first_seen``. Repeating the
        merge or changing its order gives the same result (see "Several writers" above)
        """
        for r in records:
            key = (r.process_name, r.kind, r.value)
            theirs = _Memory(
                familiarity=r.familiarity, last_seen=r.last_seen, first_seen=r.first_seen
            )
            mine = self._memory.get(key)
            if mine is None:
                self._memory[key] = theirs
                continue
            at = max(mine.last_seen, theirs.last_seen)
            mine.familiarity = max(self._decayed(mine, at), self._decayed(theirs, at))
            mine.last_seen = at
            mine.first_seen = min(mine.first_seen, theirs.first_seen)
