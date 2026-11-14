"""SQLite access layer: everything netwatch stores goes through here

Stages read and write through :class:`Database` and never issue SQL themselves, so the schema can
change in one place. It is a thin wrapper over the standard ``sqlite3`` module: no ORM, no
background threads, no network.

Notes:
    * Records go in and come out as the dataclasses from :mod:`netwatch.models`.
    * Windows store their features and context as JSON, so a new feature needs no migration.
    * An alert comes back with its repeat windows attached (the ``alert_repeats`` table).
    * Several commands can use one file at once (``watch`` runs for hours while you ``label``
      alerts in another terminal). A read followed by a write that depends on it must run under
      :meth:`Database.write_lock`, so no other process can commit in between and be overwritten.
    * The database is the *only* place real captured data is written. It lives under ``data/``,
      which is gitignored
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from types import TracebackType
from typing import Any

from netwatch.models import (
    Alert,
    AlertStatus,
    ConnectionEvent,
    FeatureWindow,
    Label,
    NoveltyRecord,
    Verdict,
    WindowFeatures,
)

IN_MEMORY = ":memory:"

_EVENT_COLUMNS = (
    "timestamp",
    "pid",
    "process_name",
    "exe_path",
    "parent_name",
    "local_ip",
    "local_port",
    "remote_ip",
    "remote_port",
    "protocol",
    "status",
)


def load_schema() -> str:
    """text of the bundled ``schema.sql``"""
    return resources.files("netwatch.storage").joinpath("schema.sql").read_text(encoding="utf-8")


def _time_filter(
    column: str, start: float | None, end: float | None, process_name: str | None
) -> tuple[str, list[Any]]:
    """``WHERE`` clause for the usual [start, end) + process filter"""
    clauses: list[str] = []
    params: list[Any] = []
    if start is not None:
        clauses.append(f"{column} >= ?")
        params.append(start)
    if end is not None:
        clauses.append(f"{column} < ?")
        params.append(end)
    if process_name is not None:
        clauses.append("process_name = ?")
        params.append(process_name)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


class Database:
    """typed helpers over one SQLite connection

    Create one with :meth:`open`. It applies the schema, so a new path gives a ready database.
    Works as a context manager
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._write_locked = False

    # opening and closing

    @classmethod
    def open(cls, path: str | Path) -> Database:
        """open the database at ``path`` (created if missing) and apply the schema

        ``":memory:"`` gives a throwaway in-memory database. The demo and the tests use it so
        they never touch ``data/``
        """
        if str(path) != IN_MEMORY:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        db = cls(sqlite3.connect(str(path)))
        db.init_schema()
        return db

    def init_schema(self) -> None:
        """create any missing tables and indexes (safe to repeat)"""
        self.conn.executescript(load_schema())

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """commit on success, roll back on error

        Inside :meth:`write_lock` this joins the outer transaction, which commits or rolls back
        as a whole
        """
        if self._write_locked:
            yield self.conn
            return
        with self.conn:
            yield self.conn

    @contextmanager
    def write_lock(self) -> Iterator[None]:
        """run a read and the write that depends on it as one transaction other writers wait for

        Without it, another process can commit between the read and the write, and the write
        then silently discards that commit (a lost update). ``BEGIN IMMEDIATE`` takes SQLite's
        write lock *before* the first read. A writer in another process waits (up to the busy
        timeout, 5 seconds by default) until this block has committed, and every read inside the
        block sees the latest committed state. Readers are only blocked during the commit itself.

        A plain ``BEGIN`` takes the lock at the first write. Two processes that had both read
        would then deadlock, and one would fail with "database is locked" instead of waiting.

        Writes made through this class inside the block join the transaction. It commits when
        the block exits normally and rolls back if anything raises, the commit included.

        Blocks nest: an inner block joins the outer transaction, which commits or rolls back as
        a whole
        """
        if self._write_locked:
            yield
            return
        self.conn.execute("BEGIN IMMEDIATE")
        self._write_locked = True
        try:
            yield
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise
        finally:
            self._write_locked = False

    # events

    def insert_events(
        self, events: Iterable[ConnectionEvent], seen_since: float | None = None
    ) -> int:
        """append raw connection events. Returns how many were written

        With ``seen_since``, an event is left out when the same socket (process id, both
        addresses and ports, protocol) is already stored with a later timestamp: another
        collector writing to this database saw the connection too. Without it, two collectors
        would store every shared connection twice and each window would count it twice
        """
        events = list(events)
        if not events:
            return 0
        placeholders = ", ".join("?" for _ in _EVENT_COLUMNS)
        with self.write_lock():
            if seen_since is not None:
                events = [e for e in events if not self._stored_since(e, seen_since)]
            self.conn.executemany(
                f"INSERT INTO events ({', '.join(_EVENT_COLUMNS)}) VALUES ({placeholders})",
                [tuple(getattr(e, c) for c in _EVENT_COLUMNS) for e in events],
            )
        return len(events)

    def _stored_since(self, event: ConnectionEvent, since: float) -> bool:
        """whether the socket behind ``event`` is already stored with a timestamp after ``since``"""
        row = self.conn.execute(
            "SELECT 1 FROM events WHERE timestamp > ? AND pid IS ? AND local_ip = ? "
            "AND local_port = ? AND remote_ip = ? AND remote_port = ? AND protocol = ? LIMIT 1",
            (
                since,
                event.pid,
                event.local_ip,
                event.local_port,
                event.remote_ip,
                event.remote_port,
                event.protocol,
            ),
        ).fetchone()
        return row is not None

    def query_events(
        self,
        start: float | None = None,
        end: float | None = None,
        process_name: str | None = None,
    ) -> list[ConnectionEvent]:
        """events with ``start <= timestamp < end``, oldest first"""
        where, params = _time_filter("timestamp", start, end, process_name)
        cursor = self.conn.execute(
            f"SELECT {', '.join(_EVENT_COLUMNS)} FROM events{where} ORDER BY timestamp, id",
            params,
        )
        return [ConnectionEvent(**dict(row)) for row in cursor]

    def count_events(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def event_time_bounds(self) -> tuple[float, float] | None:
        """``(earliest, latest)`` event timestamp, or ``None`` if empty."""
        row = self.conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM events").fetchone()
        return None if row[0] is None else (float(row[0]), float(row[1]))

    # feature windows

    def upsert_windows(self, windows: Iterable[FeatureWindow]) -> int:
        """insert windows, replacing any stored window with the same process and start"""
        rows = [
            (
                w.process_name,
                w.window_start,
                w.window_end,
                json.dumps(w.features.to_dict()),
                json.dumps(w.context_dict()),
            )
            for w in windows
        ]
        if not rows:
            return 0
        with self._transaction() as conn:
            conn.executemany(
                """
                INSERT INTO windows (process_name, window_start, window_end, features, context)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (process_name, window_start) DO UPDATE SET
                    window_end = excluded.window_end,
                    features = excluded.features,
                    context = excluded.context
                """,
                rows,
            )
        return len(rows)

    @staticmethod
    def _row_to_window(row: sqlite3.Row) -> FeatureWindow:
        context = json.loads(row["context"])
        return FeatureWindow(
            process_name=row["process_name"],
            window_start=row["window_start"],
            window_end=row["window_end"],
            features=WindowFeatures.from_dict(json.loads(row["features"])),
            exe_path=context.get("exe_path"),
            remote_ips=tuple(context.get("remote_ips", ())),
            remote_ports=tuple(context.get("remote_ports", ())),
            new_remote_ips=tuple(context.get("new_remote_ips", ())),
            new_remote_ports=tuple(context.get("new_remote_ports", ())),
        )

    def query_windows(
        self,
        start: float | None = None,
        end: float | None = None,
        process_name: str | None = None,
    ) -> list[FeatureWindow]:
        """windows with ``start <= window_start < end``, oldest first"""
        where, params = _time_filter("window_start", start, end, process_name)
        cursor = self.conn.execute(
            f"SELECT * FROM windows{where} ORDER BY window_start, process_name", params
        )
        return [self._row_to_window(row) for row in cursor]

    def get_window(self, process_name: str, window_start: float) -> FeatureWindow | None:
        row = self.conn.execute(
            "SELECT * FROM windows WHERE process_name = ? AND window_start = ?",
            (process_name, window_start),
        ).fetchone()
        return None if row is None else self._row_to_window(row)

    def window_time_bounds(self) -> tuple[float, float] | None:
        """``(earliest window_start, latest window_end)``, or ``None`` if empty."""
        row = self.conn.execute("SELECT MIN(window_start), MAX(window_end) FROM windows").fetchone()
        return None if row[0] is None else (float(row[0]), float(row[1]))

    def count_windows(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM windows").fetchone()[0])

    # alerts

    def insert_alert(self, alert: Alert) -> int:
        """store an alert, set ``alert.id`` and return the new id"""
        with self._transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO alerts (timestamp, process_name, window_start, combined_score,
                                    per_detector_scores, reasons, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.timestamp,
                    alert.process_name,
                    alert.window_start,
                    alert.combined_score,
                    json.dumps(alert.per_detector_scores),
                    json.dumps(alert.reasons),
                    str(alert.status),
                ),
            )
        alert.id = int(cursor.lastrowid or 0)
        return alert.id

    @staticmethod
    def _row_to_alert(row: sqlite3.Row, repeat_window_starts: list[float]) -> Alert:
        return Alert(
            id=row["id"],
            timestamp=row["timestamp"],
            process_name=row["process_name"],
            window_start=row["window_start"],
            combined_score=row["combined_score"],
            per_detector_scores=json.loads(row["per_detector_scores"]),
            reasons=json.loads(row["reasons"]),
            status=AlertStatus(row["status"]),
            repeat_window_starts=repeat_window_starts,
        )

    def _fetch_alerts(self, query: str, params: Sequence[Any] = ()) -> list[Alert]:
        """run ``query`` (a ``SELECT * FROM alerts ...``) and attach each alert's repeats

        The repeats come from one more query that uses ``query`` as a subquery for the alert
        ids: no extra query per alert, and no limit on how many alerts can be loaded at once
        """
        rows = self.conn.execute(query, params).fetchall()
        repeats: dict[int, list[float]] = defaultdict(list)
        if rows:
            cursor = self.conn.execute(
                "SELECT alert_id, window_start FROM alert_repeats "
                f"WHERE alert_id IN (SELECT id FROM ({query})) ORDER BY window_start",
                params,
            )
            for alert_id, window_start in cursor:
                repeats[alert_id].append(window_start)
        return [self._row_to_alert(row, repeats[row["id"]]) for row in rows]

    def get_alert(self, alert_id: int) -> Alert | None:
        found = self._fetch_alerts("SELECT * FROM alerts WHERE id = ?", (alert_id,))
        return found[0] if found else None

    def list_alerts(self, limit: int = 20) -> list[Alert]:
        """most recent alerts first"""
        return self._fetch_alerts(
            "SELECT * FROM alerts ORDER BY timestamp DESC, id DESC LIMIT ?", (limit,)
        )

    def query_alerts(self, start: float | None = None, end: float | None = None) -> list[Alert]:
        """alerts whose window starts in ``[start, end)``, oldest first"""
        where, params = _time_filter("window_start", start, end, None)
        return self._fetch_alerts(f"SELECT * FROM alerts{where} ORDER BY window_start, id", params)

    def open_alerts(self, process_name: str) -> list[Alert]:
        """unlabelled alerts for ``process_name``, oldest first"""
        return self._fetch_alerts(
            "SELECT * FROM alerts WHERE process_name = ? AND status = ? ORDER BY window_start, id",
            (process_name, str(AlertStatus.OPEN)),
        )

    def attach_repeat(self, alert_id: int, window_start: float) -> None:
        """record that the window starting at ``window_start`` repeats alert ``alert_id``

        The window belongs to the alert's process. Attaching the same window twice does nothing,
        so a slot can safely be processed again
        """
        with self._transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO alert_repeats (alert_id, window_start) VALUES (?, ?)",
                (alert_id, window_start),
            )

    def set_alert_status(self, alert_id: int, status: AlertStatus) -> None:
        with self._transaction() as conn:
            conn.execute("UPDATE alerts SET status = ? WHERE id = ?", (str(status), alert_id))

    def suspicious_windows(self) -> list[tuple[FeatureWindow, float]]:
        """every stored window covered by an alert labelled suspicious, with the label's time

        Covers the window that raised the alert and its repeats. The time is that of the alert's
        latest label, the one that made it suspicious
        """
        cursor = self.conn.execute(
            """
            SELECT w.*, (SELECT MAX(l.timestamp) FROM labels l WHERE l.alert_id = a.id)
                       AS labelled_at
            FROM alerts a
            JOIN windows w ON w.process_name = a.process_name
            WHERE a.status = ?
              AND (w.window_start = a.window_start
                   OR w.window_start IN (SELECT r.window_start FROM alert_repeats r
                                         WHERE r.alert_id = a.id))
            ORDER BY w.window_start, a.id
            """,
            (str(AlertStatus.SUSPICIOUS),),
        )
        return [
            (self._row_to_window(row), float(row["labelled_at"]))
            for row in cursor
            if row["labelled_at"] is not None
        ]

    # labels

    def insert_label(self, label: Label) -> int:
        with self._transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO labels (alert_id, verdict, timestamp, note) VALUES (?, ?, ?, ?)",
                (label.alert_id, str(label.verdict), label.timestamp, label.note),
            )
        return int(cursor.lastrowid or 0)

    def list_labels(self) -> list[Label]:
        """all labels, oldest first, so the last one for an alert is the one that wins"""
        cursor = self.conn.execute(
            "SELECT alert_id, verdict, timestamp, note FROM labels ORDER BY timestamp, id"
        )
        return [
            Label(
                alert_id=row["alert_id"],
                verdict=Verdict(row["verdict"]),
                timestamp=row["timestamp"],
                note=row["note"],
            )
            for row in cursor
        ]

    # novelty memory

    def save_novelty(self, records: Iterable[NoveltyRecord]) -> int:
        """replace the stored novelty memory with ``records``

        Only safe when ``records`` already include everything stored: merge in
        :meth:`load_novelty` first, both under :meth:`write_lock` (see
        :func:`netwatch.pipeline.sync_tracker`)
        """
        rows = [tuple(r) for r in records]
        with self._transaction() as conn:
            conn.execute("DELETE FROM novelty")
            conn.executemany(
                """
                INSERT INTO novelty (process_name, kind, value, familiarity, last_seen, first_seen)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def load_novelty(self) -> list[NoveltyRecord]:
        cursor = self.conn.execute(
            "SELECT process_name, kind, value, familiarity, last_seen, first_seen FROM novelty"
        )
        return [NoveltyRecord(*row) for row in cursor]

    # meta

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )


def hand_back_to_sudo_user(*paths: Path) -> None:
    """when running under ``sudo``, give ``paths`` back to the user who ran it

    Collection may need root on macOS. Without this, ``sudo netwatch collect`` leaves a
    root-owned database that later commands without root (``train``, ``alerts``) cannot write
    to. Does nothing when not root or not started through sudo
    """
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if os.name != "posix" or os.geteuid() != 0 or uid is None or gid is None:
        return
    for path in paths:
        if path.exists():
            os.chown(path, int(uid), int(gid))
