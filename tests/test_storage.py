"""SQLite storage: every record type comes back unchanged after saving and loading"""

from __future__ import annotations

import sqlite3

import pytest

from netwatch.models import Alert, AlertStatus, Label, NoveltyRecord, Verdict
from netwatch.storage.db import Database


@pytest.fixture
def db():
    with Database.open(":memory:") as database:
        yield database


def test_schema_is_idempotent(db: Database) -> None:
    db.init_schema()
    db.init_schema()
    tables = {
        row[0] for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert {"events", "windows", "alerts", "alert_repeats", "labels", "novelty", "meta"} <= tables


def test_spec_indexes_exist(db: Database) -> None:
    indexes = {
        row[0] for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
    }
    assert {
        "idx_events_timestamp",
        "idx_events_process_name",
        "idx_windows_process_start",
    } <= indexes


def test_file_database_creates_its_directory(tmp_path) -> None:
    path = tmp_path / "nested" / "netwatch.db"
    with Database.open(path) as database:
        assert database.count_events() == 0
    assert path.exists()


def test_events_round_trip_with_filters(db: Database, make_event, noon) -> None:
    events = [
        make_event(noon, "a"),
        make_event(noon + 10, "b", pid=None, exe_path=None),
        make_event(noon + 20, "a"),
    ]
    assert db.insert_events(events) == 3
    assert db.insert_events([]) == 0
    assert db.query_events() == events
    assert db.query_events(start=noon + 5, end=noon + 20) == [events[1]]
    assert db.query_events(process_name="a") == [events[0], events[2]]
    assert db.event_time_bounds() == (noon, noon + 20)


def test_events_another_collector_stored_are_left_out(db: Database, make_event, noon) -> None:
    seen = make_event(noon, "a")
    assert db.insert_events([seen]) == 1

    again = make_event(noon + 1, "a")  # the same socket, reported by a second collector
    other = make_event(noon + 1, "a", remote_port=8443)
    assert db.insert_events([again, other], seen_since=noon - 1) == 1
    assert db.query_events() == [seen, other]

    # the stored one is older than the gap being checked: this is a new connection
    assert db.insert_events([again], seen_since=noon) == 1
    # and without a gap nothing is compared
    assert db.insert_events([again]) == 1
    assert db.count_events() == 4


def test_windows_round_trip_and_upsert(db: Database, make_window, noon) -> None:
    window = make_window(
        "app",
        noon,
        connection_count=7,
        exe_in_suspicious_path=True,
        remote_ips=("192.0.2.1",),
        remote_ports=(443,),
        new_remote_ips=("192.0.2.1",),
    )
    db.upsert_windows([window])
    assert db.get_window("app", noon) == window

    replacement = make_window("app", noon, connection_count=9)
    db.upsert_windows([replacement])
    assert db.count_windows() == 1
    assert db.get_window("app", noon) == replacement
    assert db.window_time_bounds() == (noon, noon + 300)
    assert db.get_window("app", noon + 300) is None


def test_alerts_and_status_updates(db: Database, noon) -> None:
    older = Alert(noon, "app", noon - 300, 0.999, {"novelty": 0.999}, ["first connection"])
    newer = Alert(noon + 600, "app", noon + 300, 1.0, {"baseline": 1.0}, ["spike", "periodic"])
    assert db.insert_alert(older) == older.id == 1
    db.insert_alert(newer)
    assert db.get_alert(newer.id) == newer
    assert db.list_alerts(limit=1) == [newer]  # most recent first
    assert db.query_alerts(start=noon) == [newer]

    db.set_alert_status(older.id, AlertStatus.EXPECTED)
    assert db.get_alert(older.id).status == AlertStatus.EXPECTED
    assert db.get_alert(999) is None


def test_alerts_come_back_with_their_repeats(db: Database, noon) -> None:
    quiet = Alert(noon + 300, "other", noon, 1.0, {}, ["x"])
    alert = Alert(noon + 600, "app", noon + 300, 1.0, {}, ["first connection"])
    db.insert_alert(quiet)
    db.insert_alert(alert)
    db.attach_repeat(alert.id, noon + 1200)
    db.attach_repeat(alert.id, noon + 900)
    db.attach_repeat(alert.id, noon + 900)  # attaching twice does nothing

    stored = db.get_alert(alert.id)
    assert stored.repeat_window_starts == [noon + 900, noon + 1200]  # oldest first
    assert stored.window_keys == [("app", noon + 300), ("app", noon + 900), ("app", noon + 1200)]
    assert (stored.occurrences, stored.last_window_start) == (3, noon + 1200)
    assert db.get_alert(quiet.id).repeat_window_starts == []
    assert db.get_alert(quiet.id).occurrences == 1

    # every way of loading alerts attaches the right repeats to the right alert
    assert db.list_alerts(limit=1) == [stored]
    assert db.query_alerts() == [db.get_alert(quiet.id), stored]
    assert db.query_alerts(start=noon + 300) == [stored]


def test_open_alerts_are_per_process_and_unlabelled(db: Database, noon) -> None:
    older = Alert(noon, "app", noon, 1.0, {}, ["a"])
    newer = Alert(noon + 300, "app", noon + 300, 1.0, {}, ["b"])
    labelled = Alert(noon + 600, "app", noon + 600, 1.0, {}, ["c"])
    elsewhere = Alert(noon, "other", noon, 1.0, {}, ["d"])
    for alert in (newer, older, labelled, elsewhere):
        db.insert_alert(alert)
    db.set_alert_status(labelled.id, AlertStatus.SUSPICIOUS)
    assert db.open_alerts("app") == [older, newer]  # oldest first
    assert db.open_alerts("nobody") == []


def test_labels_round_trip(db: Database, noon) -> None:
    alert = Alert(noon, "app", noon, 1.0, {}, ["x"])
    db.insert_alert(alert)
    db.insert_label(Label(alert.id, Verdict.SUSPICIOUS, noon + 1, "looked odd"))
    db.insert_label(Label(alert.id, Verdict.EXPECTED, noon + 2))
    assert db.list_labels() == [
        Label(alert.id, Verdict.SUSPICIOUS, noon + 1, "looked odd"),
        Label(alert.id, Verdict.EXPECTED, noon + 2, None),
    ]


def test_novelty_records_replace_previous_state(db: Database) -> None:
    first = [NoveltyRecord("app", "ip", "192.0.2.1", 1.0, 10.0, 10.0)]
    second = [
        NoveltyRecord("app", "ip", "192.0.2.2", 2.5, 20.0, 5.0),
        NoveltyRecord("app", "port", "443", 1.0, 20.0, 20.0),
    ]
    db.save_novelty(first)
    db.save_novelty(second)
    assert sorted(db.load_novelty()) == sorted(second)


def test_write_lock_commits_or_rolls_back_as_a_whole(db: Database) -> None:
    record = NoveltyRecord("app", "ip", "192.0.2.1", 1.0, 10.0, 10.0)
    with pytest.raises(RuntimeError), db.write_lock():
        db.save_novelty([record])
        db.set_meta("windowed_until", "100.0")
        raise RuntimeError("interrupted halfway")
    assert db.load_novelty() == []
    assert db.get_meta("windowed_until") is None

    with db.write_lock():
        db.save_novelty([record])
        db.set_meta("windowed_until", "100.0")
    assert db.load_novelty() == [record]
    assert db.get_meta("windowed_until") == "100.0"


def test_write_locks_nest_into_one_transaction(db: Database) -> None:
    with pytest.raises(RuntimeError), db.write_lock():
        with db.write_lock():
            db.set_meta("inner", "1")
        db.set_meta("outer", "1")  # the inner block did not commit or release anything
        raise RuntimeError("interrupted after the inner block")
    assert db.get_meta("inner") is None
    assert db.get_meta("outer") is None

    with db.write_lock(), db.write_lock():
        db.set_meta("inner", "1")
    assert db.get_meta("inner") == "1"


def test_write_lock_is_held_from_the_first_read(tmp_path) -> None:
    path = tmp_path / "netwatch.db"
    record = NoveltyRecord("app", "ip", "192.0.2.1", 1.0, 10.0, 10.0)
    with Database.open(path) as holder, Database(sqlite3.connect(path, timeout=0)) as other:
        with holder.write_lock():
            holder.load_novelty()  # only a read so far
            with pytest.raises(sqlite3.OperationalError, match="locked"), other.write_lock():
                pytest.fail("a second process started a read-modify-write")
            holder.save_novelty([record])
        assert other.load_novelty() == [record]


def test_meta_values(db: Database) -> None:
    assert db.get_meta("windowed_until") is None
    db.set_meta("windowed_until", "100.0")
    db.set_meta("windowed_until", "200.0")
    assert db.get_meta("windowed_until") == "200.0"


def test_suspicious_windows_cover_the_alert_and_its_repeats(
    db: Database, make_window, noon
) -> None:
    first, repeat, other = (make_window("app", noon + i * 300) for i in range(3))
    db.upsert_windows([first, repeat, other, make_window("browser", noon)])
    suspicious = db.insert_alert(Alert(noon, "app", first.window_start, 1.0, {}, ["a"]))
    db.attach_repeat(suspicious, repeat.window_start)
    expected = db.insert_alert(Alert(noon, "app", other.window_start, 1.0, {}, ["b"]))
    assert db.suspicious_windows() == []  # nothing labelled yet

    for alert_id, verdict, at in (
        (suspicious, Verdict.EXPECTED, 10.0),
        (suspicious, Verdict.SUSPICIOUS, 20.0),
        (expected, Verdict.EXPECTED, 30.0),
    ):
        db.insert_label(Label(alert_id, verdict, at))
        db.set_alert_status(alert_id, AlertStatus(verdict.value))

    # only the suspicious alert's own process and windows, each with its latest label's time
    assert db.suspicious_windows() == [(first, 20.0), (repeat, 20.0)]
