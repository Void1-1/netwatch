"""psutil collector, run entirely on fakes: no root, no real sockets"""

from __future__ import annotations

import contextlib
import socket
from collections import namedtuple
from typing import Any

import psutil
import pytest

from netwatch.collector.base import Collector, CollectorPermissionError
from netwatch.collector.psutil_collector import UNKNOWN_PROCESS, PsutilCollector
from netwatch.pipeline import store_new_connections
from netwatch.storage.db import Database

Addr = namedtuple("Addr", "ip port")
SConn = namedtuple("SConn", "fd family type laddr raddr status pid")
PConn = namedtuple("PConn", "fd family type laddr raddr status")

TCP, UDP = socket.SOCK_STREAM, socket.SOCK_DGRAM


def conn(pid: int | None, rport: int, rip: str = "192.0.2.1", type_: int = TCP) -> SConn:
    raddr = Addr(rip, rport) if rport else ()
    status = "ESTABLISHED" if rport else "LISTEN"
    return SConn(3, socket.AF_INET, type_, Addr("192.168.1.23", 50000 + rport), raddr, status, pid)


class FakeOS:
    """a tiny process table plus a socket list, standing in for psutil"""

    def __init__(self) -> None:
        self.processes: dict[int, dict[str, Any]] = {
            1: {"name": "launchd", "exe": "/sbin/launchd", "ppid": None, "created": 1.0},
            100: {"name": "browser", "exe": "/Apps/browser", "ppid": 1, "created": 10.0},
            200: {"name": "locked", "exe": None, "ppid": 1, "created": 20.0},
        }
        self.sockets: list[SConn] = []
        self.deny_system_wide = False

    def net_connections(self, kind: str = "inet") -> list[SConn]:
        if self.deny_system_wide:
            raise psutil.AccessDenied()
        return list(self.sockets)

    def make_process_class(self) -> type:
        fake_os = self

        class FakeProcess:
            def __init__(self, pid: int) -> None:
                if pid not in fake_os.processes:
                    raise psutil.NoSuchProcess(pid)
                self.pid = pid
                self.info = fake_os.processes[pid]

            def create_time(self) -> float:
                return self.info["created"]

            def oneshot(self) -> contextlib.AbstractContextManager[None]:
                return contextlib.nullcontext()

            def name(self) -> str:
                return self.info["name"]

            def exe(self) -> str:
                if self.info["exe"] is None:
                    raise psutil.AccessDenied(self.pid)
                return self.info["exe"]

            def parent(self) -> FakeProcess | None:
                ppid = self.info["ppid"]
                return FakeProcess(ppid) if ppid else None

            def net_connections(self, kind: str = "inet") -> list[PConn]:
                return [PConn(*s[:6]) for s in fake_os.sockets if s.pid == self.pid]

        return FakeProcess

    def process_iter(self) -> list[Any]:
        cls = self.make_process_class()
        return [cls(pid) for pid in self.processes]


@pytest.fixture
def fake_os(monkeypatch: pytest.MonkeyPatch) -> FakeOS:
    fake = FakeOS()
    monkeypatch.setattr(psutil, "net_connections", fake.net_connections)
    monkeypatch.setattr(psutil, "Process", fake.make_process_class())
    monkeypatch.setattr(psutil, "process_iter", fake.process_iter)
    return fake


def test_collector_satisfies_the_protocol() -> None:
    assert isinstance(PsutilCollector(), Collector)


def test_connections_are_attributed_to_their_process(fake_os: FakeOS) -> None:
    fake_os.sockets = [conn(100, 443), conn(100, 0), conn(200, 53, type_=UDP)]
    events = PsutilCollector(clock=lambda: 123.0).collect()

    assert len(events) == 2  # the listening socket has no peer and is skipped
    browser, locked = sorted(events, key=lambda e: e.process_name)
    assert browser.process_name == "browser"
    assert browser.exe_path == "/Apps/browser"
    assert browser.parent_name == "launchd"
    assert (browser.remote_ip, browser.remote_port, browser.protocol) == ("192.0.2.1", 443, "tcp")
    assert browser.timestamp == 123.0
    # exe() denied: still attributed by name, path unknown
    assert (locked.process_name, locked.exe_path, locked.protocol) == ("locked", None, "udp")


def test_each_connection_is_reported_once(fake_os: FakeOS) -> None:
    collector = PsutilCollector()
    fake_os.sockets = [conn(100, 443)]
    assert len(collector.collect()) == 1
    assert collector.collect() == []  # same socket, already reported
    fake_os.sockets = [conn(100, 443), conn(100, 80)]
    assert [e.remote_port for e in collector.collect()] == [80]


def test_reused_pids_are_re_resolved(fake_os: FakeOS) -> None:
    collector = PsutilCollector()
    fake_os.sockets = [conn(100, 443)]
    assert collector.collect()[0].process_name == "browser"
    fake_os.processes[100] = {"name": "imposter", "exe": "/tmp/x", "ppid": 1, "created": 99.0}
    fake_os.sockets = [conn(100, 444)]
    assert collector.collect()[0].process_name == "imposter"


def test_vanished_processes_are_unknown(fake_os: FakeOS) -> None:
    fake_os.sockets = [conn(4242, 443), conn(None, 443)]
    names = {e.process_name for e in PsutilCollector().collect()}
    assert names == {UNKNOWN_PROCESS}


def test_falls_back_to_per_process_view_without_root(fake_os: FakeOS) -> None:
    fake_os.deny_system_wide = True
    fake_os.sockets = [conn(100, 443)]
    collector = PsutilCollector(allow_partial=True)
    events = collector.collect()
    assert collector.partial is True
    assert [(e.pid, e.process_name) for e in events] == [(100, "browser")]


def test_strict_mode_raises_a_helpful_permission_error(fake_os: FakeOS) -> None:
    fake_os.deny_system_wide = True
    with pytest.raises(CollectorPermissionError, match="sudo"):
        PsutilCollector(allow_partial=False).collect()


def test_priming_keeps_connections_that_were_already_open_out(fake_os: FakeOS) -> None:
    """a collection loop must not start with a burst of every socket that is already open"""
    collector = PsutilCollector()
    fake_os.sockets = [conn(100, 443), conn(100, 80), conn(200, 53, type_=UDP)]
    collector.prime()
    assert collector.collect() == []
    fake_os.sockets.append(conn(100, 8443))
    assert [e.remote_port for e in collector.collect()] == [8443]


# two collectors on one database (`collect` left running next to `watch`)


def test_two_collectors_store_a_connection_once(fake_os: FakeOS) -> None:
    """both see the same sockets, and a window must not count them twice"""
    now = [0.0]
    first, second = (PsutilCollector(clock=lambda: now[0]) for _ in range(2))

    def snapshot(collector: PsutilCollector, at: float) -> int:
        now[0] = at
        return store_new_connections(db, collector, interval=5.0)

    with Database.open(":memory:") as db:
        first.prime()
        second.prime()

        fake_os.sockets = [conn(100, 443)]
        assert (snapshot(first, 5.0), snapshot(second, 6.0)) == (1, 0)

        # whichever collector gets there first stores it
        fake_os.sockets.append(conn(100, 80))
        assert (snapshot(second, 11.0), snapshot(first, 15.0)) == (1, 0)

        # closed and opened again: a second connection, stored once more
        fake_os.sockets = [conn(100, 80)]
        assert (snapshot(first, 20.0), snapshot(second, 21.0)) == (0, 0)
        fake_os.sockets.append(conn(100, 443))
        assert (snapshot(first, 25.0), snapshot(second, 26.0)) == (1, 0)

        assert [(e.timestamp, e.remote_port) for e in db.query_events()] == [
            (5.0, 443),
            (11.0, 80),
            (25.0, 443),
        ]


def test_one_collector_never_loses_its_own_connections(fake_os: FakeOS) -> None:
    """a socket that closes and reopens with the same addresses is a new connection each time"""
    now = [0.0]
    collector = PsutilCollector(clock=lambda: now[0])
    with Database.open(":memory:") as db:
        collector.prime()
        for step in range(1, 9):
            now[0] = step * 5.0
            fake_os.sockets = [conn(100, 443)] if step % 2 else []
            store_new_connections(db, collector, interval=5.0)
        assert [e.timestamp for e in db.query_events()] == [5.0, 15.0, 25.0, 35.0]


def test_an_unprimed_snapshot_stores_everything_that_is_open(fake_os: FakeOS) -> None:
    """`collect --once` records what is open now, whatever a running collector stored before"""
    fake_os.sockets = [conn(100, 443)]
    with Database.open(":memory:") as db:
        running = PsutilCollector(clock=lambda: 5.0)
        assert store_new_connections(db, running, interval=5.0) == 1
        assert running.previous_snapshot_at is None
        assert store_new_connections(db, PsutilCollector(clock=lambda: 6.0), interval=5.0) == 1
