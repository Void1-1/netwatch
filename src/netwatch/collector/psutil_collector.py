"""Live connection collector with ``psutil``

Permissions:
    macOS:``psutil.net_connections()`` needs root to see all process's sockets
    The collector can fall back to asking each process the current user owns.

Only sockets with remote addresses are reported
"""

from __future__ import annotations

import logging
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import psutil

from netwatch.collector.base import CollectorPermissionError
from netwatch.models import ConnectionEvent

log = logging.getLogger(__name__)

# Name when PID cannot resolve. modelled as one bucket
UNKNOWN_PROCESS = "unknown"

# socket identity across snapshots (pid, local ip, local port, remote ip, remote port, socket type)
SocketKey = tuple[int | None, str, int, str, int, int]


@dataclass(frozen=True, slots=True)
class ProcessInfo:
    """Info about a process behind a PID"""

    name: str
    exe_path: str | None
    parent_name: str | None
    create_time: float | None


_UNKNOWN_INFO = ProcessInfo(UNKNOWN_PROCESS, None, None, None)


@dataclass(frozen=True, slots=True)
class RawSocket:
    """psutil-agnostic view of a socket"""

    pid: int | None
    type: int
    laddr: tuple[Any, ...]
    raddr: tuple[Any, ...]
    status: str


def _protocol_name(sock_type: int) -> str:
    return "tcp" if sock_type == socket.SOCK_STREAM else "udp"


def _safe_call(fn: Callable[[], Any]) -> Any:
    """call psutil accessor. map "not allowed / gone" to ``None``"""
    try:
        return fn()
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        return None


class PsutilCollector:
    """snapshot-diffing collector over ``psutil``

    Arguments:
        kind: psutil connection kind (``"inet"`` = IPv4 + IPv6, TCP + UDP)
        allow_partial: enable fall back to per-process enumeration
        clock: time source (for tests)
    """

    def __init__(
        self,
        kind: str = "inet",
        allow_partial: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.kind = kind
        self.allow_partial = allow_partial
        self._clock = clock
        self._previous: set[SocketKey] = set()
        self._snapshot_at: float | None = None
        # when the snapshot before the latest one was taken (None until there are two): the
        # connections the latest :meth:`collect` returned all opened after this time
        self.previous_snapshot_at: float | None = None
        self._process_cache: dict[int, ProcessInfo] = {}
        # True once fallen back to partial/per-process
        self.partial = False

    def _connected(self) -> list[tuple[SocketKey, RawSocket]]:
        """one snapshot: every socket with a remote address, with its identity"""
        connected: list[tuple[SocketKey, RawSocket]] = []
        for sock in self._snapshot():
            if not sock.raddr:
                continue  # listening or unconnected
            # psutil addresses are (ip, port) named tuples
            key: SocketKey = (
                sock.pid,
                str(sock.laddr[0]),
                int(sock.laddr[1]),
                str(sock.raddr[0]),
                int(sock.raddr[1]),
                sock.type,
            )
            connected.append((key, sock))
        return connected

    def prime(self) -> None:
        """note the connections open right now without reporting them

        Call once before a collection loop. Otherwise the first :meth:`collect` reports every
        socket that is already open as a new connection, all at one instant: a burst that never
        happened, and that looks like a spike to the detectors every time collection starts
        """
        self._snapshot_at = self._clock()
        self._previous = {key for key, _ in self._connected()}

    def collect(self) -> list[ConnectionEvent]:
        """return new connections since last call

        The first call reports every open connection, unless :meth:`prime` ran before it
        """
        now = self._clock()
        self.previous_snapshot_at, self._snapshot_at = self._snapshot_at, now
        current: set[SocketKey] = set()
        events: list[ConnectionEvent] = []

        for key, sock in self._connected():
            current.add(key)
            if key in self._previous:
                continue  # same connection as last snapshot

            info = self._process_info(sock.pid)
            events.append(
                ConnectionEvent(
                    timestamp=now,
                    pid=sock.pid,
                    process_name=info.name,
                    exe_path=info.exe_path,
                    parent_name=info.parent_name,
                    local_ip=key[1],
                    local_port=key[2],
                    remote_ip=key[3],
                    remote_port=key[4],
                    protocol=_protocol_name(sock.type),
                    status=sock.status,
                )
            )

        self._previous = current
        # Forget cached processes without a socket
        live_pids = {key[0] for key in current}
        self._process_cache = {
            pid: info for pid, info in self._process_cache.items() if pid in live_pids
        }
        return events

    # snapshots

    def _snapshot(self) -> list[RawSocket]:
        try:
            return [
                RawSocket(c.pid, c.type, tuple(c.laddr), tuple(c.raddr), c.status)
                for c in psutil.net_connections(kind=self.kind)
            ]
        except psutil.AccessDenied as exc:
            if not self.allow_partial:
                raise CollectorPermissionError(
                    "listing all connections needs elevated privileges. `sudo netwatch collect`"
                ) from exc
            if not self.partial:
                log.warning(
                    "No permission to list all process connections. "
                    "Falling back to processes owned by the current user. "
                    "Run with sudo for full coverage."
                )
                self.partial = True
            return self._per_process_snapshot()

    def _per_process_snapshot(self) -> list[RawSocket]:
        sockets: list[RawSocket] = []
        readable = 0
        for proc in psutil.process_iter():
            try:
                conns = proc.net_connections(kind=self.kind)
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            readable += 1
            sockets.extend(
                RawSocket(proc.pid, c.type, tuple(c.laddr), tuple(c.raddr), c.status) for c in conns
            )
        if readable == 0:
            raise CollectorPermissionError(
                "could not read connections for processes. `sudo netwatch collect`"
            )
        return sockets

    # attribution

    def _process_info(self, pid: int | None) -> ProcessInfo:
        """Resolve PID to name/exe/parent"""
        if pid is None:
            return _UNKNOWN_INFO
        try:
            proc = psutil.Process(pid)
            create_time = proc.create_time()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return self._process_cache.get(pid, _UNKNOWN_INFO)

        cached = self._process_cache.get(pid)
        if cached is not None and cached.create_time == create_time:
            return cached

        with proc.oneshot():
            name = _safe_call(proc.name) or UNKNOWN_PROCESS
            exe_path = _safe_call(proc.exe) or None
            parent = _safe_call(proc.parent)
            parent_name = _safe_call(parent.name) if parent is not None else None
        info = ProcessInfo(name, exe_path, parent_name, create_time)
        self._process_cache[pid] = info
        return info
