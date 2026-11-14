"""Collector interface

A collector reports the connections that are new since it was last asked, and the process behind
each one. Anything with a matching ``collect()`` fits, so psutil can later be swapped for packet
capture, eBPF or recorded data without changing the rest of the pipeline
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from netwatch.models import ConnectionEvent


class CollectorPermissionError(PermissionError):
    """OS refused to list connections"""


@runtime_checkable
class Collector(Protocol):
    """Anything that can snapshot of current connections"""

    def collect(self) -> list[ConnectionEvent]:
        """Return new connections since previous call"""
        ...
