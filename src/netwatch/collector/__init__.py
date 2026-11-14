"""Collection

Collectors turn sockets into :class:`~netwatch.models.ConnectionEvent` records, each attributed
to the process that owns the socket
"""

from netwatch.collector.base import Collector, CollectorPermissionError

__all__ = ["Collector", "CollectorPermissionError"]
