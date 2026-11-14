"""Alert delivery

Sinks are kept dumb: they format and deliver an :class:`~netwatch.models.Alert`

:mod:`netwatch.alerts.dedup`, comes before delivery
"""

from __future__ import annotations

from rich.console import Console

from netwatch.alerts.base import AlertSink, CollectingSink
from netwatch.alerts.console import ConsoleAlertSink
from netwatch.alerts.macos_notify import MacOSNotificationSink
from netwatch.config import AlertsConfig

__all__ = [
    "AlertSink",
    "CollectingSink",
    "ConsoleAlertSink",
    "MacOSNotificationSink",
    "build_sinks",
]


def build_sinks(config: AlertsConfig, console: Console | None = None) -> list[AlertSink]:
    """sinks enabled in ``alerts:`` config block"""
    sinks: list[AlertSink] = []
    if config.console:
        sinks.append(ConsoleAlertSink(console))
    if config.macos_notifications:
        sinks.append(MacOSNotificationSink())
    return sinks
