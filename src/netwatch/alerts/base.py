"""alert sink interface and in-memory sink for tests, demo"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from netwatch.models import Alert


@runtime_checkable
class AlertSink(Protocol):
    def emit(self, alert: Alert) -> None:
        """deliver one alert"""
        ...


class CollectingSink:
    """keep alerts in a list, not shown"""

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def emit(self, alert: Alert) -> None:
        self.alerts.append(alert)
