"""Rich-format alerts with ``netwatch watch``"""

from __future__ import annotations

from datetime import datetime

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from netwatch.models import Alert


def format_time(timestamp: float) -> str:
    """Local ``YYYY-MM-DD HH:MM`` for display"""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def format_detector_scores(scores: dict[str, float]) -> str:
    """``"novelty 1.000, isolation_forest 0.982"``"""
    ordered = sorted(scores.items(), key=lambda item: -item[1])
    return ", ".join(f"{name} {value:.3f}" for name, value in ordered)


def render_alert(alert: Alert) -> Panel:
    """Rich panel for an alert"""
    title = f"netwatch alert #{alert.id}" if alert.id is not None else "netwatch alert"
    header = Text.assemble(
        (alert.process_name, "bold"),
        f"  window {format_time(alert.window_start)}",
        f"  score {alert.combined_score:.3f}",
    )
    reasons = Text("\n".join(f"- {reason}" for reason in alert.reasons))
    detail = Text(f"detectors: {format_detector_scores(alert.per_detector_scores)}", style="dim")
    return Panel(Group(header, reasons, detail), title=title, border_style="red", expand=False)


class ConsoleAlertSink:
    """prints each alert as a Rich panel"""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def emit(self, alert: Alert) -> None:
        self.console.print(render_alert(alert))
