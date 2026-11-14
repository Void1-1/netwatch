"""Alert sinks: terminal output and macOS notifications (never actually shown)"""

from __future__ import annotations

import io
import subprocess
from typing import Any

from rich.console import Console

from netwatch.alerts import build_sinks
from netwatch.alerts.base import CollectingSink
from netwatch.alerts.console import ConsoleAlertSink, render_alert
from netwatch.alerts.macos_notify import MacOSNotificationSink, applescript_quote
from netwatch.config import AlertsConfig
from netwatch.models import Alert


def sample_alert(reasons: list[str] | None = None) -> Alert:
    return Alert(
        timestamp=1_800_000_300.0,
        process_name="chatapp",
        window_start=1_800_000_000.0,
        combined_score=0.9995,
        per_detector_scores={"novelty": 0.98, "baseline": 1.0},
        reasons=reasons or ["connections are highly periodic (possible beaconing)"],
        id=7,
    )


def test_console_rendering_leads_with_reasons() -> None:
    console = Console(record=True, width=120)
    console.print(render_alert(sample_alert()))
    text = console.export_text()
    assert "netwatch alert #7" in text
    assert "chatapp" in text
    assert "- connections are highly periodic (possible beaconing)" in text
    assert "baseline 1.000, novelty 0.980" in text  # strongest detector first


def test_console_sink_prints() -> None:
    buffer = io.StringIO()
    ConsoleAlertSink(Console(file=buffer, width=120)).emit(sample_alert())
    assert "possible beaconing" in buffer.getvalue()


class Recorder:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.error = error

    def __call__(self, argv: list[str], **kwargs: Any) -> None:
        self.calls.append((argv, kwargs))
        if self.error:
            raise self.error


def test_macos_notifications_are_a_no_op_elsewhere() -> None:
    runner = Recorder()
    MacOSNotificationSink(runner=runner, platform="linux").emit(sample_alert())
    assert runner.calls == []


def test_macos_notifications_use_argv_and_escape_untrusted_text() -> None:
    runner = Recorder()
    alert = sample_alert(['evil "name" \\ path', "second reason"])
    MacOSNotificationSink(runner=runner, platform="darwin").emit(alert)
    ((argv, kwargs),) = runner.calls
    assert argv[:2] == ["osascript", "-e"]
    assert 'evil \\"name\\" \\\\ path (+1 more)' in argv[2]
    assert "shell" not in kwargs  # never through a shell
    assert applescript_quote('a"b\nc') == 'a\\"b c'


def test_notification_failures_never_raise() -> None:
    sink = MacOSNotificationSink(runner=Recorder(OSError("no osascript")), platform="darwin")
    sink.emit(sample_alert())
    sink = MacOSNotificationSink(
        runner=Recorder(subprocess.TimeoutExpired("osascript", 5)), platform="darwin"
    )
    sink.emit(sample_alert())


def test_build_sinks_and_collecting_sink() -> None:
    sinks = build_sinks(AlertsConfig(console=True, macos_notifications=True))
    assert [type(s).__name__ for s in sinks] == ["ConsoleAlertSink", "MacOSNotificationSink"]
    assert build_sinks(AlertsConfig(console=False)) == []
    collector = CollectingSink()
    collector.emit(sample_alert())
    assert len(collector.alerts) == 1
