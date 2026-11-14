"""Native macOS notifications via ``osascript``"""

from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Callable
from typing import Any

from netwatch.models import Alert

log = logging.getLogger(__name__)


def applescript_quote(text: str) -> str:
    """escape ``text`` to use inside an AppleScript double-quoted string"""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


class MacOSNotificationSink:
    """shows one desktop notification per alert on macOS.

    Arguments:
        runner: Function used to run the command (``subprocess.run``)
        platform: Platform string; notifications only fire on ``"darwin"``
    """

    def __init__(
        self,
        runner: Callable[..., Any] = subprocess.run,
        platform: str = sys.platform,
    ) -> None:
        self._runner = runner
        self.enabled = platform == "darwin"

    def emit(self, alert: Alert) -> None:
        if not self.enabled:
            return
        title = f"netwatch: {alert.process_name}"
        body = alert.reasons[0] if alert.reasons else "anomalous network activity"
        if len(alert.reasons) > 1:
            body += f" (+{len(alert.reasons) - 1} more)"
        script = (
            f'display notification "{applescript_quote(body)}" '
            f'with title "{applescript_quote(title)}"'
        )
        try:
            self._runner(["osascript", "-e", script], check=False, capture_output=True, timeout=5)
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("could not show macOS notification: %s", exc)
