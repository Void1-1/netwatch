"""Alert deduplication

An alerting window *repeats* an open alert for the same process when it tells the user nothing
that alert has not already told them:

1. its new destinations (IPs and ports) are a subset of the new destinations of the window that
   raised the alert
2. every reason from the detectors that look beyond destinations (baseline, isolation forest) is
   already covered by one of the alert's reasons

Covered means the same kind of reason at no more than :data:`ESCALATION_FACTOR` times the
amounts the alert reported. Reasons quote counts, so comparing them word for word would make
every window of a lasting spike a new alert: 47 connections, then 52 an hour later, is the same
news. But a spike that grows from 47 connections to 4,700 is not, and raises a new alert even
though the wording is otherwise identical.

A repeat is attached to the first alert instead of raising a new one. Labelling the alert labels
every window attached to it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from netwatch.models import Alert, AlertStatus, FeatureWindow

# an amount more than this many times the one an open alert reported is news again
ESCALATION_FACTOR = 2.0

# A number standing on its own, as detectors quote counts and ratios ("47", "2.5", "12x"). Digits
# inside a path, a clock time or an address are left alone ("/tmp/a1", "03:00-04:00", "192.0.2.7"):
# those name something and must match exactly
_AMOUNT = re.compile(r"(?<![\w./:-])(\d+(?:\.\d+)?)(x?)(?![\w/:-]|\.\d)")


def reason_kind(reason: str) -> str:
    """``reason`` with its amounts blanked: what it says, without how much

    ``"contacted 150 distinct remote ports (typical: 1) - possible port scan"`` and the same
    sentence with 90 ports are one kind. The wording still tells kinds apart: the sentence
    without "possible port scan" is another kind
    """
    return _AMOUNT.sub(r"#\2", reason)


def reason_amounts(reason: str) -> tuple[float, ...]:
    """the amounts :func:`reason_kind` blanks, in order"""
    return tuple(float(match.group(1)) for match in _AMOUNT.finditer(reason))


def covers(known: str, later: str) -> bool:
    """if ``known``, a reason an alert already gave, says all that ``later`` says

    The same kind of reason, with no amount in ``later`` more than
    :data:`ESCALATION_FACTOR` times its counterpart in ``known``. Smaller is always covered
    """
    if reason_kind(known) != reason_kind(later):
        return False
    return all(
        after <= before * ESCALATION_FACTOR
        for before, after in zip(reason_amounts(known), reason_amounts(later), strict=True)
    )


@dataclass(frozen=True, slots=True)
class OpenAlert:
    """alert that may absorb repeats, with the window that raised it"""

    alert: Alert
    window: FeatureWindow


def is_repeat(window: FeatureWindow, reasons: Iterable[str], candidate: OpenAlert) -> bool:
    """if alerting ``window`` only repeats the cause of ``candidate``

    Arguments:
        window: the window that crossed the alert threshold
        reasons: its reasons from detectors that look beyond destinations
        candidate: alert raised earlier, with its window
    """
    alert, first = candidate.alert, candidate.window
    return (
        alert.status == AlertStatus.OPEN
        and alert.process_name == window.process_name
        and set(window.new_remote_ips) <= set(first.new_remote_ips)
        and set(window.new_remote_ports) <= set(first.new_remote_ports)
        and all(any(covers(known, reason) for known in alert.reasons) for reason in reasons)
    )


def find_repeated(
    window: FeatureWindow, reasons: Iterable[str], candidates: Iterable[OpenAlert]
) -> Alert | None:
    """first of ``candidates`` that ``window`` repeats, or ``None``"""
    reasons = list(reasons)
    return next((c.alert for c in candidates if is_repeat(window, reasons, c)), None)
