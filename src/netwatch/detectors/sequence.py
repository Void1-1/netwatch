"""Layer 4: sequence detector (planned, not implemented)

Every method raises ``NotImplementedError`` and config validation refuses to enable it.

plan:
    * window features ignore *order*, but many processes connect in a habitual order (a mail
      client: its IMAP host, then its SMTP host). Malware inside such a process breaks the habit
    * model each process's destinations as a sequence, starting with a simple Markov model of
      "which destination comes next" (cheap and easy to explain)
    * score = average surprise of the window's transitions. Explain with the least likely ones
      ("after imap.example.net this process has never gone to 203.0.113.7")
    * also the place to catch slow beacons: regular timing across many windows, which the
      per-window ``interval_variance`` feature cannot see

The model needs the order of events, so windowing will store a short destination sequence in
each window's context. ``score(windows)`` stays the same
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np

from netwatch.detectors.base import Detector
from netwatch.models import FeatureWindow

_PLANNED = "the sequence detector (Layer 4) is planned but not implemented yet"


class SequenceDetector(Detector):
    """Placeholder for the Layer 4 detector; see module docstring"""

    name = "sequence"

    def fit(self, windows: Sequence[FeatureWindow]) -> Self:
        raise NotImplementedError(_PLANNED)

    def score(self, windows: Sequence[FeatureWindow]) -> np.ndarray:
        raise NotImplementedError(_PLANNED)

    def explain(self, window: FeatureWindow) -> list[str]:
        raise NotImplementedError(_PLANNED)
