"""Layer 3: autoencoder detector (planned, not implemented)

plan:
    * train a small autoencoder per process on the same feature windows the isolation forest
      uses (later also byte counts and DNS names)
    * score = reconstruction error: a window the model cannot rebuild does not look like
      anything in training
    * explain with the features the model got most wrong ("expected ~3 destinations, saw 40")
    * keep it small (scikit-learn's ``MLPRegressor`` or a few hundred lines of numpy), so
      netwatch stays CPU-only with few dependencies

Why a separate layer: a forest splits on one feature at a time. An autoencoder can learn how
features go together (a process that is only busy when it talks to one particular host)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np

from netwatch.detectors.base import Detector
from netwatch.models import FeatureWindow

_PLANNED = "the autoencoder detector (Layer 3) is planned but not implemented yet"


class AutoencoderDetector(Detector):
    """Placeholder for the Layer 3 detector; see the module docstring"""

    name = "autoencoder"

    def fit(self, windows: Sequence[FeatureWindow]) -> Self:
        raise NotImplementedError(_PLANNED)

    def score(self, windows: Sequence[FeatureWindow]) -> np.ndarray:
        raise NotImplementedError(_PLANNED)

    def explain(self, window: FeatureWindow) -> list[str]:
        raise NotImplementedError(_PLANNED)
