"""common detector interface for detection layers"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar, Self

import numpy as np

from netwatch.models import FeatureWindow


class NotFittedError(RuntimeError):
    """detector was asked to score or explain before being fit"""


class Detector(ABC):
    """abstract base class for anomaly detectors over feature windows"""

    # identifier, used as key in alerts and config
    name: ClassVar[str]
    explains_destinations: ClassVar[bool] = False

    def __init__(self) -> None:
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError(f"{self.name} detector must be fit() before use")

    @abstractmethod
    def fit(self, windows: Sequence[FeatureWindow]) -> Self:
        """learn 'normal' behaviour from (poisoning-filtered) windows"""

    @abstractmethod
    def score(self, windows: Sequence[FeatureWindow]) -> np.ndarray:
        """anomaly score per window ``(len(windows),)``. higher is worse"""

    @abstractmethod
    def explain(self, window: FeatureWindow) -> list[str]:
        """reasons this window is unusual, most important first"""
