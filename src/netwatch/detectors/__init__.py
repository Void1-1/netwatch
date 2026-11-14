"""Anomaly detectors, all behind the same interface

1. simple rules, compared against the process's own history:

   * :class:`NoveltyDetector`: destinations the process has not used before
   * :class:`BaselineDetector`: more connections, more destinations or steadier timing than usual

2. :class:`IsolationForestDetector`: unusual *combinations* of window features
3. :class:`AutoencoderDetector`: planned
4. :class:`SequenceDetector`: planned

:func:`build_detectors` creates the detectors that are enabled in the ``detectors:`` config
"""

from __future__ import annotations

from netwatch.config import DetectorsConfig
from netwatch.detectors.autoencoder import AutoencoderDetector
from netwatch.detectors.base import Detector, NotFittedError
from netwatch.detectors.baseline import BaselineDetector
from netwatch.detectors.isolation_forest import IsolationForestDetector
from netwatch.detectors.novelty_detector import NoveltyDetector
from netwatch.detectors.sequence import SequenceDetector

__all__ = [
    "AutoencoderDetector",
    "BaselineDetector",
    "Detector",
    "IsolationForestDetector",
    "NotFittedError",
    "NoveltyDetector",
    "SequenceDetector",
    "build_detectors",
    "detector_weights",
]


def build_detectors(config: DetectorsConfig) -> list[Detector]:
    """instantiate enabled, implemented detector from config"""
    detectors: list[Detector] = []
    if config.novelty.enabled:
        detectors.append(
            NoveltyDetector(unseen_process_surprise=config.novelty.unseen_process_surprise)
        )
    if config.baseline.enabled:
        detectors.append(
            BaselineDetector(min_windows_per_process=config.baseline.min_windows_per_process)
        )
    if config.isolation_forest.enabled:
        forest = config.isolation_forest
        detectors.append(
            IsolationForestDetector(
                n_estimators=forest.n_estimators,
                min_windows_per_process=forest.min_windows_per_process,
                random_state=forest.random_state,
            )
        )
    return detectors


def detector_weights(config: DetectorsConfig) -> dict[str, float]:
    """combiner weight per detector"""
    return {
        NoveltyDetector.name: config.novelty.weight,
        BaselineDetector.name: config.baseline.weight,
        IsolationForestDetector.name: config.isolation_forest.weight,
        AutoencoderDetector.name: config.autoencoder.weight,
        SequenceDetector.name: config.sequence.weight,
    }
