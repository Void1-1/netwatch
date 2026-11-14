"""Layer 1: novelty detector

The windowing stage counts new remote IPs and ports (see :mod:`netwatch.features.novelty`)

This detector learns each process's base rate of new IPs and new ports per window, and scores a
window by how unlikely its novelty count is under that rate. New-destination arrivals are
modelled as Poisson, and the score is the surprise ``-log10 P(X >= k)``. IP and port surprises
are added.

Rates are smoothed toward the machine-wide rate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np

from netwatch.detectors.base import Detector
from netwatch.features.encoders import describe_port
from netwatch.models import FeatureWindow

# Floor on any learned rate
MIN_RATE = 1e-4
# How many reasons of each kind (IPs, ports) to list before summarising
MAX_LISTED = 3


def poisson_tail_surprise(k: int, rate: float) -> float:
    """``-log10 P(X >= k)`` for ``X ~ Poisson(rate)``; 0 when ``k <= 0``"""
    if k <= 0:
        return 0.0
    rate = max(rate, 1e-12)
    log_rate = math.log(rate)

    def log_pmf(i: int) -> float:
        return i * log_rate - rate - math.lgamma(i + 1)

    if k <= rate:
        cdf = sum(math.exp(log_pmf(i)) for i in range(k))
        return -math.log10(max(1.0 - cdf, 1e-300))

    terms = [log_pmf(k)]
    i = k + 1
    # Stop once terms are e^-40 below the first (negligible), with a hard cap
    while terms[-1] > terms[0] - 40 and i < k + 10_000:
        terms.append(log_pmf(i))
        i += 1
    peak = max(terms)
    log_tail = peak + math.log(sum(math.exp(t - peak) for t in terms))
    return -log_tail / math.log(10)


@dataclass(frozen=True, slots=True)
class NoveltyRates:
    """A process's learned base rates of new destinations per window"""

    windows: int
    ip_rate: float
    port_rate: float


class NoveltyDetector(Detector):
    """Scores windows by Poisson surprise of new IPs & ports

    Arguments:
        unseen_process_surprise: Added to the score of any process that had no windows in
            training
        prior_windows: Strength of the pull toward the machine-wide rate, in pseudo-windows
    """

    name = "novelty"
    # Its reasons name the process (if unseen) and its new destinations, and count them against
    # a rate fixed at training time
    explains_destinations = True

    def __init__(self, unseen_process_surprise: float = 4.0, prior_windows: float = 5.0) -> None:
        super().__init__()
        self.unseen_process_surprise = unseen_process_surprise
        self.prior_windows = prior_windows
        self.rates: dict[str, NoveltyRates] = {}
        self.global_rates = NoveltyRates(0, MIN_RATE, MIN_RATE)

    def fit(self, windows: Sequence[FeatureWindow]) -> Self:
        if not windows:
            raise ValueError("cannot fit the novelty detector on zero windows")

        totals: dict[str, list[int]] = {}
        for w in windows:
            counts = totals.setdefault(w.process_name, [0, 0, 0])
            counts[0] += 1
            counts[1] += w.features.new_ip_count
            counts[2] += w.features.new_port_count

        n_all = len(windows)
        global_ip = max(sum(c[1] for c in totals.values()) / n_all, MIN_RATE)
        global_port = max(sum(c[2] for c in totals.values()) / n_all, MIN_RATE)
        self.global_rates = NoveltyRates(n_all, global_ip, global_port)

        # Empirical-Bayes smoothing: each process starts with `prior_windows` imaginary windows
        # at the machine rate
        prior = self.prior_windows
        self.rates = {
            process: NoveltyRates(
                windows=n,
                ip_rate=max((ips + prior * global_ip) / (n + prior), MIN_RATE),
                port_rate=max((ports + prior * global_port) / (n + prior), MIN_RATE),
            )
            for process, (n, ips, ports) in totals.items()
        }
        self._fitted = True
        return self

    def _rates_for(self, process_name: str) -> tuple[NoveltyRates, bool]:
        """Rates to judge ``process_name`` by, and if it was in training"""
        rates = self.rates.get(process_name)
        return (rates, True) if rates is not None else (self.global_rates, False)

    def score_window(self, window: FeatureWindow) -> float:
        rates, seen = self._rates_for(window.process_name)
        surprise = poisson_tail_surprise(
            window.features.new_ip_count, rates.ip_rate
        ) + poisson_tail_surprise(window.features.new_port_count, rates.port_rate)
        if not seen:
            surprise += self.unseen_process_surprise
        return surprise

    def score(self, windows: Sequence[FeatureWindow]) -> np.ndarray:
        self._require_fitted()
        return np.array([self.score_window(w) for w in windows], dtype=float)

    def explain(self, window: FeatureWindow) -> list[str]:
        self._require_fitted()
        rates, seen = self._rates_for(window.process_name)
        reasons: list[str] = []
        if not seen:
            reasons.append(f"process '{window.process_name}' was never seen during training")

        new_ips = window.new_remote_ips
        if len(new_ips) <= MAX_LISTED:
            reasons.extend(f"first connection from this process to {ip}" for ip in new_ips)
        else:
            examples = ", ".join(new_ips[:MAX_LISTED])
            reasons.append(
                f"first connections to {len(new_ips)} new destinations (e.g. {examples})"
            )

        new_ports = window.new_remote_ports
        if len(new_ports) <= MAX_LISTED:
            reasons.extend(
                f"first connection from this process on port {describe_port(p)}" for p in new_ports
            )
        else:
            examples = ", ".join(str(p) for p in new_ports[:MAX_LISTED])
            reasons.append(f"first connections on {len(new_ports)} new ports (e.g. {examples})")

        # Put the counts in context when there are several at once
        if seen and len(new_ips) >= 2:
            reasons.append(
                f"{len(new_ips)} new destinations in one window "
                f"(this process averages {rates.ip_rate:.2f} per window)"
            )
        return reasons
