"""Small statistics helpers (FINAL_PLAN §10), numpy only.

- paired_bootstrap_ci: CI for the mean of paired differences (e.g. rounded
  minus float score on the same seeds).
- clopper_pearson: exact binomial CI for a success rate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float

    def contains(self, x: float) -> bool:
        return self.low <= x <= self.high

    def to_json(self) -> dict:
        return {"estimate": self.estimate, "low": self.low, "high": self.high}


def paired_bootstrap_ci(
    a: Sequence[float], b: Sequence[float], *, n_boot: int = 10_000, level: float = 0.95, seed: int = 0
) -> Interval:
    """Percentile bootstrap CI for mean(a - b) over paired samples (deterministic for a given seed)."""
    x = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("need two equal-length, non-empty 1-D samples")
    rng = np.random.default_rng([seed, 0xB007])
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return Interval(float(x.mean()), float(lo), float(hi))


def _binom_logpmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p)


def _binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k)."""
    return min(1.0, sum(math.exp(_binom_logpmf(i, n, p)) for i in range(k, n + 1)))


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k)."""
    return min(1.0, sum(math.exp(_binom_logpmf(i, n, p)) for i in range(0, k + 1)))


def _bisect(f, lo: float, hi: float, iters: int = 100) -> float:
    """Root of a monotone increasing f on [lo, hi]."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(k: int, n: int, level: float = 0.95) -> Interval:
    """Exact (Clopper–Pearson) CI for a binomial proportion k/n."""
    if not 0 <= k <= n or n <= 0:
        raise ValueError(f"need 0 <= k <= n and n > 0, got k={k}, n={n}")
    alpha = 1.0 - level
    low = 0.0 if k == 0 else _bisect(lambda p: _binom_sf(k, n, p) - alpha / 2, 0.0, 1.0)
    high = 1.0 if k == n else _bisect(lambda p: (1.0 - _binom_cdf(k, n, p)) - (1.0 - alpha / 2), 0.0, 1.0)
    return Interval(k / n, low, high)
