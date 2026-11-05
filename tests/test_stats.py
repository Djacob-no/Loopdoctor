"""Delivery 3: statistics helpers."""

from __future__ import annotations

import numpy as np
import pytest

from weightshift.stats import clopper_pearson, paired_bootstrap_ci


@pytest.mark.parametrize(
    "k, n, low, high",
    [
        (0, 1024, 0.0, 0.0035960),  # 1 - 0.025**(1/1024)
        (1024, 1024, 0.9964040, 1.0),
        (5, 10, 0.1870860, 0.8129140),  # textbook values
        (1, 256, 0.0000989, 0.0215709),  # checked: P(X<=1 | p=high) = 0.025
    ],
)
def test_clopper_pearson_known_values(k, n, low, high):
    ci = clopper_pearson(k, n)
    assert ci.estimate == k / n
    assert ci.low == pytest.approx(low, abs=2e-6) and ci.high == pytest.approx(high, abs=2e-6)


def test_clopper_pearson_rejects_bad_input():
    with pytest.raises(ValueError):
        clopper_pearson(5, 3)


def test_paired_bootstrap():
    rng = np.random.default_rng(0)
    a = rng.normal(10, 1, 500)
    same = paired_bootstrap_ci(a, a)
    assert same.estimate == 0 and same.low == 0 and same.high == 0
    shifted = paired_bootstrap_ci(a + 1.0, a)
    assert shifted.estimate == pytest.approx(1.0) and shifted.contains(1.0) and not shifted.contains(0.0)
    noisy = paired_bootstrap_ci(a + rng.normal(0, 1, 500), a)
    assert noisy.contains(0.0)
    assert paired_bootstrap_ci(a + 1, a) == paired_bootstrap_ci(a + 1, a)  # deterministic
    with pytest.raises(ValueError):
        paired_bootstrap_ci([], [])
