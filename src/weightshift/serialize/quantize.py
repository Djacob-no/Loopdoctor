"""Per-tensor power-of-ten quantization (FINAL_PLAN §7.1, checked against Exp 4).

Each tensor is shown as integers times one scale `10^k`, with
`k = floor(log10(max|w|)) - 2`, so its largest entry has 3 significant digits.

Rules confirmed by the Exp 4 golden test (DELIVERY_2_ACCEPTANCE §8):
- rounding is done in float64 on `w / 10.0**k` with round-half-to-even
  (`np.rint`); on Exp 4's weights this gives byte-identical integers;
- dequantizing is `int * 10.0**k` in float64, which reproduces every
  unchanged entry of Exp 4's `edited_actor.npz` exactly.

An all-zero tensor uses k = 0. If the largest entry rounds up to 1000 (for
example 999.6 x 10^k), that one integer has 4 digits; the scale is not bumped,
so the rule stays the simple published one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuantizedTensor:
    name: str
    ints: np.ndarray  # int64, same shape as the tensor
    k: int  # scale = 10**k

    @property
    def scale(self) -> float:
        return 10.0**self.k

    def dequantize(self) -> np.ndarray:
        return self.ints.astype(np.float64) * 10.0**self.k


def scale_exponent(w: np.ndarray) -> int:
    """k such that the tensor's largest |entry| shows with 3 significant digits."""
    a = np.asarray(w, dtype=np.float64)
    m = float(np.max(np.abs(a))) if a.size else 0.0
    if not math.isfinite(m):
        raise ValueError("cannot quantize a tensor with NaN or infinite values")
    if m == 0.0:
        return 0
    e = math.floor(math.log10(m))
    # guard against log10 rounding at exact powers of ten
    if 10.0**e > m:
        e -= 1
    elif 10.0 ** (e + 1) <= m:
        e += 1
    return e - 2


def quantize_tensor(name: str, w: np.ndarray) -> QuantizedTensor:
    a = np.asarray(w, dtype=np.float64)
    k = scale_exponent(a)
    ints = np.rint(a / 10.0**k).astype(np.int64)
    return QuantizedTensor(name, ints, k)


def quantize(params: Mapping[str, np.ndarray]) -> dict[str, QuantizedTensor]:
    return {name: quantize_tensor(name, w) for name, w in params.items()}


def dequantize(q: Mapping[str, QuantizedTensor]) -> dict[str, np.ndarray]:
    return {name: t.dequantize() for name, t in q.items()}


def rounded(params: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The network as shown to the model: quantize, then dequantize (float64)."""
    return dequantize(quantize(params))
