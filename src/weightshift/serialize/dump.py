"""Weight dump text in Exp 4's format, and a reader for it.

    L1.W (600 x 32), scale 1e-2: one line per row (row i)
    0: 1 -3 -4 ...
    ...

    L1.b (32), scale 1e-4:
    -117 39 ...

Rows of a matrix are its *inputs* (W is (inputs, outputs)). Tensors are
separated by one blank line and the text ends with a newline. The layout is
copied byte for byte from Exp 4's `prompt.txt` (golden test F1).
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np

from ..policy.actor import ACTOR_SPEC, validate
from .quantize import QuantizedTensor, quantize

WEIGHTS_INTRO = "Each tensor is written as integers with one scale: value = integer x scale."


def format_scale(k: int) -> str:
    return f"1e{k}"


def _ints(row: np.ndarray) -> str:
    return " ".join(str(int(v)) for v in row)


def dump_tensor(t: QuantizedTensor) -> str:
    a = t.ints
    if a.ndim == 2:
        r, c = a.shape
        lines = [f"{t.name} ({r} x {c}), scale {format_scale(t.k)}: one line per row (row i)"]
        lines += [f"{i}: {_ints(a[i])}" for i in range(r)]
    elif a.ndim == 1:
        lines = [f"{t.name} ({a.shape[0]}), scale {format_scale(t.k)}:", _ints(a)]
    else:
        raise ValueError(f"{t.name}: only 1-D and 2-D tensors can be dumped, got shape {a.shape}")
    return "\n".join(lines) + "\n"


def dump_quantized(q: Mapping[str, QuantizedTensor]) -> str:
    return "\n".join(dump_tensor(t) for t in q.values())


def dump_weights(params: Mapping[str, np.ndarray]) -> str:
    """Quantize and dump any ordered dict of 1-D/2-D tensors (e.g. Exp 4's)."""
    return dump_quantized(quantize(params))


def dump_actor(params: Mapping[str, np.ndarray]) -> str:
    """Dump our actor after checking names and shapes (a transposed W is rejected)."""
    return dump_weights(validate(params, ACTOR_SPEC))


# ------------------------------------------------------------------ reader
_HEADER_2D = re.compile(r"^(\S+) \((\d+) x (\d+)\), scale 1e(-?\d+): one line per row \(row i\)$")
_HEADER_1D = re.compile(r"^(\S+) \((\d+)\), scale 1e(-?\d+):$")


class DumpError(ValueError):
    """Text is not a well-formed weight dump."""


def _parse_ints(text: str, n: int, where: str) -> list[int]:
    parts = text.split()
    if len(parts) != n:
        raise DumpError(f"{where}: expected {n} integers, found {len(parts)}")
    try:
        return [int(p) for p in parts]
    except ValueError as e:
        raise DumpError(f"{where}: {e}") from None


def read_dump(text: str) -> dict[str, QuantizedTensor]:
    """Parse dump text back into integers and scales. Text outside tensors is ignored."""
    lines = text.split("\n")
    out: dict[str, QuantizedTensor] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        m2 = _HEADER_2D.match(line)
        m1 = None if m2 else _HEADER_1D.match(line)
        if m2:
            name, r, c, k = m2.group(1), int(m2.group(2)), int(m2.group(3)), int(m2.group(4))
            if i + r >= len(lines):
                raise DumpError(f"{name}: expected {r} rows, text ends early")
            rows = []
            for j in range(r):
                row = lines[i + 1 + j]
                prefix = f"{j}: "
                if not row.startswith(prefix):
                    raise DumpError(f"{name}: row {j} must start with {prefix!r}")
                rows.append(_parse_ints(row[len(prefix):], c, f"{name} row {j}"))
            ints = np.array(rows, dtype=np.int64).reshape(r, c)
            i += 1 + r
        elif m1:
            name, n, k = m1.group(1), int(m1.group(2)), int(m1.group(3))
            if i + 1 >= len(lines):
                raise DumpError(f"{name}: values line missing")
            ints = np.array(_parse_ints(lines[i + 1], n, name), dtype=np.int64)
            i += 2
        else:
            i += 1
            continue
        if name in out:
            raise DumpError(f"tensor {name} appears twice")
        out[name] = QuantizedTensor(name, ints, k)
    if not out:
        raise DumpError("no tensors found")
    return out
