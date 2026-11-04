"""Edit analysis (FINAL_PLAN §10) and unit ablation (§6.4, mechanism ablation).

Layers are found from the `.W` tensors in order (L1, L2, L3 for our actor;
embed, actor0..4, actor_head for Exp 4). Every layer except the last has
hidden units; the last layer's outputs are the action logits.

A unit counts as *touched* (taken over) when its bias changed, or its
incoming weights changed on a row that is **not** itself a touched unit of the
previous layer (for the first layer, every row is an input). Layers are
processed in order. The exclusion matters because taking over a unit usually
starts by zeroing its outgoing row, which crosses every column of the next
matrix; without it, every unit of the next layer would count. On Exp 4 this
rule gives Astra's published counts: embed 3, actor0 6, actor1 4, actor2-4 2
each, and outputs 0-2. Output (action) units use the same rule.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from ..policy.actor import Params, copy_params, layer_names
from .edits import EditParse


@dataclass(frozen=True)
class EditAnalysis:
    n_changed: int
    n_params: int
    frac_changed: float
    l2_change: float
    changed_per_tensor: dict[str, int]
    units_touched: dict[str, tuple[int, ...]]  # hidden layer -> touched units (see module doc)
    outputs_touched: tuple[int, ...]  # touched output (action) units
    n_statements: int | None = None
    n_skipped: int | None = None
    tensors_changed: tuple[str, ...] = field(default=())

    @property
    def units_per_layer(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.units_touched.items()}

    def to_json(self) -> dict:
        return {
            "n_changed": self.n_changed,
            "n_params": self.n_params,
            "frac_changed": self.frac_changed,
            "l2_change": self.l2_change,
            "changed_per_tensor": dict(self.changed_per_tensor),
            "units_touched": {k: list(v) for k, v in self.units_touched.items()},
            "units_per_layer": self.units_per_layer,
            "outputs_touched": list(self.outputs_touched),
            "n_statements": self.n_statements,
            "n_skipped": self.n_skipped,
        }


def _check_same_structure(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray]) -> None:
    if list(a) != list(b):
        raise ValueError(f"tensor names differ: {list(a)} vs {list(b)}")
    for k in a:
        if np.shape(a[k]) != np.shape(b[k]):
            raise ValueError(f"{k}: shape {np.shape(a[k])} vs {np.shape(b[k])}")


def analyse_edit(
    before: Mapping[str, np.ndarray],
    after: Mapping[str, np.ndarray],
    parsed: EditParse | None = None,
) -> EditAnalysis:
    """Compare an edited network with the network it was edited from (the rounded one)."""
    _check_same_structure(before, after)
    diff = {k: np.asarray(after[k], dtype=np.float64) - np.asarray(before[k], dtype=np.float64) for k in before}
    changed = {k: np.asarray(after[k]) != np.asarray(before[k]) for k in before}
    per_tensor = {k: int(m.sum()) for k, m in changed.items()}
    n_changed = sum(per_tensor.values())
    n = sum(int(np.size(v)) for v in before.values())
    l2 = float(np.sqrt(sum(float(np.sum(d * d)) for d in diff.values())))

    layers = layer_names(before)
    touched_per_layer: list[np.ndarray] = []
    prev_touched: np.ndarray | None = None  # rows of the current W that are touched units
    for layer in layers:
        w_changed = changed[f"{layer}.W"]
        if prev_touched is not None:
            w_changed = w_changed & ~prev_touched[:, None]
        hit = w_changed.any(axis=0) | changed[f"{layer}.b"]
        touched_per_layer.append(hit)
        prev_touched = hit
    units = {
        layer: tuple(int(u) for u in np.flatnonzero(hit)) for layer, hit in zip(layers[:-1], touched_per_layer[:-1])
    }
    out_hit = touched_per_layer[-1]
    return EditAnalysis(
        n_changed=n_changed,
        n_params=n,
        frac_changed=n_changed / n if n else 0.0,
        l2_change=l2,
        changed_per_tensor=per_tensor,
        units_touched=units,
        outputs_touched=tuple(int(u) for u in np.flatnonzero(out_hit)),
        n_statements=None if parsed is None else parsed.n_statements,
        n_skipped=None if parsed is None else parsed.n_skipped,
        tensors_changed=tuple(k for k, c in per_tensor.items() if c),
    )


# ------------------------------------------------------------------ ablation
def _next_layer(params: Mapping[str, np.ndarray], layer: str) -> str:
    layers = layer_names(params)
    if layer not in layers:
        raise ValueError(f"unknown layer {layer!r}; layers are {layers}")
    i = layers.index(layer)
    if i == len(layers) - 1:
        raise ValueError(f"{layer} is the output layer; it has no hidden units")
    return layers[i + 1]


def zero_units(params: Mapping[str, np.ndarray], layer: str, units: Iterable[int]) -> Params:
    """Switch off hidden units: zero their incoming column, bias and outgoing row (returns a copy)."""
    nxt = _next_layer(params, layer)
    out = copy_params(params)
    idx = sorted({int(u) for u in units})
    width = out[f"{layer}.b"].shape[0]
    bad = [u for u in idx if not 0 <= u < width]
    if bad:
        raise ValueError(f"{layer} has {width} units; got {bad}")
    out[f"{layer}.W"][:, idx] = 0.0
    out[f"{layer}.b"][idx] = 0.0
    out[f"{nxt}.W"][idx, :] = 0.0
    return out


def rank_units_by_outgoing_l2(params: Mapping[str, np.ndarray], layer: str) -> list[int]:
    """Units of a hidden layer, lowest outgoing-weight L2 norm first; ties by lower index."""
    nxt = _next_layer(params, layer)
    norms = np.linalg.norm(np.asarray(params[f"{nxt}.W"], dtype=np.float64), axis=1)
    return [int(u) for u in np.argsort(norms, kind="stable")]
