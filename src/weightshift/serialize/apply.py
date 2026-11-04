"""Apply parsed edit statements to a network (FINAL_PLAN §7.3).

Statements apply in order to a float64 copy of the network as shown (the
rounded weights), so a later statement overwrites an earlier one. The result
is left in float and is not re-quantized. The input is never modified.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from ..policy.actor import Params, copy_params
from .edits import EditParse, Statement, parse_edits, shapes_of


def _selector(st: Statement) -> tuple:
    return tuple(a if point else slice(a, b) for a, b, point in st.index)


def apply_statement(params: Params, st: Statement) -> None:
    """Apply one statement in place."""
    arr = params[st.tensor]
    if st.op == "zero":
        arr[...] = 0.0
        return
    sel = _selector(st)
    target_shape = np.shape(arr[sel])
    if st.is_list:
        value = np.asarray(st.values, dtype=np.float64).reshape(target_shape)  # row-major
    else:
        value = np.float64(st.values[0])
    if st.op == "=":
        arr[sel] = value
    elif st.op == "+=":
        arr[sel] = arr[sel] + value
    elif st.op == "-=":
        arr[sel] = arr[sel] - value
    elif st.op == "*=":
        arr[sel] = arr[sel] * value
    else:  # pragma: no cover - the parser only produces the ops above
        raise ValueError(f"unknown op {st.op!r}")


def apply_edits(params: Mapping[str, np.ndarray], statements: Iterable[Statement]) -> Params:
    out = copy_params(params)
    for st in statements:
        apply_statement(out, st)
    return out


def edit_from_reply(rounded_params: Mapping[str, np.ndarray], reply: str) -> tuple[EditParse, Params]:
    """Parse the last edits block of a stored reply and apply it to the rounded network."""
    parsed = parse_edits(reply, shapes_of(rounded_params))
    return parsed, apply_edits(rounded_params, parsed.statements)
