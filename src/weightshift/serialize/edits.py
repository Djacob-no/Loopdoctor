"""Parser for the Exp 4 edit language (FINAL_PLAN §7.2, DELIVERY_2_ACCEPTANCE §2).

    ```edits
    L3.b[0:3] = [0.5, 2.0, -1.0]   # one value per entry of the range
    L3.W[7, 0:3] = 0               # fill a block (a:b half-open, ':' = whole axis)
    L1.W[26, 5] = 1.25
    L2.W[:, 3] *= 0.5              # also += and -=
    ZERO L2.W
    ```

Only the last block opened by three backticks + `edits` is read. A statement
that cannot be read is skipped with a reason; the rest still apply.

The parser is a hand-written tokenizer and regexes. It never evaluates the
text and never raises on any input string: every failure becomes a skip.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

MAX_ABS_VALUE = 1e6

OPS = ("=", "+=", "-=", "*=")

# Skip reasons
SYNTAX = "syntax"
UNKNOWN_TENSOR = "unknown_tensor"
WRONG_RANK = "wrong_rank"
INDEX_OUT_OF_RANGE = "index_out_of_range"
LIST_LENGTH = "list_length"
VALUE_OUT_OF_RANGE = "value_out_of_range"
SKIP_REASONS = (SYNTAX, UNKNOWN_TENSOR, WRONG_RANK, INDEX_OUT_OF_RANGE, LIST_LENGTH, VALUE_OUT_OF_RANGE)

_NAME = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_RE_ZERO = re.compile(rf"^ZERO\s+({_NAME})$")
_RE_ASSIGN = re.compile(rf"^({_NAME})\s*\[([^\[\]]*)\]\s*(\+=|-=|\*=|=)\s*(.+)$")
_RE_NUMBER = re.compile(rf"^{_NUMBER}$")
_RE_INT = re.compile(r"^[+-]?\d+$")
_RE_STARTS_STATEMENT = re.compile(rf"^(?:ZERO\s|{_NAME}\s*\[)")
_RE_FENCE_OPEN = re.compile(r"^\s*```\s*edits\s*$")
_RE_FENCE_CLOSE = re.compile(r"^\s*```\s*$")


@dataclass(frozen=True)
class Statement:
    """One readable, validated statement.

    `index` holds one (start, stop, is_point) per axis; a point selects a
    single position and drops that axis. `ZERO` has op "zero" and no index.
    """

    line: int  # 1-based line number in the reply
    text: str
    tensor: str
    op: str  # "=", "+=", "-=", "*=" or "zero"
    index: tuple[tuple[int, int, bool], ...] = ()
    values: tuple[float, ...] = ()  # one value (broadcast) or one per selected entry
    is_list: bool = False

    @property
    def n_selected(self) -> int:
        n = 1
        for a, b, _ in self.index:
            n *= b - a
        return n


@dataclass(frozen=True)
class Skipped:
    line: int
    text: str
    reason: str
    detail: str = ""


@dataclass
class EditParse:
    statements: list[Statement] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
    has_block: bool = False
    n_blocks: int = 0
    cut_off: bool = False  # the last block has no closing fence
    block_text: str = ""

    @property
    def no_block(self) -> bool:
        return not self.has_block

    @property
    def n_statements(self) -> int:
        return len(self.statements)

    @property
    def n_skipped(self) -> int:
        return len(self.skipped)

    @property
    def clean(self) -> bool:
        """Pre-registered: an attempt with any skipped line is not a clean success."""
        return self.has_block and not self.skipped


class _Skip(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


# --------------------------------------------------------------- block finding
def find_last_block(reply: str) -> tuple[list[tuple[int, str]], int, bool]:
    """Lines (1-based number, text) of the last edits block, the block count, and cut_off.

    Only a bare fence closes a block. A new ```edits opener inside an open
    block ends that block as cut off and starts the next one.
    """
    lines = reply.splitlines()
    blocks: list[tuple[list[tuple[int, str]], bool]] = []
    i = 0
    while i < len(lines):
        if _RE_FENCE_OPEN.match(lines[i]):
            body: list[tuple[int, str]] = []
            j = i + 1
            while j < len(lines) and not _RE_FENCE_CLOSE.match(lines[j]) and not _RE_FENCE_OPEN.match(lines[j]):
                body.append((j + 1, lines[j]))
                j += 1
            closed = j < len(lines) and bool(_RE_FENCE_CLOSE.match(lines[j]))
            blocks.append((body, not closed))
            i = j + 1 if closed else j  # do not consume a new opener
        else:
            i += 1
    if not blocks:
        return [], 0, False
    body, cut = blocks[-1]
    return body, len(blocks), cut


def _strip_comment(s: str) -> str:
    k = s.find("#")
    return (s if k < 0 else s[:k]).strip()


def _bracket_depth(s: str) -> int:
    return s.count("[") - s.count("]")


def _logical_statements(body: list[tuple[int, str]]) -> list[tuple[int, str, str]]:
    """Join continued list lines. Returns (first line no, raw text, code without comments)."""
    out: list[tuple[int, str, str]] = []
    cur: tuple[int, list[str], list[str]] | None = None  # (line, raw parts, code parts)
    for no, raw in body:
        code = _strip_comment(raw)
        if cur is not None:
            # an unclosed list continues, unless this line clearly starts a new statement
            if code and _RE_STARTS_STATEMENT.match(code):
                out.append((cur[0], "\n".join(cur[1]), " ".join(cur[2])))
                cur = None
            else:
                cur[1].append(raw)
                if code:
                    cur[2].append(code)
                if _bracket_depth(" ".join(cur[2])) <= 0:
                    out.append((cur[0], "\n".join(cur[1]), " ".join(cur[2])))
                    cur = None
                continue
        if not code:
            continue
        rhs = code.split("=", 1)[1] if "=" in code else ""
        if rhs and _bracket_depth(rhs) > 0:
            cur = (no, [raw], [code])
        else:
            out.append((no, raw, code))
    if cur is not None:
        out.append((cur[0], "\n".join(cur[1]), " ".join(cur[2])))
    return out


# --------------------------------------------------------------- statements
def _parse_number(s: str) -> float:
    s = s.strip()
    if not _RE_NUMBER.match(s):
        raise _Skip(SYNTAX, f"not a plain number: {s!r}")
    v = float(s)
    if not math.isfinite(v) or abs(v) > MAX_ABS_VALUE:
        raise _Skip(VALUE_OUT_OF_RANGE, f"|{s}| > {MAX_ABS_VALUE:g}")
    return v


def _parse_rhs(s: str) -> tuple[tuple[float, ...], bool]:
    s = s.strip()
    if s.startswith("["):
        if not s.endswith("]") or _bracket_depth(s) != 0 or s.count("[") != 1:
            raise _Skip(SYNTAX, "malformed list")
        inner = s[1:-1].strip()
        if inner.endswith(","):
            inner = inner[:-1]
        if not inner:
            raise _Skip(SYNTAX, "empty list")
        return tuple(_parse_number(p) for p in inner.split(",")), True
    return (_parse_number(s),), False


def _parse_bound(s: str, dim: int, axis: int) -> int:
    s = s.strip()
    if not _RE_INT.match(s):
        raise _Skip(SYNTAX, f"index {s!r} is not an integer")
    v = int(s)
    if v < 0 or v > dim:
        raise _Skip(INDEX_OUT_OF_RANGE, f"axis {axis}: {v} outside 0..{dim}")
    return v


def _parse_index(s: str, shape: tuple[int, ...]) -> tuple[tuple[int, int, bool], ...]:
    parts = s.split(",")
    if any(not p.strip() for p in parts):
        raise _Skip(SYNTAX, "empty index")
    if len(parts) != len(shape):
        raise _Skip(WRONG_RANK, f"{len(parts)} indices for a {len(shape)}-D tensor")
    out = []
    for axis, (p, dim) in enumerate(zip(parts, shape)):
        if ":" in p:
            pieces = p.split(":")
            if len(pieces) != 2:
                raise _Skip(SYNTAX, f"slice {p.strip()!r} (steps are not allowed)")
            a = _parse_bound(pieces[0], dim, axis) if pieces[0].strip() else 0
            b = _parse_bound(pieces[1], dim, axis) if pieces[1].strip() else dim
            if a >= b:
                raise _Skip(INDEX_OUT_OF_RANGE, f"axis {axis}: empty slice {a}:{b}")
            out.append((a, b, False))
        else:
            v = _parse_bound(p, dim, axis)
            if v >= dim:
                raise _Skip(INDEX_OUT_OF_RANGE, f"axis {axis}: {v} outside 0..{dim - 1}")
            out.append((v, v + 1, True))
    return tuple(out)


def parse_statement(line: int, raw: str, code: str, shapes: Mapping[str, tuple[int, ...]]) -> Statement:
    """Parse one logical statement; raises _Skip."""
    m = _RE_ZERO.match(code)
    if m:
        name = m.group(1)
        if name not in shapes:
            raise _Skip(UNKNOWN_TENSOR, name)
        return Statement(line, raw, name, "zero")
    m = _RE_ASSIGN.match(code)
    if not m:
        raise _Skip(SYNTAX, "expected TENSOR[index] op value, or ZERO TENSOR")
    name, idx, op, rhs = m.groups()
    if name not in shapes:
        raise _Skip(UNKNOWN_TENSOR, name)
    index = _parse_index(idx, tuple(shapes[name]))
    values, is_list = _parse_rhs(rhs)
    st = Statement(line, raw, name, op, index, values, is_list)
    if is_list and len(values) != st.n_selected:
        raise _Skip(LIST_LENGTH, f"{len(values)} values for {st.n_selected} entries")
    return st


def parse_edits(reply: str, shapes: Mapping[str, tuple[int, ...]]) -> EditParse:
    """Read the last edits block of `reply` against tensors of the given shapes. Never raises."""
    result = EditParse()
    try:
        body, n_blocks, cut = find_last_block(reply)
    except Exception as e:  # pragma: no cover - defensive; find_last_block has no failure path
        result.skipped.append(Skipped(0, "", SYNTAX, f"internal: {e!r}"))
        return result
    result.n_blocks = n_blocks
    result.has_block = n_blocks > 0
    result.cut_off = cut
    result.block_text = "\n".join(t for _, t in body)
    for line, raw, code in _logical_statements(body):
        try:
            result.statements.append(parse_statement(line, raw, code, shapes))
        except _Skip as s:
            result.skipped.append(Skipped(line, raw, s.reason, s.detail))
        except Exception as e:  # never let a reply crash the harness
            result.skipped.append(Skipped(line, raw, SYNTAX, f"internal: {e!r}"))
    return result


def shapes_of(params: Mapping[str, object]) -> dict[str, tuple[int, ...]]:
    return {k: tuple(v.shape) for k, v in params.items()}  # type: ignore[attr-defined]
