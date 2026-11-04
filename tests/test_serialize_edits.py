"""Delivery 2, acceptance D (edit parser) and E (applier)."""

from __future__ import annotations

import numpy as np
import pytest

from weightshift.policy import random_actor
from weightshift.serialize import SKIP_REASONS, apply_edits, edit_from_reply, parse_edits, rounded, shapes_of

SHAPES = {"A.W": (4, 3), "A.b": (3,), "B.W": (3, 5), "B.b": (5,)}


def net():
    """A small non-square network with distinct entries, so orientation mistakes show."""
    return {
        "A.W": np.arange(12, dtype=np.float64).reshape(4, 3) + 1,
        "A.b": np.array([0.5, 1.5, 2.5]),
        "B.W": np.arange(15, dtype=np.float64).reshape(3, 5) * 10 + 100,
        "B.b": np.zeros(5),
    }


def block(*lines: str) -> str:
    return "Explanation first.\n```edits\n" + "\n".join(lines) + "\n```\nWhat the edits do.\n"


def run(*lines: str):
    n = net()
    parsed, after = edit_from_reply(n, block(*lines))
    return n, parsed, after


# ------------------------------------------------------------------ D1: forms and ops
def test_single_entry():
    _, p, a = run("B.W[1, 4] = 1.25")
    assert p.n_statements == 1 and p.n_skipped == 0 and p.clean
    assert a["B.W"][1, 4] == 1.25


def test_column_slice_and_whole_column():
    n, p, a = run("A.W[1:3, 2] = -2", "B.W[:, 0] = 7")
    assert a["A.W"][1:3, 2].tolist() == [-2, -2]
    assert a["A.W"][0, 2] == n["A.W"][0, 2] and a["A.W"][3, 2] == n["A.W"][3, 2]
    assert a["B.W"][:, 0].tolist() == [7, 7, 7]


def test_row_slice_open_ended_and_full():
    n, _, a = run("A.W[2:, :] *= 0.5", "B.W[:1, 3:] = 9")
    assert np.array_equal(a["A.W"][2:], n["A.W"][2:] * 0.5)
    assert np.array_equal(a["A.W"][:2], n["A.W"][:2])
    assert a["B.W"][0, 3:].tolist() == [9, 9] and a["B.W"][0, 2] == n["B.W"][0, 2]


def test_one_d_entry_and_slice_list():
    _, _, a = run("A.b[1] = 3", "B.b[0:3] = [0.1, 0.2, 0.3]")
    assert a["A.b"].tolist() == [0.5, 3.0, 2.5]
    assert a["B.b"].tolist() == [0.1, 0.2, 0.3, 0, 0]


@pytest.mark.parametrize("op, want", [("=", 2.0), ("+=", 7.0), ("-=", 3.0), ("*=", 10.0)])
def test_operators(op, want):
    # A.W[1, 1] is 5
    _, _, a = run(f"A.W[1, 1] {op} 2")
    assert a["A.W"][1, 1] == want


def test_zero_tensor():
    _, p, a = run("ZERO B.W")
    assert p.statements[0].op == "zero"
    assert not a["B.W"].any()


def test_number_forms():
    _, p, a = run("B.b[:] = [1, -2.5, +3., .5, 1e-3]")
    assert p.n_skipped == 0
    assert a["B.b"].tolist() == [1, -2.5, 3.0, 0.5, 0.001]


def test_two_d_list_is_row_major():
    _, _, a = run("A.W[0:2, 1:3] = [1, 2, 3, 4]")
    assert a["A.W"][0:2, 1:3].tolist() == [[1, 2], [3, 4]]


def test_multi_line_list_and_comments():
    _, p, a = run(
        "# a comment line",
        "B.b[0:5] = [1, 2,   # first two",
        "            3,",
        "   4, 5]",
        "A.b[0] = 9   # trailing comment",
    )
    assert p.n_statements == 2 and p.n_skipped == 0
    assert a["B.b"].tolist() == [1, 2, 3, 4, 5]
    assert a["A.b"][0] == 9
    assert p.statements[0].line == 4  # line numbers are in the reply (block starts on line 3)


def test_unclosed_list_does_not_swallow_the_next_statement():
    _, p, a = run("B.b[0:2] = [1, 2", "A.b[0] = 9")
    assert [s.reason for s in p.skipped] == ["syntax"]
    assert p.n_statements == 1 and a["A.b"][0] == 9


def test_trailing_comma_allowed():
    _, p, a = run("A.b[0:2] = [1, 2,]")
    assert p.n_skipped == 0 and a["A.b"][:2].tolist() == [1, 2]


def test_last_block_wins():
    reply = block("A.b[0] = 1") + "\nOn second thought:\n" + block("A.b[0] = 2")
    p, a = edit_from_reply(net(), reply)
    assert p.n_blocks == 2 and p.n_statements == 1 and a["A.b"][0] == 2


def test_other_code_blocks_are_ignored():
    reply = "```python\nA.b[0] = 5\n```\n" + block("A.b[1] = 6") + "```\nA.b[2] = 7\n```\n"
    p, a = edit_from_reply(net(), reply)
    assert p.n_blocks == 1 and a["A.b"].tolist() == [0.5, 6, 2.5]


def test_no_block():
    n = net()
    p, a = edit_from_reply(n, "I cannot do this.\nA.b[0] = 5\n")
    assert p.no_block and not p.has_block and not p.clean
    assert p.n_statements == 0 and all(np.array_equal(a[k], n[k]) for k in n)


def test_unterminated_block_is_read_and_flagged():
    p, a = edit_from_reply(net(), "text\n```edits\nA.b[0] = 4\nA.b[1] = 5")
    assert p.cut_off and p.n_statements == 2 and a["A.b"][:2].tolist() == [4, 5]


# ------------------------------------------------------------------ D2: skip reasons
@pytest.mark.parametrize(
    "line, reason",
    [
        ("A.W[1, 1] == 2", "syntax"),
        ("A.W[1, 1] = two", "syntax"),
        ("A.W[1, 1]", "syntax"),
        ("A.W[0:4:2, 1] = 0", "syntax"),
        ("A.W[1.5, 1] = 0", "syntax"),
        ("A.W[, 1] = 0", "syntax"),
        ("A.b[0] = [1, [2]]", "syntax"),
        ("A.b[0:2] = []", "syntax"),
        ("A.W = 0", "syntax"),
        ("ZERO", "syntax"),
        ("A.b[0] = nan", "syntax"),
        ("A.b[0] = inf", "syntax"),
        ("A.b[0] = 1/3", "syntax"),
        ("C.W[0, 0] = 1", "unknown_tensor"),
        ("ZERO C.W", "unknown_tensor"),
        ("a.w[0, 0] = 1", "unknown_tensor"),
        ("A.W[1] = 0", "wrong_rank"),
        ("A.b[1, 0] = 0", "wrong_rank"),
        ("A.W[4, 0] = 1", "index_out_of_range"),
        ("A.W[0, 3] = 1", "index_out_of_range"),
        ("A.W[0:5, 0] = 1", "index_out_of_range"),
        ("A.W[2:2, 0] = 1", "index_out_of_range"),
        ("A.W[3:1, 0] = 1", "index_out_of_range"),
        ("A.W[-1, 0] = 1", "index_out_of_range"),
        ("A.b[0:2] = [1, 2, 3]", "list_length"),
        ("A.W[0:2, 0:2] = [1, 2, 3]", "list_length"),
        ("A.b[0] = 1000001", "value_out_of_range"),
        ("A.b[0] = -2e6", "value_out_of_range"),
        ("A.b[0] *= 1e400", "value_out_of_range"),
        ("A.b[0:2] = [1, 5e6]", "value_out_of_range"),
    ],
)
def test_skip_reasons(line, reason):
    n, p, a = run("A.b[2] = 42", line, "B.b[4] = -1")
    assert [s.reason for s in p.skipped] == [reason], p.skipped
    assert p.skipped[0].line == 4 and p.skipped[0].text == line
    assert not p.clean
    # the other statements still apply
    assert p.n_statements == 2
    assert a["A.b"][2] == 42 and a["B.b"][4] == -1
    assert reason in SKIP_REASONS


def test_value_limit_is_inclusive():
    _, p, a = run("A.b[0] = 1e6", "A.b[1] = -1000000")
    assert p.n_skipped == 0 and a["A.b"][:2].tolist() == [1e6, -1e6]


# ------------------------------------------------------------------ D3: fuzz
SENTINELS = [
    "__import__('os').system('echo pwned')",
    "exec('x=1')",
    "eval('1+1')",
    "open('/etc/passwd').read()",
    "lambda: 0",
]


def _mutate(rng: np.random.Generator, s: str) -> str:
    chars = list(s)
    alphabet = list("[]:,=+-*/#.0123456789eE ABWbZERO_\n\t()'\"`\\x")
    for _ in range(int(rng.integers(1, 6))):
        op = rng.integers(3)
        pos = int(rng.integers(len(chars) + 1))
        if op == 0:
            chars.insert(pos, alphabet[int(rng.integers(len(alphabet)))])
        elif op == 1 and chars:
            del chars[min(pos, len(chars) - 1)]
        elif chars:
            chars[min(pos, len(chars) - 1)] = alphabet[int(rng.integers(len(alphabet)))]
    return "".join(chars)


def test_fuzz_never_raises_and_never_executes(monkeypatch):
    import builtins

    def boom(*a, **k):  # any evaluation attempt fails the test
        raise AssertionError("parser evaluated text")

    monkeypatch.setattr(builtins, "eval", boom)
    monkeypatch.setattr(builtins, "exec", boom)
    rng = np.random.default_rng(2026)
    seeds = [
        "A.W[1, 2] = 3.5",
        "B.b[0:3] = [1, 2, 3]",
        "ZERO A.b",
        "A.W[:, 1] *= 0.5",
        "B.W[0:2, :] += -1e-3",
    ]
    for i in range(10_000):
        kind = i % 4
        if kind == 0:
            lines = [_mutate(rng, seeds[int(rng.integers(len(seeds)))]) for _ in range(int(rng.integers(1, 6)))]
            text = block(*lines)
        elif kind == 1:
            n = int(rng.integers(0, 200))
            text = "".join(chr(int(c)) for c in rng.integers(0, 0x2FF, size=n))
        elif kind == 2:
            text = block(f"A.b[0] = {SENTINELS[i % len(SENTINELS)]}", f"{SENTINELS[(i + 1) % len(SENTINELS)]}")
        else:
            text = _mutate(rng, block(*seeds))
        p = parse_edits(text, SHAPES)
        apply_edits(net(), p.statements)  # whatever was accepted must also apply cleanly
        if kind == 2:
            assert p.n_statements == 0 and {s.reason for s in p.skipped} == {"syntax"}


# ------------------------------------------------------------------ E
def test_statements_apply_in_order():
    _, _, a = run("A.b[0] = 5", "A.b[0] = 7", "A.b[1] = 1", "A.b[1] += 2", "A.W[:, 0] = 1", "A.W[2, 0] *= 3")
    assert a["A.b"][:2].tolist() == [7, 3]
    assert a["A.W"][:, 0].tolist() == [1, 1, 3, 1]


def test_zero_then_rewire_like_exp4():
    _, _, a = run("B.W[:, 3:5] = 0", "B.W[1, 4] = 80")
    assert a["B.W"][:, 3].tolist() == [0, 0, 0]
    assert a["B.W"][:, 4].tolist() == [0, 80, 0]


def test_orientation_on_non_square_tensors():
    """Row index = first axis (input), column = second (output); a swapped applier fails."""
    _, _, a = run("A.W[3, 0] = -1", "B.W[0:3, 4] = [1, 2, 3]", "B.W[2, 0:5] = [5, 6, 7, 8, 9]")
    assert a["A.W"][3, 0] == -1 and a["A.W"][0, 3 % 3] != -1
    assert a["B.W"][:2, 4].tolist() == [1, 2]
    assert a["B.W"][2].tolist() == [5, 6, 7, 8, 9]


def test_input_is_not_modified():
    n = net()
    before = {k: v.copy() for k, v in n.items()}
    edit_from_reply(n, block("ZERO A.W", "B.b[:] = 1", "A.b[0] *= 2"))
    assert all(np.array_equal(n[k], before[k]) for k in n)


def test_result_is_float_and_not_requantized():
    r = rounded(random_actor(0))
    p, a = edit_from_reply(r, block("L2.W[0, 0] = 0.123456789"))
    assert p.n_statements == 1 and a["L2.W"][0, 0] == 0.123456789 and a["L2.W"].dtype == np.float64


def test_our_actor_shapes():
    p = parse_edits(block("L1.W[599, 31] = 1", "L1.W[600, 0] = 1", "L3.b[6] = 0", "L3.b[7] = 0"), shapes_of(random_actor(0)))
    assert p.n_statements == 2 and [s.reason for s in p.skipped] == ["index_out_of_range"] * 2


def test_new_edits_opener_inside_an_open_block_starts_a_new_block():
    """A reply that opens a second block without closing the first: the second is the last block."""
    p, a = edit_from_reply(net(), "```edits\nA.b[0] = 1\n```edits\nA.b[0] = 2\n```\n")
    assert p.n_blocks == 2 and not p.cut_off
    assert [s.values for s in p.statements] == [(2.0,)] and a["A.b"][0] == 2


def test_unclosed_block_before_a_new_opener_is_cut_off_but_not_last():
    p, a = edit_from_reply(net(), "```edits\nA.b[0] = 1\n```edits\nA.b[1] = 2")
    assert p.n_blocks == 2 and p.cut_off  # the last block is the unterminated one
    assert a["A.b"][:2].tolist() == [0.5, 2]


def test_only_a_bare_fence_closes_a_block():
    p, a = edit_from_reply(net(), "```edits\nA.b[0] = 1\n```python\nA.b[1] = 2\n```\n")
    assert p.n_blocks == 1 and a["A.b"][:2].tolist() == [1, 2]
    assert [s.reason for s in p.skipped] == ["syntax"]  # the ```python line is not a statement
