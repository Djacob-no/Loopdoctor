"""Delivery 2, acceptance C (quantization and dump) and G (round trip on our own network)."""

from __future__ import annotations

import re

import numpy as np
import pytest

from weightshift.env import generate_layout, seeds_for
from weightshift.policy import PolicyPlayer, action_probs, random_actor
from weightshift.serialize import (
    DumpError,
    analyse_edit,
    dequantize,
    dump_actor,
    dump_weights,
    edit_from_reply,
    quantize,
    quantize_tensor,
    read_dump,
    rounded,
    scale_exponent,
)
from weightshift.trajectory import run_episode


# ------------------------------------------------------------------ C1
@pytest.mark.parametrize(
    "mx, k",
    [
        (0.0123, -4),
        (1.23, -2),
        (80.0, -1),
        (0.001, -5),
        (1.0, -2),
        (9.99, -2),
        (10.0, -1),
        (999.0, 0),
        (2.37, -2),  # Exp 4 embed.W
    ],
)
def test_scale_exponent(mx, k):
    w = np.array([0.0, mx / 3, -mx / 7, mx])
    assert scale_exponent(w) == k


def test_scale_uses_largest_magnitude_even_if_negative():
    assert scale_exponent(np.array([0.5, -4.2, 0.1])) == -2
    q = quantize_tensor("t", np.array([0.5, -4.2, 0.1]))
    assert q.ints.tolist() == [50, -420, 10]


def test_all_zero_tensor_uses_scale_1e0():
    q = quantize_tensor("t", np.zeros((3, 2)))
    assert q.k == 0 and not q.ints.any()
    assert "scale 1e0:" in dump_weights({"t.W": np.zeros((3, 2))})


def test_nonfinite_tensor_rejected():
    with pytest.raises(ValueError):
        scale_exponent(np.array([1.0, np.nan]))


def test_round_half_to_even_and_dequantize():
    q = quantize_tensor("t", np.array([1.0, 0.0125, 0.0135, -0.0125]))  # k = -2
    assert q.k == -2
    assert q.ints.tolist() == [100, 1, 1, -1]
    assert np.array_equal(q.dequantize(), q.ints * 10.0**-2)


def test_exact_halves_round_to_even():
    q = quantize_tensor("t", np.array([100.0, 2.5, 3.5, -2.5]))  # k = 0
    assert q.k == 0 and q.ints.tolist() == [100, 2, 4, -2]


# ------------------------------------------------------------------ C2
def test_dumped_integers_have_at_most_three_digits():
    for seed in range(20):
        a = random_actor(seed, scale=10.0 ** (seed % 5 - 2))
        text = dump_actor(a)
        nums = [int(t) for line in text.splitlines() if not line.startswith("L") for t in re.findall(r"-?\d+", line.split(": ", 1)[-1])]
        assert max(abs(n) for n in nums) <= 1000
        # the 4-digit case can only be the rounding edge of the largest entry
        assert sum(abs(n) == 1000 for n in nums) <= 6


def test_rounding_edge_documented():
    """999.6 x 10^k rounds to 1000: four digits, scale unchanged (DELIVERY_2 §8)."""
    q = quantize_tensor("t", np.array([9.996, 1.0]))
    assert q.k == -2 and q.ints.tolist() == [1000, 100]


def test_dump_format():
    a = random_actor(0)
    text = dump_actor(a)
    lines = text.split("\n")
    assert re.fullmatch(r"L1\.W \(600 x 32\), scale 1e-?\d+: one line per row \(row i\)", lines[0])
    assert lines[1].startswith("0: ") and lines[600].startswith("599: ")
    assert lines[601] == ""
    assert re.fullmatch(r"L1\.b \(32\), scale 1e-?\d+:", lines[602])
    assert len(lines[603].split()) == 32
    assert text.endswith("\n") and not text.endswith("\n\n")
    assert [m.group(1) for m in re.finditer(r"^(L\d\.[Wb]) \(", text, re.M)] == ["L1.W", "L1.b", "L2.W", "L2.b", "L3.W", "L3.b"]


def test_dump_rejects_3d():
    with pytest.raises(ValueError):
        dump_weights({"t.W": np.zeros((2, 2, 2))})


# ------------------------------------------------------------------ C3
def test_dump_reader_round_trip_100_actors():
    for seed in range(100):
        a = random_actor(seed, scale=10.0 ** (seed % 7 - 3))
        q = quantize(a)
        back = read_dump(dump_actor(a))
        assert list(back) == list(q)
        for k in q:
            assert back[k].k == q[k].k
            assert np.array_equal(back[k].ints, q[k].ints)
        r = rounded(a)
        assert all(np.array_equal(dequantize(back)[k], r[k]) for k in r)


def test_reader_ignores_surrounding_text():
    a = random_actor(3)
    text = "intro\nTHE WEIGHTS\n\n" + dump_actor(a) + "\nthe end\n"
    assert np.array_equal(read_dump(text)["L2.W"].ints, quantize(a)["L2.W"].ints)


@pytest.mark.parametrize(
    "text, msg",
    [
        ("nothing here", "no tensors"),
        ("t.W (2 x 2), scale 1e-2: one line per row (row i)\n0: 1 2", "ends early"),
        ("t.W (2 x 2), scale 1e-2: one line per row (row i)\n0: 1 2\n2: 3 4\n", "must start"),
        ("t.W (2 x 2), scale 1e-2: one line per row (row i)\n0: 1 2\n1: 3\n", "expected 2"),
        ("t.b (2), scale 1e-2:\n1 x\n", "invalid literal"),
        ("t.b (2), scale 1e-2:", "missing"),
        ("t.b (1), scale 1e-2:\n1\n\nt.b (1), scale 1e-2:\n1\n", "twice"),
    ],
)
def test_reader_errors(text, msg):
    with pytest.raises(DumpError, match=msg):
        read_dump(text)


# ------------------------------------------------------------------ G1
def test_round_trip_preserves_action_distributions():
    obs = (np.random.default_rng(1).random((1000, 600)) < 0.08).astype(np.float32)
    for seed in range(5):
        a = random_actor(seed)
        r = rounded(a)
        back = dequantize(read_dump(dump_actor(a)))
        pr, pb = action_probs(r, obs), action_probs(back, obs)
        assert np.max(np.abs(pr - pb)) <= 1e-12
        # rounding really changed something, so this is not a trivial comparison
        assert not np.array_equal(action_probs(a, obs), pr)


def test_round_trip_preserves_sampled_episodes():
    a = random_actor(11)
    r = rounded(a)
    back = dequantize(read_dump(dump_actor(a)))
    lay = generate_layout("T0")
    for seed in seeds_for("eval", 100):
        t1 = run_episode(lay, seed, PolicyPlayer(r, "sampled"))
        t2 = run_episode(lay, seed, PolicyPlayer(back, "sampled"))
        assert t1.to_jsonl() == t2.to_jsonl()


# ------------------------------------------------------------------ G2
def test_empty_edits_block_changes_nothing():
    r = rounded(random_actor(12))
    parsed, after = edit_from_reply(r, "Diagnosis...\n```edits\n# nothing to change\n```\n")
    assert parsed.has_block and parsed.n_statements == 0 and parsed.n_skipped == 0
    an = analyse_edit(r, after, parsed)
    assert an.n_changed == 0 and an.l2_change == 0.0
