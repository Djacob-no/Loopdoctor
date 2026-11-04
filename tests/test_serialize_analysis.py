"""Delivery 2, acceptance H: edit analysis and unit ablation."""

from __future__ import annotations

import numpy as np
import pytest

from weightshift.policy import ACTOR_SPEC, action_probs, random_actor
from weightshift.serialize import analyse_edit, edit_from_reply, rank_units_by_outgoing_l2, rounded, zero_units


def block(*lines):
    return "```edits\n" + "\n".join(lines) + "\n```\n"


# ------------------------------------------------------------------ H1
def test_known_counts_and_l2():
    r = rounded(random_actor(0))
    before = r["L2.W"][3, 4]
    parsed, after = edit_from_reply(r, block("L2.W[3, 4] += 3", "L3.b[6] = 0", "L3.b[6] += 4", "L2.b[0] = nonsense"))
    an = analyse_edit(r, after, parsed)
    assert after["L2.W"][3, 4] == before + 3
    assert an.n_changed == 2 and an.n_params == 20_519
    assert an.frac_changed == pytest.approx(2 / 20_519)
    assert an.l2_change == pytest.approx(np.hypot(3.0, 4.0 - r["L3.b"][6]))
    assert an.changed_per_tensor == {"L1.W": 0, "L1.b": 0, "L2.W": 1, "L2.b": 0, "L3.W": 0, "L3.b": 1}
    assert an.tensors_changed == ("L2.W", "L3.b")
    assert an.units_touched == {"L1": (), "L2": (4,)}
    assert an.outputs_touched == (6,)
    assert an.n_statements == 3 and an.n_skipped == 1


def test_setting_a_value_to_itself_is_not_a_change():
    r = rounded(random_actor(1))
    v = float(r["L1.W"][10, 2])  # repr of a Python float is a grammar-legal, exact literal
    parsed, after = edit_from_reply(r, block(f"L1.W[10, 2] = {v!r}", "L1.W[10, 2] *= 1"))
    assert parsed.n_statements == 2 and parsed.n_skipped == 0  # both really applied
    assert after["L1.W"][10, 2] == v
    assert analyse_edit(r, after).n_changed == 0


def test_takeover_counts_units_like_exp4():
    """Zeroing a unit's outgoing row crosses every column of the next layer;
    those columns do not count as taken-over units."""
    r = rounded(random_actor(2))
    parsed, after = edit_from_reply(
        r,
        block(
            "L1.W[:, 30:32] = 0",  # take over L1 units 30, 31
            "L1.W[5, 30] = 80",
            "L1.b[31] = -2",
            "L2.W[30:32, :] = 0",  # clear their outgoing rows (crosses every L2 column)
            "L2.W[:, 29] = 0",  # take over L2 unit 29 ...
            "L2.W[30, 29] = 10",  # ... wired from the new L1 unit
            "L3.W[29, 0:2] = [-5, 5]",
        ),
    )
    an = analyse_edit(r, after, parsed)
    assert an.units_touched == {"L1": (30, 31), "L2": (29,)}
    assert an.units_per_layer == {"L1": 2, "L2": 1}
    assert an.outputs_touched == ()  # only rows of taken-over units changed in L3.W
    assert set(an.to_json()) >= {"n_changed", "frac_changed", "l2_change", "units_per_layer", "n_skipped"}


def test_analysis_rejects_mismatched_networks():
    a = random_actor(0)
    b = dict(a)
    b["L1.b"] = np.zeros(31)
    with pytest.raises(ValueError):
        analyse_edit(a, b)
    with pytest.raises(ValueError):
        analyse_edit(a, {k: a[k] for k in reversed(list(a))})


# ------------------------------------------------------------------ H2
@pytest.mark.parametrize("layer, unit", [("L1", 0), ("L1", 17), ("L2", 31)])
def test_zeroed_unit_is_irrelevant(layer, unit):
    a = random_actor(3)
    z = zero_units(a, layer, [unit])
    obs = (np.random.default_rng(0).random((200, 600)) < 0.1).astype(np.float32)
    base = action_probs(z, obs)
    # whatever the zeroed unit's incoming weights become, the output does not move
    for seed in range(5):
        w = {k: v.copy() for k, v in z.items()}
        rng = np.random.default_rng(seed)
        w[f"{layer}.W"][:, unit] = rng.normal(0, 5, size=w[f"{layer}.W"].shape[0])
        w[f"{layer}.b"][unit] = rng.normal(0, 5)
        assert np.array_equal(action_probs(w, obs), base)
    # zero_units zeroed exactly the column, the bias and the outgoing row
    nxt = {"L1": "L2", "L2": "L3"}[layer]
    assert not z[f"{layer}.W"][:, unit].any() and z[f"{layer}.b"][unit] == 0 and not z[f"{nxt}.W"][unit].any()
    assert sum(int((z[k] != a[k]).sum()) for k in a) == int(
        (a[f"{layer}.W"][:, unit] != 0).sum() + (a[f"{layer}.b"][unit] != 0) + (a[f"{nxt}.W"][unit] != 0).sum()
    )


def test_zero_units_does_not_modify_input_and_validates():
    a = random_actor(4)
    before = {k: v.copy() for k, v in a.items()}
    zero_units(a, "L1", [0, 1, 2, 3])
    assert all(np.array_equal(a[k], before[k]) for k in a)
    with pytest.raises(ValueError, match="output layer"):
        zero_units(a, "L3", [0])
    with pytest.raises(ValueError, match="unknown layer"):
        zero_units(a, "L4", [0])
    with pytest.raises(ValueError, match="32 units"):
        zero_units(a, "L2", [32])


def test_rank_units_by_outgoing_l2():
    a = {n: np.zeros(s) for n, s in ACTOR_SPEC}
    norms = np.arange(32, 0, -1, dtype=np.float64)  # unit 0 largest ... unit 31 smallest
    norms[[5, 9]] = 0.5  # a tie, below everything else
    a["L2.W"][:, 0] = norms
    ranked = rank_units_by_outgoing_l2(a, "L1")
    assert ranked[:3] == [5, 9, 31]  # tie broken by lower index
    assert ranked[-1] == 0 and sorted(ranked) == list(range(32))
    assert rank_units_by_outgoing_l2(a, "L2") == list(range(32))  # all zero: index order
