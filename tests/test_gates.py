"""Delivery 3, acceptance E: gate computations, and E7 (committed reports recompute exactly)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from weightshift.gates import ablate_capacity, compute_gates, report_json, report_markdown
from weightshift.policy import random_actor, save_actor
from weightshift.serialize import rank_units_by_outgoing_l2, rounded

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = sorted((ROOT / "artifacts" / "d3").glob("*/gate_report.json"))


def test_ablate_capacity_zeroes_the_four_weakest_units_per_layer():
    a = rounded(random_actor(0))
    ablated, zeroed = ablate_capacity(a)
    assert set(zeroed) == {"L1", "L2"}
    for layer, nxt in (("L1", "L2"), ("L2", "L3")):
        units = zeroed[layer]
        assert units == sorted(rank_units_by_outgoing_l2(a, layer)[:4])
        assert not ablated[f"{nxt}.W"][units].any() and not ablated[f"{layer}.b"][units].any()
    changed_units = {l: [u for u in range(32) if not np.array_equal(ablated[f"{l}.b"][u : u + 1], a[f"{l}.b"][u : u + 1])] for l in ("L1", "L2")}
    assert all(set(changed_units[l]) <= set(zeroed[l]) for l in changed_units)


def test_gates_report_structure_on_a_small_run(tmp_path):
    path = save_actor(tmp_path / "a.npz", random_actor(1))
    rep = compute_gates(path, n=24)
    assert set(rep["gates"]) == {"E1_failure", "E2_pocket", "E3_competence", "E4_capacity", "E5_rounding"}
    assert rep["pass"] == all(g["pass"] for g in rep["gates"].values())
    # a random network delivers almost nothing anywhere: competence must fail
    assert not rep["gates"]["E3_competence"]["pass"]
    assert rep["seeds"] == [3_000_000, 3_000_023]
    assert set(rep["rounded_sampled"]) == {"open", "detour", "probe", "trap", "T0"}
    md = report_markdown(rep)
    assert "E4 capacity" in md and rep["actor_sha256"] in md
    # deterministic
    assert report_json(compute_gates(path, n=24)) == report_json(rep)


@pytest.mark.gates
@pytest.mark.skipif(not ARTIFACTS, reason="no committed Delivery 3 artifacts yet")
@pytest.mark.parametrize("report_path", ARTIFACTS, ids=lambda p: p.parent.name)
def test_e7_committed_gate_reports_recompute_exactly(report_path):
    committed = json.loads(report_path.read_text(encoding="utf-8"))
    actor = report_path.parent / "actor.npz"
    again = compute_gates(actor, committed["episodes_per_condition"])
    again["actor"] = committed["actor"]
    assert report_json(again) == report_path.read_text(encoding="utf-8")


def test_e5_requires_float_greedy_t0_too():
    from weightshift.gates import e5_rounding_pass
    from weightshift.stats import Interval

    small = Interval(estimate=0.1, low=-0.2, high=0.4)  # float open score 20: margin +-0.6
    assert e5_rounding_pass(0.01, 0.0, small, 20.0)
    # sampled under 2% but greedy not: E5 must fail (float network fails E1)
    assert not e5_rounding_pass(0.01, 0.30, small, 20.0)
    assert not e5_rounding_pass(0.05, 0.0, small, 20.0)
    # a CI that excludes 0 but is small relative to the score passes (rule of 2026-11-05)
    assert e5_rounding_pass(0.0, 0.0, Interval(estimate=-0.6, low=-1.2, high=-0.1), 43.0)
    # ... and one that reaches past 3% fails, even if it contains 0
    assert not e5_rounding_pass(0.0, 0.0, Interval(estimate=0.0, low=-0.7, high=0.3), 20.0)


def test_gate_report_records_float_greedy_t0(tmp_path):
    path = save_actor(tmp_path / "a.npz", random_actor(2))
    e5 = compute_gates(path, n=16)["gates"]["E5_rounding"]
    assert {"float_T0_success_sampled", "float_T0_success_greedy"} <= set(e5)


def test_gates_cli_exit_status(tmp_path):
    from weightshift.cli import gates as cli

    path = save_actor(tmp_path / "a.npz", random_actor(3))
    # a random network fails competence, so the command must not exit 0
    assert cli.main([str(path), "--episodes", "8", "--out", str(tmp_path / "r")]) == 1
