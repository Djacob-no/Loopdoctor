"""Delivery 3, acceptance B1-B2: the `detour` layout family."""

from __future__ import annotations

from functools import lru_cache

from weightshift.env import generate_layout, manhattan, rc, seeds_for
from weightshift.env.layout import monotone_path_exists, shortest_path
from weightshift.evaluation import evaluate
from weightshift.metrics import aggregate

from test_layouts import N_LAYOUTS, _check_e1


def _wall_is_one_straight_segment(layout) -> bool:
    cells = [rc(p) for p in layout.rack_cells]
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) == 1:
        span = sorted(cols)
    elif len(cols) == 1:
        span = sorted(rows)
    else:
        return False
    return span == list(range(span[0], span[0] + len(span))) and 4 <= len(span) <= 6


def test_b1_detour_layouts():
    for seed in range(N_LAYOUTS):
        layout = generate_layout("detour", seed)
        _check_e1(layout)
        assert layout.family == "detour"
        assert _wall_is_one_straight_segment(layout), layout.name  # no U around the dock
        for s in layout.spawns:
            assert not monotone_path_exists(layout, s, layout.dock), f"{layout.name}: straight line open from {rc(s)}"
            path = shortest_path(layout, s, layout.dock)
            dists = [manhattan(p, layout.dock) for p in path]
            assert any(b > a for a, b in zip(dists, dists[1:])), f"{layout.name}: never steps away from the dock"


def test_b1_detour_uses_all_orientations_and_varies():
    docks = {rc(generate_layout("detour", s).dock) for s in range(400)}
    sides = {"top" if r == 1 else "bottom" if r == 8 else "left" if c == 1 else "right" for r, c in docks}
    assert sides == {"top", "bottom", "left", "right"}
    assert len({generate_layout("detour", s).cells for s in range(50)}) > 40


@lru_cache(maxsize=None)
def _agg(player: str) -> dict:
    return aggregate(r.metrics for r in evaluate(player, "detour", seeds_for("select", 1000), role="select"))


def test_b2_oracle_solves_detour():
    assert _agg("oracle")["frac_3plus_orders"] >= 0.95


def test_b2_greedy_straight_fails_on_detour():
    """The family really needs a detour: a straight walker almost never delivers."""
    a = _agg("greedy_straight")
    assert a["frac_any_delivery"] <= 0.10, a
