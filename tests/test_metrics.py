"""Acceptance G1-G2: metrics on hand-built trajectories with known answers."""

from __future__ import annotations

import pytest

from weightshift.env import load_fixed_layout
from weightshift.env.constants import DOWN, DROP, NOOP, PICK, UP
from weightshift.metrics import (
    EpisodeMetrics,
    aggregate,
    compute_metrics,
    full_not_delivering,
    idle_fraction,
    loop_index,
    wrong_picks,
)
from weightshift.players.base import ActionListPlayer
from weightshift.trajectory import StepRecord, Trajectory, run_episode


def rec(t, *, action=NOOP, pos=(1, 1), carried=(0, 0, 0, 0), remaining=(1, 1, 1, 0), deliverable=0,
        delta=0, delivered=0, completed=False, bump=False, picked=-1):
    return StepRecord(t, action, pos, carried, remaining, deliverable, delta, delivered, completed, bump, picked)


def traj(steps, start=(1, 1)):
    return Trajectory(header={"spawn": list(start)}, steps=list(steps), end={})


# --------------------------------------------------------------------- G1
def test_g1_score_orders_and_deliveries_from_steps():
    steps = [
        rec(0),
        rec(1, action=DROP, delivered=2, delta=2),
        rec(2, action=DROP, delivered=1, delta=1 + 5, completed=True),
        rec(3, action=DROP, delivered=3, delta=3 + 5, completed=True),
    ]
    m = compute_metrics(traj(steps))
    assert (m.score, m.orders_completed, m.items_delivered, m.steps) == (16, 2, 6, 4)


def test_g1_metrics_agree_with_the_game_on_a_real_episode():
    t = run_episode(load_fixed_layout("T0"), 1_000_000, ActionListPlayer([NOOP] * 10 + [DOWN, PICK] * 30))
    m = compute_metrics(t)
    assert m.score == t.end["score"]
    assert m.orders_completed == t.end["orders_completed"]
    assert m.items_delivered == t.end["items_delivered"]


# --------------------------------------------------------------------- G2
def test_loop_index_two_cell_oscillation():
    # start a=(1,1); steps alternate b, a, b, a, b, a -> only the first step is new.
    cells = [(1, 2), (1, 1)] * 3
    steps = [rec(i, pos=p) for i, p in enumerate(cells)]
    assert loop_index((1, 1), steps) == pytest.approx(5 / 6)


def test_loop_index_counts_carried_state():
    # Same cell every step, but the inventory changes on steps 0 and 1: those are new states.
    steps = [rec(0, carried=(1, 0, 0, 0)), rec(1, carried=(2, 0, 0, 0)), rec(2, carried=(2, 0, 0, 0))]
    assert loop_index((1, 1), steps) == pytest.approx(1 / 3)


def test_loop_index_window_is_exactly_20_states():
    # Walk through 21 distinct cells, then revisit the cell from 20 states back and from 21 states back.
    distinct = [(r, c) for r in range(1, 9) for c in range(1, 9)][:22]
    start, walk = distinct[0], distinct[1:22]  # 21 steps, all new
    steps = [rec(i, pos=p) for i, p in enumerate(walk)]
    # state k=21 is walk[20]; revisiting the state 20 back (k=2 -> walk[1]) is inside the window
    inside = steps + [rec(21, pos=walk[1])]
    assert loop_index(start, inside) == pytest.approx(1 / 22)
    # revisiting the state 21 back (k=1 -> walk[0]) is outside the window
    outside = steps + [rec(21, pos=walk[0])]
    assert loop_index(start, outside) == 0.0


def test_idle_fraction_counts_noops_and_bumps_only():
    steps = [
        rec(0, action=NOOP),
        rec(1, action=UP, bump=True),
        rec(2, action=UP),
        rec(3, action=PICK),  # a pick that did nothing is not idle by this definition
        rec(4, action=DROP),
    ]
    assert idle_fraction(steps) == pytest.approx(2 / 5)


def test_full_not_delivering_window():
    full = dict(carried=(2, 1, 0, 0), deliverable=1)
    steps = [rec(t, **full) for t in range(25)]
    steps[3] = rec(3, delivered=1, delta=1, **full)
    # Steps 0-2 have no delivery yet; steps 3..22 have the step-3 delivery within the last 20 steps;
    # steps 23 and 24 are 20+ steps after it.
    assert full_not_delivering(steps) == pytest.approx(5 / 25)


def test_full_not_delivering_needs_full_and_deliverable():
    steps = [
        rec(0, carried=(1, 1, 0, 0), deliverable=2),  # not full
        rec(1, carried=(0, 0, 0, 3), deliverable=0),  # full of junk
        rec(2, carried=(1, 1, 1, 0), deliverable=1),  # counts
    ]
    assert full_not_delivering(steps) == pytest.approx(1 / 3)


def test_wrong_picks():
    steps = [
        rec(0, action=PICK, picked=0, carried=(1, 0, 0, 0), remaining=(1, 0, 0, 0)),  # needed
        rec(1, action=PICK, picked=0, carried=(2, 0, 0, 0), remaining=(1, 0, 0, 0)),  # excess -> wrong
        rec(2, action=PICK, picked=3, carried=(2, 0, 0, 1), remaining=(1, 0, 0, 0)),  # not in order -> wrong
        rec(3, action=PICK, picked=-1, carried=(2, 0, 0, 1)),  # failed pick
    ]
    assert wrong_picks(steps) == 2
    assert compute_metrics(traj(steps)).picks == 3


def test_empty_trajectory_metrics_are_zero():
    m = compute_metrics(traj([]))
    assert m.steps == 0 and m.loop_index == 0.0 and m.idle_fraction == 0.0 and m.full_not_delivering == 0.0


def test_aggregate():
    a = EpisodeMetrics(200, 10, 1, 5, 5, 0, 0, 0.1, 0.2, 0.0)
    b = EpisodeMetrics(200, 0, 0, 0, 2, 1, 4, 0.5, 0.6, 0.3)
    out = aggregate([a, b])
    assert out["episodes"] == 2 and out["mean_score"] == 5
    assert out["frac_any_delivery"] == 0.5 and out["frac_zero_delivery"] == 0.5
    assert out["mean_loop_index"] == pytest.approx(0.3)
    with pytest.raises(ValueError):
        aggregate([])
