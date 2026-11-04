"""Acceptance F1-F4 and G3: the scripted players calibrate the game itself.

Each condition is 1,000 episodes on eval-range seeds. Results are computed
once per test session and shared between tests.
"""

from __future__ import annotations

from functools import lru_cache

import pytest

from weightshift.env import Game, generate_layout, seeds_for
from weightshift.env.constants import MAX_CARRY
from weightshift.evaluation import evaluate
from weightshift.metrics import aggregate
from weightshift.players.scripted import OraclePlayer, RandomPlayer, make_player

N = 1000
SEEDS = seeds_for("eval", N)


@lru_cache(maxsize=None)
def results(player: str, family: str):
    return tuple(evaluate(player, family, SEEDS, role="eval"))


def agg(player: str, family: str) -> dict:
    return aggregate(r.metrics for r in results(player, family))


# ---------------------------------------------------------------------- F1
@pytest.mark.parametrize("family", ["open", "T0", "probe"])
def test_f1_oracle_completes_3_orders_in_95_percent(family):
    a = agg("oracle", family)
    assert a["frac_3plus_orders"] >= 0.95, a


# ---------------------------------------------------------------------- F2
def test_f2_random_scores_below_10_percent_of_oracle():
    assert agg("random", "open")["mean_score"] < 0.10 * agg("oracle", "open")["mean_score"]


def test_f2_random_player_never_violates_invariants():
    # B3 already checks uniform-random actions; this runs the RandomPlayer itself.
    for seed in SEEDS:
        layout = generate_layout("open", seed)
        g = Game(layout, seed)
        p = RandomPlayer()
        p.reset(g)
        last = 0
        while not g.done:
            g.step(p.act(g))
            assert not layout.is_obstacle(g.p)
            assert 0 <= g.n_carried <= MAX_CARRY
            assert g.score >= last
            last = g.score


# ---------------------------------------------------------------------- F3
def test_f3_greedy_never_delivers_on_t0_while_oracle_does():
    greedy = results("greedy_straight", "T0")
    oracle = results("oracle", "T0")
    zero = [g.metrics.items_delivered == 0 for g in greedy]
    assert sum(zero) / N >= 0.99
    for g, o, z in zip(greedy, oracle, zero):
        assert g.seed == o.seed
        if z:
            assert o.metrics.items_delivered > 0, f"seed {o.seed}: the oracle also failed"


# ---------------------------------------------------------------------- F4
def test_f4_greedy_delivers_on_probe_layouts():
    assert agg("greedy_straight", "probe")["frac_any_delivery"] >= 0.90


# ---------------------------------------------------------------------- G3
def test_g3_metrics_see_the_trap():
    stuck = agg("greedy_straight", "T0")
    healthy = agg("oracle", "open")
    for key in ("mean_loop_index", "mean_idle_fraction"):
        assert stuck[key] >= 3 * healthy[key], (key, stuck[key], healthy[key])
        # Guard against a trivial pass when the healthy value is ~0.
        assert stuck[key] >= healthy[key] + 0.2, (key, stuck[key], healthy[key])


# ------------------------------------------------------------ player sanity
def test_oracle_never_picks_wrong_items():
    assert all(r.metrics.wrong_picks == 0 for r in results("oracle", "open"))


def test_evaluation_is_identical_serial_and_parallel():
    seeds = SEEDS[:40]
    a = evaluate("oracle", "trap", seeds, workers=1)
    b = evaluate("oracle", "trap", seeds, workers=4)
    assert a == b


def test_players_are_deterministic_per_seed():
    from weightshift.trajectory import run_episode

    lay = generate_layout("open", 5)
    for name in ("random", "greedy_straight", "oracle"):
        a = run_episode(lay, 2_000_123, make_player(name)).to_jsonl()
        b = run_episode(lay, 2_000_123, make_player(name)).to_jsonl()
        assert a == b, name


def test_make_player_rejects_unknown_names():
    with pytest.raises(ValueError):
        make_player("alphago")
    assert isinstance(make_player("oracle"), OraclePlayer)
