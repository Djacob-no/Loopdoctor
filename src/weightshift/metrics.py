"""Per-episode failure metrics (FINAL_PLAN §6.5, acceptance G1-G2).

All metrics are computed from a Trajectory's step records only, so they work
on recorded files, partial episodes and (later) policy rollouts alike.

Definitions (window W = 20 steps):

- loop index: fraction of steps whose state (robot cell, carried multiset)
  already occurred among the W states before it. The state before step 0 is
  the spawn with an empty inventory.
- idle fraction: fraction of steps that were a no-op or a bump.
- full-but-not-delivering: fraction of steps where the robot carries 3 items,
  at least one is currently deliverable, and no delivery happened in the last
  W steps (the current step included).
- wrong picks: picks of a SKU the order no longer needed (an excess item).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

from .env.constants import MAX_CARRY, NOOP, PICK
from .trajectory import StepRecord, Trajectory

LOOP_WINDOW = 20


@dataclass(frozen=True)
class EpisodeMetrics:
    steps: int
    score: int
    orders_completed: int
    items_delivered: int
    picks: int
    wrong_picks: int
    bumps: int
    loop_index: float
    idle_fraction: float
    full_not_delivering: float

    def as_dict(self) -> dict:
        return asdict(self)


def _states(start_pos: tuple[int, int], steps: Sequence[StepRecord]) -> list[tuple]:
    return [(tuple(start_pos), (0, 0, 0, 0))] + [(s.pos, s.carried) for s in steps]


def loop_index(start_pos: tuple[int, int], steps: Sequence[StepRecord], window: int = LOOP_WINDOW) -> float:
    if not steps:
        return 0.0
    states = _states(start_pos, steps)
    hits = 0
    for t in range(len(steps)):
        current = states[t + 1]
        earlier = states[max(0, t + 1 - window) : t + 1]
        if current in earlier:
            hits += 1
    return hits / len(steps)


def idle_fraction(steps: Sequence[StepRecord]) -> float:
    if not steps:
        return 0.0
    return sum(1 for s in steps if s.action == NOOP or s.bump) / len(steps)


def full_not_delivering(steps: Sequence[StepRecord], window: int = LOOP_WINDOW) -> float:
    if not steps:
        return 0.0
    last_delivery = -(10**9)
    count = 0
    for t, s in enumerate(steps):
        if s.delivered > 0:
            last_delivery = t
        full = sum(s.carried) == MAX_CARRY and s.deliverable > 0
        if full and t - last_delivery >= window:
            count += 1
    return count / len(steps)


def wrong_picks(steps: Sequence[StepRecord]) -> int:
    # A pick is wrong when, after it, the robot carries more of that SKU than the order still needs.
    return sum(1 for s in steps if s.picked >= 0 and s.carried[s.picked] > s.remaining[s.picked])


def compute_metrics(traj: Trajectory, window: int = LOOP_WINDOW) -> EpisodeMetrics:
    steps = traj.steps
    return EpisodeMetrics(
        steps=len(steps),
        score=sum(s.delta for s in steps),
        orders_completed=sum(1 for s in steps if s.completed),
        items_delivered=sum(s.delivered for s in steps),
        picks=sum(1 for s in steps if s.action == PICK and s.picked >= 0),
        wrong_picks=wrong_picks(steps),
        bumps=sum(1 for s in steps if s.bump),
        loop_index=loop_index(traj.start_pos, steps, window),
        idle_fraction=idle_fraction(steps),
        full_not_delivering=full_not_delivering(steps, window),
    )


def aggregate(metrics: Iterable[EpisodeMetrics]) -> dict[str, float]:
    """Means over episodes, plus a few rates used by the acceptance criteria."""
    ms = list(metrics)
    if not ms:
        raise ValueError("no episodes to aggregate")
    n = len(ms)
    out: dict[str, float] = {"episodes": n}
    for key in (
        "score",
        "orders_completed",
        "items_delivered",
        "picks",
        "wrong_picks",
        "bumps",
        "loop_index",
        "idle_fraction",
        "full_not_delivering",
    ):
        out[f"mean_{key}"] = sum(getattr(m, key) for m in ms) / n
    out["frac_any_delivery"] = sum(1 for m in ms if m.items_delivered > 0) / n
    out["frac_zero_delivery"] = 1.0 - out["frac_any_delivery"]
    out["frac_3plus_orders"] = sum(1 for m in ms if m.orders_completed >= 3) / n
    return out
