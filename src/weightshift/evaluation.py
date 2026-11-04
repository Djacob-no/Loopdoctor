"""Run a scripted player over many seeded episodes and collect metrics.

For generated families the same seed is used as layout seed and episode
seed; T0 is a fixed layout, so only the episode seed varies. Results are
returned in seed order and are identical whether run serially or in worker
processes.

Set WEIGHTSHIFT_WORKERS to override the worker count (1 = serial).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from .env.generate import generate_layout
from .metrics import EpisodeMetrics, compute_metrics
from .players.scripted import make_player
from .trajectory import run_episode


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    metrics: EpisodeMetrics


def run_one(player_name: str, family: str, seed: int, role: str | None = None) -> EpisodeResult:
    layout = generate_layout(family, seed, role=role)
    traj = run_episode(layout, seed, make_player(player_name), role=role)
    return EpisodeResult(seed, compute_metrics(traj))


def _run_chunk(args: tuple[str, str, list[int], str | None]) -> list[EpisodeResult]:
    player_name, family, seeds, role = args
    return [run_one(player_name, family, s, role) for s in seeds]


def evaluate(
    player_name: str,
    family: str,
    seeds: list[int],
    *,
    role: str | None = None,
    workers: int | None = None,
) -> list[EpisodeResult]:
    """Metrics for `player_name` on `family`, one episode per seed (in seed order)."""
    if workers is None:
        env = os.environ.get("WEIGHTSHIFT_WORKERS")
        if env is not None:
            workers = int(env)
        else:
            workers = min(16, os.cpu_count() or 1) if len(seeds) >= 200 else 1
    if workers <= 1:
        return _run_chunk((player_name, family, list(seeds), role))
    size = -(-len(seeds) // workers)
    chunks = [(player_name, family, list(seeds[i : i + size]), role) for i in range(0, len(seeds), size)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_run_chunk, chunks))
    return [r for part in parts for r in part]
