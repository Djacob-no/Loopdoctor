"""Environment step throughput on one core (acceptance H2).

    uv run python scripts/bench_steps.py            # ~3 s, open layouts
    uv run python scripts/bench_steps.py --family T0 --seconds 5 --min 10000

Measures WeightShiftEnv.step() including observation building, with random
actions; resets (new layout and seed) every 200 steps. Exits non-zero if the
rate is below --min.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from weightshift.env.gym_env import WeightShiftEnv


def bench(family: str = "open", seconds: float = 3.0, seed: int = 0) -> tuple[float, int]:
    env = WeightShiftEnv(family=family)
    rng = np.random.default_rng(seed)
    actions = rng.integers(0, 7, size=1 << 16).tolist()
    episode = 0
    env.reset(seed=episode)
    steps = 0
    t0 = time.perf_counter()
    deadline = t0 + seconds
    while True:
        for _ in range(1000):
            _, _, _, truncated, _ = env.step(actions[steps & 0xFFFF])
            steps += 1
            if truncated:
                episode += 1
                env.reset(seed=episode)
        if time.perf_counter() >= deadline:
            break
    elapsed = time.perf_counter() - t0
    return steps / elapsed, steps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", default="open")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--min", type=float, default=10_000.0, help="fail below this many steps/s")
    ap.add_argument("--no-pin", action="store_true", help="do not pin the process to one CPU core")
    args = ap.parse_args(argv)
    if not args.no_pin and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[0]})
    rate, steps = bench(args.family, args.seconds)
    print(f"{args.family}: {rate:,.0f} env steps/s on one core ({steps:,} steps incl. observation building)")
    if rate < args.min:
        print(f"FAIL: below the {args.min:,.0f} steps/s target", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
