"""Train PPO from a config (DELIVERY_3_ACCEPTANCE §2). Run natively on Windows (README).

    uv run python -m weightshift.cli.train --config configs/attempt01.toml            # all seeds, in parallel
    uv run python -m weightshift.cli.train --config configs/attempt01.toml --seed 0   # one seed

Runs go to runs/train/<name>-<config hash>-s<seed>/ and resume from full.pt
if interrupted. Each checkpoint (every 250k steps) is evaluated on 128
select-seed episodes per family.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from ..gates import quick_eval
from ..train.config import load_config
from ..train.ppo import train


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None, help="one seed (default: every seed in the config, in parallel)")
    ap.add_argument("--out", default="runs/train")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing full.pt and start over")
    ap.add_argument("--no-eval", action="store_true", help="skip checkpoint evaluation")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.seed is None:
        t0 = time.perf_counter()
        procs = []
        for seed in cfg.seeds:
            cmd = [sys.executable, "-m", "weightshift.cli.train", "--config", args.config, "--seed", str(seed), "--out", args.out]
            cmd += ["--fresh"] if args.fresh else []
            cmd += ["--no-eval"] if args.no_eval else []
            procs.append((seed, subprocess.Popen(cmd)))
        codes = {seed: p.wait() for seed, p in procs}
        print(f"{cfg.name} ({cfg.config_hash}): exit codes {codes}, {time.perf_counter() - t0:,.0f} s wall")
        return 0 if all(c == 0 for c in codes.values()) else 1

    # one seed: checkpoint evaluation runs in this process. Starting worker processes
    # costs more than the 640 short episodes themselves, especially on Windows.
    os.environ.setdefault("WEIGHTSHIFT_WORKERS", "1")
    eval_fn = None if args.no_eval else (lambda actor_path, run_dir: quick_eval(actor_path, cfg.eval_episodes))
    rd = train(cfg, args.seed, Path(args.out), resume=not args.fresh, eval_fn=eval_fn)
    print(f"done: {rd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
