"""Summarise training runs from their metrics.jsonl files (for TRAINING_LOG.md).

    uv run python scripts/summarize_runs.py                 # every run under runs/train
    uv run python scripts/summarize_runs.py attempt02 --every 8

Shows the checkpoint-time evaluation (128 select-seed episodes per family,
sampled, float network): success rate and mean score per family.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FAMS = ("open", "detour", "probe", "trap", "T0")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("filter", nargs="?", default="")
    ap.add_argument("--root", default="runs/train")
    ap.add_argument("--every", type=int, default=0, help="also show every Nth checkpoint")
    args = ap.parse_args(argv)
    head = f"{'run':<34} {'step':>11} {'steps/s':>8}  " + "  ".join(f"{f:>13}" for f in FAMS)
    print(head)
    for rd in sorted(Path(args.root).glob(f"*{args.filter}*")):
        mf = rd / "metrics.jsonl"
        if not mf.exists():
            continue
        rows = [json.loads(x) for x in mf.read_text(encoding="utf-8").splitlines() if x.strip()]
        pick = rows[:: args.every] if args.every else []
        if rows and (not pick or pick[-1] is not rows[-1]):
            pick.append(rows[-1])
        for r in pick:
            ev = r.get("eval", {})
            cells = "  ".join(
                f"{100 * ev[f]['success']:5.1f}% {ev[f]['mean_score']:6.1f}" if f in ev else f"{'-':>13}" for f in FAMS
            )
            print(f"{rd.name:<34} {r['global_step']:>11,} {r.get('steps_per_s') or 0:>8,.0f}  {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
