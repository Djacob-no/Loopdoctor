"""Run the Gate A checks on one or more actors (DELIVERY_3_ACCEPTANCE §4 E).

    uv run python -m weightshift.cli.gates runs/train/<run>/checkpoints/step0010000000.npz [more.npz ...]
    uv run python -m weightshift.cli.gates ... --out reports/

Writes gate_report.json and gate_report.md next to each actor (or in --out),
prints the summary, and checks E6: at least 2 of the given actors pass.
Exit status: with one actor, 0 only if it passes E1-E5; with several, 0 only
if E6 passes (at least 2 of them pass E1-E5).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..gates import compute_gates, report_json, report_markdown


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("actors", nargs="+")
    ap.add_argument("--out", default=None, help="directory for the reports (default: next to each actor)")
    ap.add_argument("--episodes", type=int, default=1024)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    passed = 0
    for a in args.actors:
        path = Path(a)
        report = compute_gates(path, args.episodes)
        out = Path(args.out) / path.stem if args.out else path.parent
        out.mkdir(parents=True, exist_ok=True)
        (out / "gate_report.json").write_text(report_json(report), encoding="utf-8", newline="\n")
        md = report_markdown(report)
        (out / "gate_report.md").write_text(md, encoding="utf-8", newline="\n")
        print(md)
        passed += report["pass"]
    if len(args.actors) > 1:
        print(f"E6 second seed: {passed} of {len(args.actors)} actors pass (need >= 2): {'pass' if passed >= 2 else 'FAIL'}")
        return 0 if passed >= 2 else 1
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
