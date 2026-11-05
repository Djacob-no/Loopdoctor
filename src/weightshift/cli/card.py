"""Build the failure card for an actor (FINAL_PLAN §6.5).

    uv run python -m weightshift.cli.card artifacts/d3/<sha12>/actor.npz --out artifacts/d3/<sha12>/failure_card.md
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

from ..diagnose.failure_card import build_card


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("actor")
    ap.add_argument("--out", default=None, help="write the card here (and the numbers to a .json next to it)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    text, data = build_card(args.actor)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8", newline="\n")
        out.with_suffix(".json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {out} ({len(text):,} characters)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
