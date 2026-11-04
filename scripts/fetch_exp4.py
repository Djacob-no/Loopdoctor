"""Download the published Exp 4 files used by the golden test (DELIVERY_2_ACCEPTANCE §2, A2).

    uv run python scripts/fetch_exp4.py            # into fixtures/exp4/ (gitignored)
    uv run python scripts/fetch_exp4.py --check    # verify only, download nothing

Source: github.com/RikiyaT/lm_understand-logs at a pinned commit, CC BY 4.0
(see ATTRIBUTION.md). Every file is checked against the SHA-256 below. A file
that is already present with the right hash is not downloaded again; a file
with the wrong hash is an error (use --force to replace it).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

REPO = "RikiyaT/lm_understand-logs"
COMMIT = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
BASE = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/exp4_fix_car"

# local name -> (path under exp4_fix_car/, sha256)
FILES: dict[str, tuple[str, str]] = {
    "ppo_final_seed0.npz": (
        "networks/ppo_final_seed0.npz",
        "b0cdbccd5b4751d5a6dc4cb0deb18a036e6d08d71ad34802cea44dd26c6f4d11",
    ),
    "edited_actor.npz": (
        "networks/edited_actor.npz",
        "610d62dbacdb16389f698cdf7d3bf39b185e410b8c6e03a1ac4833768066060a",
    ),
    "prompt.txt": (
        "runs/oneshot_repair/reply/prompt.txt",
        "2ed66051f4fa554d43d0163eb48e4256dc80aef78d875a218826beefeb7d0559",
    ),
    "answer.md": (
        "runs/oneshot_repair/reply/answer.md",
        "dc295adb701bb943a6a324fd7fd4aeb97bb69a2d8fb73b9472e3a9371275f959",
    ),
    "oneshot_result.json": (
        "eval/oneshot_result.json",
        "d9e57db254079b776ee574ba94bfc061ca1e4ce199fad316dea6f8faa2049330",
    ),
}

DEFAULT_DEST = Path(__file__).resolve().parents[1] / "fixtures" / "exp4"


class HashMismatch(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_ok(path: Path, sha: str) -> bool:
    return path.is_file() and sha256_of(path) == sha


def missing_or_bad(dest: Path = DEFAULT_DEST) -> list[str]:
    """Names of files that are absent or fail their hash (empty = ready for the golden test)."""
    return [name for name, (_, sha) in FILES.items() if not file_ok(dest / name, sha)]


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 - fixed https URL
        return r.read()


def fetch(
    dest: Path = DEFAULT_DEST,
    *,
    force: bool = False,
    download: Callable[[str], bytes] = _download,
    log=print,
) -> list[str]:
    """Download missing files; returns the names downloaded. Raises HashMismatch on a bad file."""
    dest.mkdir(parents=True, exist_ok=True)
    fetched = []
    for name, (rel, sha) in FILES.items():
        path = dest / name
        if path.exists():
            if sha256_of(path) == sha:
                log(f"ok        {name}")
                continue
            if not force:
                raise HashMismatch(f"{path} exists but its SHA-256 does not match; rerun with --force to replace it")
        data = download(f"{BASE}/{rel}")
        got = hashlib.sha256(data).hexdigest()
        if got != sha:
            raise HashMismatch(f"{name}: downloaded SHA-256 {got} != pinned {sha}")
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        fetched.append(name)
        log(f"fetched   {name} ({len(data):,} bytes)")
    return fetched


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--force", action="store_true", help="replace files whose hash does not match")
    ap.add_argument("--check", action="store_true", help="only verify, download nothing")
    args = ap.parse_args(argv)
    if args.check:
        bad = missing_or_bad(args.dest)
        for name in bad:
            print(f"missing or wrong hash: {name}", file=sys.stderr)
        if not bad:
            print(f"all {len(FILES)} Exp 4 files present and verified in {args.dest}")
        return 1 if bad else 0
    try:
        fetch(args.dest, force=args.force)
    except HashMismatch as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"Exp 4 files ready in {args.dest} (commit {COMMIT[:7]}, CC BY 4.0, see ATTRIBUTION.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
