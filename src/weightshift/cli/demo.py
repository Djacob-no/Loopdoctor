"""Watch a scripted player (acceptance I3).

    uv run python -m weightshift.cli.demo --player oracle --layout T0
    uv run python -m weightshift.cli.demo --player greedy_straight --layout T0 --delay 0.05
"""

from __future__ import annotations

import argparse
import sys
import time

from ..env.game import Game
from ..env.render import render_frame
from ..metrics import compute_metrics
from ..players.scripted import PLAYERS, make_player
from ..trajectory import Recorder
from .common import LAYOUT_CHOICES, format_metrics, resolve_layout


def main(argv: list[str] | None = None, out=None) -> int:
    out = out or sys.stdout
    ap = argparse.ArgumentParser(description="Watch a scripted WeightShift player.")
    ap.add_argument("--player", default="oracle", choices=sorted(PLAYERS))
    ap.add_argument("--layout", default="T0", choices=LAYOUT_CHOICES)
    ap.add_argument("--seed", type=int, default=1_000_000, help="episode seed (also the layout seed for families)")
    ap.add_argument("--role", choices=["train", "diagnosis", "eval"], default=None)
    ap.add_argument("--every", type=int, default=1, help="print every Nth frame")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to pause between printed frames")
    ap.add_argument("--final-only", action="store_true", help="print only the last frame and the metrics")
    ap.add_argument("--clear", action="store_true", help="clear the screen between frames (animation)")
    ap.add_argument("--record", default=None, help="write the episode to this JSONL file")
    args = ap.parse_args(argv)
    if args.every < 1:
        ap.error("--every must be >= 1")

    layout = resolve_layout(args.layout, args.seed, args.role)
    game = Game(layout, args.seed, role=args.role)
    player = make_player(args.player)
    rec = Recorder(game, player.name)
    player.reset(game)

    def show(last_action=None, last_reward=None):
        prefix = "\x1b[2J\x1b[H" if args.clear else ""
        out.write(prefix + render_frame(game, last_action, last_reward) + "\n\n")
        out.flush()
        if args.delay:
            time.sleep(args.delay)

    out.write(f"player {player.name}   layout {layout.name}   episode seed {args.seed}\n\n")
    if not args.final_only:
        show()
    action = reward = None
    while not game.done:
        action = player.act(game)
        r = game.step(action)
        rec.record(action, r)
        reward = r.reward
        if not args.final_only and (game.t % args.every == 0 or game.done):
            show(action, reward)
    traj = rec.finish()
    if args.final_only:
        show(action, reward)
    out.write(format_metrics(compute_metrics(traj)) + "\n")
    if args.record:
        out.write(f"recorded to {traj.write(args.record)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
