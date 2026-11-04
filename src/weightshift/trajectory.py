"""Episode recording, JSONL serialization and replay (acceptance I2).

A trajectory file is JSON Lines:

    {"type": "header", ...}        layout ASCII, seeds, spawn, first order, code version
    {"type": "step", ...} x T      action and the state after it
    {"type": "end", ...}           final score, orders, items

Everything is integers or strings and keys are sorted, so the same episode
always serializes to the same bytes (acceptance C1). The header holds the
full layout ASCII, so a file replays without regenerating the layout.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import __version__
from .env.constants import EPISODE_LEN
from .env.game import Game, StepResult
from .env.layout import Layout
from .env.seeds import SeedRole

FORMAT_VERSION = 1


@lru_cache(maxsize=1)
def code_version() -> str:
    """Package version plus the git commit when available, e.g. 0.1.0+65f4806."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        return f"{__version__}+{sha}" if sha else __version__
    except (OSError, subprocess.SubprocessError):
        return __version__


@dataclass(frozen=True)
class StepRecord:
    t: int  # step index, 0-based
    action: int
    pos: tuple[int, int]  # robot cell after the step
    carried: tuple[int, int, int, int]  # per SKU, after the step
    remaining: tuple[int, int, int, int]  # current order's remaining, after the step
    deliverable: int  # carried items the current order accepts, after the step
    delta: int  # score delta
    delivered: int
    completed: bool
    bump: bool
    picked: int  # SKU or -1

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["pos"] = list(self.pos)
        d["carried"] = list(self.carried)
        d["remaining"] = list(self.remaining)
        d["type"] = "step"
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> StepRecord:
        return cls(
            t=d["t"],
            action=d["action"],
            pos=tuple(d["pos"]),
            carried=tuple(d["carried"]),
            remaining=tuple(d["remaining"]),
            deliverable=d["deliverable"],
            delta=d["delta"],
            delivered=d["delivered"],
            completed=d["completed"],
            bump=d["bump"],
            picked=d["picked"],
        )


@dataclass
class Trajectory:
    header: dict[str, Any]
    steps: list[StepRecord] = field(default_factory=list)
    end: dict[str, Any] = field(default_factory=dict)

    # convenience
    @property
    def score(self) -> int:
        return self.end["score"]

    @property
    def actions(self) -> list[int]:
        return [s.action for s in self.steps]

    @property
    def start_pos(self) -> tuple[int, int]:
        return tuple(self.header["spawn"])

    def layout(self) -> Layout:
        h = self.header
        return Layout.from_ascii(
            "\n".join(h["layout"]), name=h["layout_name"], family=h["layout_family"], layout_seed=h["layout_seed"]
        )

    # serialization
    def to_jsonl(self) -> str:
        lines = [self.header] + [s.to_json() for s in self.steps] + [self.end]
        return "".join(json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n" for x in lines)

    @classmethod
    def from_jsonl(cls, text: str) -> Trajectory:
        rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        if not rows or rows[0].get("type") != "header":
            raise ValueError("trajectory must start with a header line")
        if rows[0].get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported trajectory format {rows[0].get('format')!r}")
        if rows[-1].get("type") != "end":
            raise ValueError("trajectory must end with an end line")
        steps = [StepRecord.from_json(r) for r in rows[1:-1]]
        if any(r.get("type") != "step" for r in rows[1:-1]):
            raise ValueError("unexpected line type in trajectory body")
        return cls(header=rows[0], steps=steps, end=rows[-1])

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_jsonl(), encoding="utf-8")
        return p

    @classmethod
    def read(cls, path: str | Path) -> Trajectory:
        return cls.from_jsonl(Path(path).read_text(encoding="utf-8"))


class Recorder:
    """Builds a Trajectory while a game is played (by a player, a human or a policy)."""

    def __init__(self, game: Game, player_name: str):
        self.game = game
        lay = game.layout
        self.header: dict[str, Any] = {
            "type": "header",
            "format": FORMAT_VERSION,
            "code_version": code_version(),
            "player": player_name,
            "layout_name": lay.name,
            "layout_family": lay.family,
            "layout_seed": lay.layout_seed,
            "episode_seed": game.episode_seed,
            "layout": lay.to_ascii().splitlines(),
            "spawn": list(game.pos),
            "first_order": list(game.required),
        }
        self.steps: list[StepRecord] = []

    def record(self, action: int, result: StepResult) -> StepRecord:
        g = self.game
        rec = StepRecord(
            t=g.t - 1,
            action=int(action),
            pos=g.pos,
            carried=tuple(g.carried),
            remaining=g.remaining,
            deliverable=g.deliverable,
            delta=result.reward,
            delivered=result.delivered,
            completed=result.order_completed,
            bump=result.bump,
            picked=result.picked,
        )
        self.steps.append(rec)
        return rec

    def finish(self) -> Trajectory:
        g = self.game
        end = {
            "type": "end",
            "steps": len(self.steps),
            "score": g.score,
            "orders_completed": g.orders_completed,
            "items_delivered": g.items_delivered,
        }
        return Trajectory(header=dict(self.header), steps=list(self.steps), end=end)


def run_episode(
    layout: Layout,
    episode_seed: int,
    player,
    *,
    role: SeedRole | str | None = None,
    max_steps: int = EPISODE_LEN,
) -> Trajectory:
    """Play one episode with `player` (anything with name / reset(game) / act(game))."""
    game = Game(layout, episode_seed, role=role)
    rec = Recorder(game, getattr(player, "name", type(player).__name__))
    player.reset(game)
    while not game.done and len(rec.steps) < max_steps:
        a = player.act(game)
        rec.record(a, game.step(a))
    return rec.finish()


class ReplayMismatch(AssertionError):
    """Re-simulating a trajectory did not reproduce it."""


def replay(traj: Trajectory) -> Trajectory:
    """Re-simulate the recorded actions from the header and return the new trajectory."""
    from .players.base import ActionListPlayer

    layout = traj.layout()
    player = ActionListPlayer(traj.actions)
    player.name = traj.header["player"]
    return run_episode(layout, traj.header["episode_seed"], player, max_steps=len(traj.steps))


def verify_replay(traj: Trajectory) -> Trajectory:
    """Raise ReplayMismatch unless replaying `traj` reproduces every step and the final result."""
    again = replay(traj)
    if again.start_pos != traj.start_pos:
        raise ReplayMismatch(f"spawn differs: recorded {traj.start_pos}, replayed {again.start_pos}")
    if again.header["first_order"] != traj.header["first_order"]:
        raise ReplayMismatch("first order differs")
    for a, b in zip(traj.steps, again.steps):
        if a != b:
            raise ReplayMismatch(f"step {a.t} differs:\n  recorded {a}\n  replayed {b}")
    if len(again.steps) != len(traj.steps):
        raise ReplayMismatch(f"length differs: {len(traj.steps)} vs {len(again.steps)}")
    for k in ("score", "orders_completed", "items_delivered", "steps"):
        if again.end[k] != traj.end[k]:
            raise ReplayMismatch(f"{k} differs: recorded {traj.end[k]}, replayed {again.end[k]}")
    return again
