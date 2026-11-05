"""The failure card (FINAL_PLAN §6.5, DELIVERY_3_ACCEPTANCE §2): what the model sees as diagnosis.

Built from the **rounded** network (the one shown in the prompt) on
`diagnosis` seeds, with sampled actions:

- the aggregate over 256 episodes on T0,
- the same aggregate on `open` layouts (to show competence),
- one full 200-step episode on T0 (diagnosis seed 1,000,000), one line per step.

Deliberately excluded: action probabilities, heatmaps, probes, and anything
about units (the capacity ranking stays hidden in the primary condition).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ..env.constants import ACTION_NAMES, N_SKUS, SKU_LETTERS
from ..env.generate import generate_layout
from ..env.seeds import seeds_for
from ..evaluation import evaluate
from ..policy.actor import load_actor, save_actor
from ..policy.player import PolicyPlayer
from ..serialize.quantize import rounded
from ..trajectory import Trajectory, run_episode

N_CARD = 256
TRACE_SEED = 1_000_000
ROLE = "diagnosis"


def _aggregate(results) -> dict:
    ms = [r.metrics for r in results]
    n = len(ms)
    return {
        "episodes": n,
        "any_delivery": sum(m.items_delivered > 0 for m in ms) / n,
        "items_delivered": float(np.mean([m.items_delivered for m in ms])),
        "orders_completed": float(np.mean([m.orders_completed for m in ms])),
        "score": float(np.mean([m.score for m in ms])),
        "loop_index": float(np.mean([m.loop_index for m in ms])),
        "idle_or_bump": float(np.mean([m.idle_fraction for m in ms])),
        "full_not_delivering": float(np.mean([m.full_not_delivering for m in ms])),
    }


def _carried_text(carried) -> str:
    items = [SKU_LETTERS[s] for s in range(N_SKUS) for _ in range(carried[s])]
    return " ".join(items) if items else "-"


def trace_lines(traj: Trajectory) -> list[str]:
    lines = ["step | action | robot (row, col) after the step | carrying | score"]
    score = 0
    r0, c0 = traj.start_pos
    lines.append(f"   0 | start  | ({r0}, {c0}) | - | 0")
    for st in traj.steps:
        score += st.delta
        lines.append(
            f"{st.t + 1:>4} | {ACTION_NAMES[st.action]:<6} | ({st.pos[0]}, {st.pos[1]}) | {_carried_text(st.carried)} | {score}"
        )
    return lines


def build_card(actor_path: str | Path, n: int = N_CARD) -> tuple[str, dict]:
    """Returns (card text, data). Deterministic for a given actor file."""
    params = rounded(load_actor(actor_path))
    seeds = seeds_for(ROLE, n)
    with tempfile.TemporaryDirectory(prefix="ws-card-") as tmp:
        p = save_actor(Path(tmp) / "rounded.npz", params)
        spec = ("policy", str(p), "sampled")
        t0 = _aggregate(evaluate(spec, "T0", seeds, role=ROLE))
        op = _aggregate(evaluate(spec, "open", seeds, role=ROLE))
    traj = run_episode(generate_layout("T0"), TRACE_SEED, PolicyPlayer(params, "sampled"), role=ROLE)
    data = {"T0": t0, "open": op, "trace_seed": TRACE_SEED, "trace_score": traj.score, "episodes": n}
    return render_card(data, traj), data


def render_card(data: dict, traj: Trajectory) -> str:
    def row(name: str, a: dict) -> str:
        return (
            f"| {name} | {100 * a['any_delivery']:.1f}% | {a['items_delivered']:.2f} | {a['orders_completed']:.2f} | "
            f"{a['score']:.2f} | {a['loop_index']:.3f} | {a['idle_or_bump']:.3f} | {a['full_not_delivering']:.3f} |"
        )

    n = data["episodes"]
    out = [
        f"RESULTS OF THE CURRENT NETWORK ({n} episodes per row, actions sampled from the policy, 200 steps each).",
        "",
        "| Layouts | Episodes with any delivery | Items delivered | Orders completed | Score | Loop index | "
        "Idle or bump fraction | Full but not delivering |",
        "|---|---|---|---|---|---|---|---|",
        row("T0 (the layout above)", data["T0"]),
        row("open (normal rooms)", data["open"]),
        "",
        "Loop index: fraction of steps whose (cell, carried items) state already occurred in the previous 20 steps. "
        "Idle or bump: fraction of steps that were a no-op or a move into a wall or rack. "
        "Full but not delivering: fraction of steps carrying 3 items, at least one of them deliverable, "
        "with no delivery in the last 20 steps.",
        "",
        f"ONE RECORDED EPISODE ON T0 (all {len(traj.steps)} steps; final score {traj.score}).",
        "",
        *trace_lines(traj),
    ]
    return "\n".join(out) + "\n"
