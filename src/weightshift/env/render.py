"""ASCII rendering of a running game."""

from __future__ import annotations

from .constants import ACTION_NAMES, EPISODE_LEN, MAX_CARRY, N_SKUS, SKU_LETTERS

LEGEND = "legend: # wall   A-D rack (SKU)   X dock   R robot   . floor"


def render_grid(game) -> str:
    """The 10x10 grid with the robot drawn as R (also when it stands on the dock)."""
    rows = game.layout.to_ascii(show_spawns=False).splitlines()
    r, c = game.pos
    rows[r] = rows[r][:c] + "R" + rows[r][c + 1 :]
    return "\n".join(rows)


def carried_text(game) -> str:
    items = [SKU_LETTERS[s] for s in range(N_SKUS) for _ in range(game.carried[s])]
    slots = [f"[{x}]" for x in items] + ["[ ]"] * (MAX_CARRY - len(items))
    return "".join(slots)


def render_frame(game, last_action: int | None = None, last_reward: int | None = None) -> str:
    rem = game.remaining
    lines = [
        f"step {game.t:3d}/{EPISODE_LEN}   score {game.score}   orders done {game.orders_completed}"
        f"   items delivered {game.items_delivered}",
        f"order    {game.order_text():<9}  still to deliver: "
        + " ".join(f"{SKU_LETTERS[s]}:{rem[s]}" for s in range(N_SKUS)),
        f"carried  {carried_text(game)}",
    ]
    if last_action is not None:
        extra = f"  (+{last_reward})" if last_reward else ""
        lines.append(f"last     {ACTION_NAMES[last_action]}{extra}")
    lines.append(render_grid(game))
    lines.append(LEGEND)
    return "\n".join(lines)
