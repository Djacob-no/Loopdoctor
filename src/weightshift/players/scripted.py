"""Scripted players used to calibrate the game (acceptance F1-F4).

They read the full game state (not the observation), which is fine: they are
measuring sticks for the environment, not agents under study.

- RandomPlayer: uniform over the 7 actions, from the episode's player RNG.
- GreedyStraightPlayer: competent at fetching items (plans its way to racks
  with BFS, like the oracle) but drives *straight* at the dock: every step
  towards the dock must reduce Manhattan distance, and it no-ops when all
  such steps are blocked. This is the local minimum the trap is built for
  (FINAL_PLAN §6.2), so on a probe layout only the pocket can stop it.
- OraclePlayer: plans with BFS distances. Picks only what the order still
  needs, delivers when full or when nothing more is needed.

Target choice for both planners (shared by goal()):
- go to the dock when holding something deliverable and either the
  inventory is full or nothing more needs picking;
- otherwise go to the nearest cell where a pick yields a needed SKU;
- otherwise (e.g. full of unneeded items) wait.
"""

from __future__ import annotations

from ..env.constants import DOWN, DROP, LEFT, MAX_CARRY, N_ACTIONS, N_SKUS, NOOP, PICK, RIGHT, UP
from ..env.game import Game, player_rng

GOAL_DOCK = "dock"
GOAL_PICK = "pick"
GOAL_WAIT = "wait"


def goal(game: Game) -> tuple[str, tuple[int, ...]]:
    """(mode, needed SKUs)."""
    need = game.need
    needed = tuple(s for s in range(N_SKUS) if need[s] > 0)
    deliverable = game.deliverable
    if deliverable > 0 and (game.n_carried >= MAX_CARRY or not needed):
        return GOAL_DOCK, needed
    if needed and game.n_carried < MAX_CARRY:
        return GOAL_PICK, needed
    return GOAL_WAIT, needed


_BFS_ORDER = (UP, DOWN, LEFT, RIGHT)


def bfs_step(layout, p: int, targets: list[int]) -> int:
    """First move of a shortest path from p to the nearest target (ties: lowest cell index), or NOOP."""
    dist = layout.dist_matrix
    reachable = [q for q in targets if dist[p][q] > 0]
    if not reachable:
        return NOOP
    target = min(reachable, key=lambda q: (dist[p][q], q))
    d = dist[p][target]
    for a in _BFS_ORDER:
        n = layout.move_table[p][a]
        if n != p and dist[n][target] == d - 1:
            return a
    return NOOP


class RandomPlayer:
    name = "random"

    def reset(self, game: Game) -> None:
        self._rng = player_rng(game.episode_seed)

    def act(self, game: Game) -> int:
        return int(self._rng.integers(N_ACTIONS))


class GreedyStraightPlayer:
    name = "greedy_straight"
    # Towards the dock, vertical moves are tried before horizontal ones.
    _ORDER = (DOWN, UP, RIGHT, LEFT)

    def reset(self, game: Game) -> None:
        pass

    def act(self, game: Game) -> int:
        lay = game.layout
        p = game.p
        mode, needed = goal(game)
        if mode == GOAL_PICK:
            if lay.pick_sku_at[p] in needed:
                return PICK
            return bfs_step(lay, p, [q for s in needed for q in lay.pick_cells[s]])
        if mode != GOAL_DOCK:
            return NOOP
        if p == lay.dock:
            return DROP
        pr, pc = divmod(p, 10)
        tr, tc = divmod(lay.dock, 10)
        for a in self._ORDER:
            reduces = (
                (a == DOWN and tr > pr) or (a == UP and tr < pr) or (a == RIGHT and tc > pc) or (a == LEFT and tc < pc)
            )
            if reduces and lay.move_table[p][a] != p:
                return a
        return NOOP


class OraclePlayer:
    name = "oracle"

    def reset(self, game: Game) -> None:
        pass

    def act(self, game: Game) -> int:
        lay = game.layout
        p = game.p
        mode, needed = goal(game)
        if mode == GOAL_DOCK:
            return DROP if p == lay.dock else bfs_step(lay, p, [lay.dock])
        if mode == GOAL_PICK:
            if lay.pick_sku_at[p] in needed:
                return PICK
            return bfs_step(lay, p, [q for s in needed for q in lay.pick_cells[s]])
        return NOOP


PLAYERS = {cls.name: cls for cls in (RandomPlayer, GreedyStraightPlayer, OraclePlayer)}


def make_player(name: str):
    try:
        return PLAYERS[name]()
    except KeyError:
        raise ValueError(f"unknown player {name!r}; expected one of {sorted(PLAYERS)}") from None
