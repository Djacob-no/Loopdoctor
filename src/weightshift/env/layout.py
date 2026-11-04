"""The static part of a level: walls, racks, dock and spawn candidates.

Cells are addressed by a flat index `p = row * 10 + col` (row 0 = top). The
`Layout` precomputes the lookup tables the game loop needs so a step is a few
tuple lookups.

ASCII format (10 lines of 10 characters):

    #  wall            .  floor
    A-D  rack (SKU)    X  dock (floor)
    s  spawn candidate (floor)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import cached_property

from .constants import (
    FLOOR,
    GRID,
    MOVE_ACTIONS,
    MOVE_DELTAS,
    N_CELLS,
    N_SKUS,
    PICK_ORDER,
    SKU_LETTERS,
    WALL,
)


def idx(r: int, c: int) -> int:
    return r * GRID + c


def rc(p: int) -> tuple[int, int]:
    return divmod(p, GRID)


def manhattan(a: int, b: int) -> int:
    ar, ac = divmod(a, GRID)
    br, bc = divmod(b, GRID)
    return abs(ar - br) + abs(ac - bc)


class LayoutError(ValueError):
    """A layout breaks a structural rule."""


@dataclass(frozen=True)
class Layout:
    cells: tuple[int, ...]  # N_CELLS codes: WALL, FLOOR or SKU 0..3
    dock: int
    spawns: tuple[int, ...]
    name: str = "custom"
    family: str = "custom"
    layout_seed: int | None = None

    def __post_init__(self) -> None:
        problems = self.structural_problems()
        if problems:
            raise LayoutError(f"invalid layout {self.name!r}: " + "; ".join(problems))

    # ------------------------------------------------------------------ checks
    def structural_problems(self) -> list[str]:
        """Rules every layout must satisfy regardless of family."""
        out: list[str] = []
        if len(self.cells) != N_CELLS:
            return [f"expected {N_CELLS} cells, got {len(self.cells)}"]
        for p, v in enumerate(self.cells):
            r, c = rc(p)
            ring = r in (0, GRID - 1) or c in (0, GRID - 1)
            if ring and v != WALL:
                out.append(f"outer ring cell {(r, c)} is not wall")
            if not ring and v == WALL:
                out.append(f"interior cell {(r, c)} is wall (only racks may block the interior)")
            if v not in (WALL, FLOOR) and not 0 <= v < N_SKUS:
                out.append(f"cell {(r, c)} has unknown code {v}")
        if not 0 <= self.dock < N_CELLS or self.cells[self.dock] != FLOOR:
            out.append("dock must be a floor cell")
        else:
            dr, dc = rc(self.dock)
            if not (dr in (1, GRID - 2) or dc in (1, GRID - 2)):
                out.append(f"dock {(dr, dc)} is not directly adjacent to the outer wall")
        if not self.spawns:
            out.append("no spawn candidates")
        for s in self.spawns:
            if not 0 <= s < N_CELLS or self.cells[s] != FLOOR:
                out.append(f"spawn {rc(s) if 0 <= s < N_CELLS else s} is not a floor cell")
            elif s == self.dock:
                out.append("a spawn candidate sits on the dock")
        if len(set(self.spawns)) != len(self.spawns):
            out.append("duplicate spawn candidates")
        present = {v for v in self.cells if v >= 0}
        missing = [SKU_LETTERS[s] for s in range(N_SKUS) if s not in present]
        if missing:
            out.append(f"SKUs missing from racks: {''.join(missing)}")
        return out

    def playability_problems(self) -> list[str]:
        """Reachability rules (acceptance E1): dock, every SKU and a free neighbour per rack."""
        out: list[str] = []
        reach = self.reachable_from(self.dock)
        for s in self.spawns:
            if s not in reach:
                out.append(f"spawn {rc(s)} cannot reach the dock")
        for sku in range(N_SKUS):
            if not any(p in reach for p in self.pick_cells[sku]):
                out.append(f"no reachable cell picks SKU {SKU_LETTERS[sku]}")
        for p in self.rack_cells:
            if not any(n in reach for n in self.neighbours(p)):
                out.append(f"rack {rc(p)} has no reachable free neighbour")
        return out

    # ------------------------------------------------------------- basic views
    def is_obstacle(self, p: int) -> bool:
        return self.cells[p] != FLOOR

    def neighbours(self, p: int) -> list[int]:
        """In-bounds 4-neighbours that are floor."""
        r, c = rc(p)
        out = []
        for a in MOVE_ACTIONS:
            dr, dc = MOVE_DELTAS[a]
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID and 0 <= nc < GRID and self.cells[idx(nr, nc)] == FLOOR:
                out.append(idx(nr, nc))
        return out

    @cached_property
    def rack_cells(self) -> tuple[int, ...]:
        return tuple(p for p, v in enumerate(self.cells) if v >= 0)

    @cached_property
    def floor_cells(self) -> tuple[int, ...]:
        return tuple(p for p, v in enumerate(self.cells) if v == FLOOR)

    @cached_property
    def move_table(self) -> tuple[tuple[int, ...], ...]:
        """move_table[p][a] = cell after action a (0..4) from p; blocked moves stay at p."""
        table = []
        for p in range(N_CELLS):
            r, c = rc(p)
            row = [p]  # NOOP
            for a in MOVE_ACTIONS:
                dr, dc = MOVE_DELTAS[a]
                nr, nc = r + dr, c + dc
                q = idx(nr, nc)
                ok = 0 <= nr < GRID and 0 <= nc < GRID and self.cells[q] == FLOOR
                row.append(q if ok else p)
            table.append(tuple(row))
        return tuple(table)

    @cached_property
    def pick_sku_at(self) -> tuple[int, ...]:
        """SKU a pick yields when standing on floor cell p (first adjacent rack in PICK_ORDER), else -1."""
        out = []
        for p in range(N_CELLS):
            sku = -1
            if self.cells[p] == FLOOR:
                r, c = rc(p)
                for a in PICK_ORDER:
                    dr, dc = MOVE_DELTAS[a]
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < GRID and 0 <= nc < GRID and self.cells[idx(nr, nc)] >= 0:
                        sku = self.cells[idx(nr, nc)]
                        break
            out.append(sku)
        return tuple(out)

    @cached_property
    def pick_cells(self) -> tuple[tuple[int, ...], ...]:
        """pick_cells[s] = floor cells where a pick yields SKU s."""
        return tuple(tuple(p for p, v in enumerate(self.pick_sku_at) if v == s) for s in range(N_SKUS))

    # ----------------------------------------------------------------- pathing
    def distances_from(self, src: int) -> list[int]:
        """BFS distance over floor cells from `src` to every cell (-1 = unreachable)."""
        dist = [-1] * N_CELLS
        if self.cells[src] != FLOOR:
            return dist
        dist[src] = 0
        q = deque([src])
        table = self.move_table
        while q:
            p = q.popleft()
            d = dist[p] + 1
            for a in MOVE_ACTIONS:
                n = table[p][a]
                if dist[n] < 0:
                    dist[n] = d
                    q.append(n)
        return dist

    def reachable_from(self, src: int) -> set[int]:
        d = self.distances_from(src)
        return {p for p, v in enumerate(d) if v >= 0}

    @cached_property
    def dist_matrix(self) -> tuple[tuple[int, ...], ...]:
        """All-pairs BFS distances (computed lazily, ~ms per layout)."""
        return tuple(tuple(self.distances_from(p)) for p in range(N_CELLS))

    # ------------------------------------------------------------------- ASCII
    def to_ascii(self, show_spawns: bool = True) -> str:
        spawns = set(self.spawns)
        rows = []
        for r in range(GRID):
            line = []
            for c in range(GRID):
                p = idx(r, c)
                v = self.cells[p]
                if v == WALL:
                    ch = "#"
                elif v >= 0:
                    ch = SKU_LETTERS[v]
                elif p == self.dock:
                    ch = "X"
                elif show_spawns and p in spawns:
                    ch = "s"
                else:
                    ch = "."
                line.append(ch)
            rows.append("".join(line))
        return "\n".join(rows)

    @classmethod
    def from_ascii(
        cls,
        text: str,
        *,
        name: str = "custom",
        family: str = "custom",
        layout_seed: int | None = None,
    ) -> Layout:
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        if len(lines) != GRID or any(len(ln) != GRID for ln in lines):
            raise LayoutError(f"ASCII layout must be {GRID} lines of {GRID} characters")
        cells: list[int] = []
        dock: list[int] = []
        spawns: list[int] = []
        for r, ln in enumerate(lines):
            for c, ch in enumerate(ln):
                p = idx(r, c)
                if ch == "#":
                    cells.append(WALL)
                elif ch in SKU_LETTERS:
                    cells.append(SKU_LETTERS.index(ch))
                elif ch in ".Xs":
                    cells.append(FLOOR)
                    if ch == "X":
                        dock.append(p)
                    elif ch == "s":
                        spawns.append(p)
                else:
                    raise LayoutError(f"unknown character {ch!r} at {(r, c)}")
        if len(dock) != 1:
            raise LayoutError(f"expected exactly one dock 'X', found {len(dock)}")
        return cls(tuple(cells), dock[0], tuple(spawns), name=name, family=family, layout_seed=layout_seed)


def layout_from_spec(
    racks: dict[tuple[int, int], int | str],
    dock: tuple[int, int],
    spawns: list[tuple[int, int]],
    *,
    name: str = "custom",
    family: str = "custom",
    layout_seed: int | None = None,
) -> Layout:
    """Build a layout from a sparse description (outer wall is added automatically)."""
    cells = []
    for p in range(N_CELLS):
        r, c = rc(p)
        cells.append(WALL if r in (0, GRID - 1) or c in (0, GRID - 1) else FLOOR)
    for (r, c), sku in racks.items():
        cells[idx(r, c)] = SKU_LETTERS.index(sku) if isinstance(sku, str) else int(sku)
    return Layout(
        tuple(cells),
        idx(*dock),
        tuple(idx(*s) for s in spawns),
        name=name,
        family=family,
        layout_seed=layout_seed,
    )


def monotone_path_exists(layout: Layout, src: int, dst: int) -> bool:
    """Is there a path where every step strictly reduces Manhattan distance to `dst`?

    This is the "straight-line" route a greedy distance-descending agent could take.
    """
    if layout.is_obstacle(src) or layout.is_obstacle(dst):
        return False
    seen = {src}
    stack = [src]
    while stack:
        p = stack.pop()
        if p == dst:
            return True
        d = manhattan(p, dst)
        for n in layout.neighbours(p):
            if n not in seen and manhattan(n, dst) < d:
                seen.add(n)
                stack.append(n)
    return False


def shortest_path(layout: Layout, src: int, dst: int) -> list[int] | None:
    """One BFS shortest path from src to dst (inclusive), or None."""
    dist = layout.distances_from(dst)
    if dist[src] < 0:
        return None
    path = [src]
    p = src
    while p != dst:
        for n in layout.neighbours(p):
            if dist[n] == dist[p] - 1:
                p = n
                break
        path.append(p)
    return path
