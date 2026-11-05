"""Seeded layout families (DELIVERY_1_ACCEPTANCE.md §2 "Layout families").

- open:  2-3 short rack segments in the middle of the room, all 4 SKUs, dock on
         the inner ring. The policy should be competent here.
- trap:  the dock sits below a U of racks whose closed side faces the spawn
         area. The straight line from spawn to dock hits the closed side; the
         only route first moves away from the dock (the Kinetix "back up first"
         analogue).
- probe: the same U with a two-cell door in the closed side, aligned with the
         dock, so a direct walk works. Used to show the policy *can* navigate
         this geometry when the pocket does not block it.
- detour: one straight rack wall between the spawn area and the dock, with
         no U around the dock. The straight line is blocked and the route
         first moves away from the dock, as in `trap`, but there is no pocket.
         Used to show the policy can route around walls (DELIVERY_3 §2).
- T0:    the fixed, hand-authored headline trap (fixtures/T0.txt).

trap, probe and detour layouts are built in a canonical orientation and then rotated
or mirrored by one of the 8 symmetries of the square, chosen by the seed.
Every generator retries from the same RNG stream until the layout passes the
playability checks, so a seed always maps to the same layout.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from .constants import FLOOR, GRID, N_SKUS, WALL
from .layout import Layout, LayoutError, idx, manhattan
from .seeds import SeedRole, validate_seed

FAMILIES = ("open", "trap", "probe", "detour")
FIXED_LAYOUTS = ("T0",)

_MAX_ATTEMPTS = 1000


# --------------------------------------------------------------------- helpers
def _empty_grid() -> list[list[int]]:
    return [[WALL if r in (0, GRID - 1) or c in (0, GRID - 1) else FLOOR for c in range(GRID)] for r in range(GRID)]


def _transform(k: int, r: int, c: int) -> tuple[int, int]:
    """One of the 8 symmetries of the 10x10 square (k in 0..7)."""
    n = GRID - 1
    if k >= 4:  # mirror left-right first
        c = n - c
    for _ in range(k % 4):  # then rotate 90 degrees clockwise k times
        r, c = c, n - r
    return r, c


def _finalize(
    grid: list[list[int]],
    dock: tuple[int, int],
    spawns: list[tuple[int, int]],
    *,
    family: str,
    seed: int,
    transform: int = 0,
) -> Layout:
    cells = [WALL] * (GRID * GRID)
    for r in range(GRID):
        for c in range(GRID):
            tr, tc = _transform(transform, r, c)
            cells[idx(tr, tc)] = grid[r][c]
    tdock = idx(*_transform(transform, *dock))
    tspawns = tuple(sorted(idx(*_transform(transform, *s)) for s in spawns))
    return Layout(tuple(cells), tdock, tspawns, name=f"{family}-{seed}", family=family, layout_seed=seed)


def _assign_skus(rng: np.random.Generator, n: int) -> list[int]:
    """n >= 4 SKU labels containing every SKU at least once, in random order."""
    labels = list(range(N_SKUS)) + [int(x) for x in rng.integers(0, N_SKUS, size=n - N_SKUS)]
    return [labels[i] for i in rng.permutation(n)]


# ------------------------------------------------------------------------ open
def _try_open(rng: np.random.Generator, seed: int) -> Layout | None:
    grid = _empty_grid()
    placed: list[tuple[int, int]] = []
    n_segments = int(rng.integers(2, 4))  # 2..3
    for _ in range(n_segments):
        length = int(rng.integers(2, 4))  # 2..3
        horizontal = bool(rng.integers(2))
        r0, c0 = int(rng.integers(2, 8)), int(rng.integers(2, 8))
        seg = [(r0, c0 + i) if horizontal else (r0 + i, c0) for i in range(length)]
        # Racks stay inside rows/cols 2..7 so the inner ring is a free corridor,
        # and segments keep one free cell between them (also diagonally).
        if any(not (2 <= r <= 7 and 2 <= c <= 7) for r, c in seg):
            return None
        if any(max(abs(r - pr), abs(c - pc)) < 2 for r, c in seg for pr, pc in placed):
            return None
        placed.extend(seg)
    skus = _assign_skus(rng, len(placed))
    for (r, c), s in zip(placed, skus):
        grid[r][c] = s
    ring = [(r, c) for r in range(1, GRID - 1) for c in range(1, GRID - 1) if r in (1, GRID - 2) or c in (1, GRID - 2)]
    dock = ring[int(rng.integers(len(ring)))]
    dock_p = idx(*dock)
    spawns = [
        (r, c)
        for r in range(1, GRID - 1)
        for c in range(1, GRID - 1)
        if grid[r][c] == FLOOR and manhattan(idx(r, c), dock_p) >= 4
    ]
    layout = _finalize(grid, dock, spawns, family="open", seed=seed)
    return None if layout.playability_problems() else layout


# ---------------------------------------------------------------- trap / probe
# Canonical orientation (before the symmetry transform):
#
#   row 1-2   spawn candidates, cols 3..6
#   row 3     closed side of the U: racks at cols 2..7
#   rows 4..  legs at cols 2 and 7, length 3 or 4
#   row 8     dock, at a column strictly inside the U (3..6)
#
# The pocket (inside the U) is open towards the dock row only, so from the
# spawn side the dock is reached by going around a leg via column 1 or 8.
BAR_ROW = 3
BAR_COLS = range(2, 8)
LEG_COLS = (2, 7)
DOCK_ROW = 8
DOCK_COLS = range(3, 7)
SPAWN_ROWS = (1, 2)
SPAWN_COLS = range(3, 7)


def _try_u_shape(rng: np.random.Generator, seed: int, *, door: bool) -> Layout | None:
    family = "probe" if door else "trap"
    grid = _empty_grid()
    leg_len = int(rng.integers(3, 5))  # 3..4
    dock_col = int(DOCK_COLS[int(rng.integers(len(DOCK_COLS)))])
    transform = int(rng.integers(8))

    bar = [(BAR_ROW, c) for c in BAR_COLS]
    if door:
        door_cols = (dock_col, dock_col + 1) if dock_col + 1 <= DOCK_COLS[-1] else (dock_col - 1, dock_col)
        bar = [(r, c) for r, c in bar if c not in door_cols]
    legs = [(BAR_ROW + i, c) for c in LEG_COLS for i in range(1, leg_len + 1)]

    # Every SKU appears on the closed side itself, so a rack of each SKU can be
    # picked from the spawn side without entering the pocket.
    bar_skus = _assign_skus(rng, len(bar))
    leg_skus = [int(x) for x in rng.integers(0, N_SKUS, size=len(legs))]
    for (r, c), s in zip(bar, bar_skus):
        grid[r][c] = s
    for (r, c), s in zip(legs, leg_skus):
        grid[r][c] = s

    spawns = [(r, c) for r in SPAWN_ROWS for c in SPAWN_COLS]
    layout = _finalize(grid, (DOCK_ROW, dock_col), spawns, family=family, seed=seed, transform=transform)
    return None if layout.playability_problems() else layout


# ---------------------------------------------------------------------- detour
# Canonical orientation (before the symmetry transform):
#
#   rows 1..2   spawn candidates, at columns inside the wall's span
#   row 3..5    one straight wall of 4..6 racks, columns a..b with 2 <= a, b <= 7
#   row 8       dock, at a column inside the wall's span
#
# Every straight (distance-reducing) route from a spawn to the dock crosses the
# wall's row between the spawn's and the dock's columns, which the wall covers,
# so it is blocked; the way round passes an end of the wall (column a-1 or b+1),
# which first moves away from the dock. Nothing encloses the dock.
DETOUR_WALL_ROWS = (3, 4, 5)
DETOUR_WALL_LEN = (4, 5, 6)


def _try_detour(rng: np.random.Generator, seed: int) -> Layout | None:
    grid = _empty_grid()
    row = int(DETOUR_WALL_ROWS[int(rng.integers(len(DETOUR_WALL_ROWS)))])
    length = int(DETOUR_WALL_LEN[int(rng.integers(len(DETOUR_WALL_LEN)))])
    a = int(rng.integers(2, 8 - length + 1))  # wall columns a .. a+length-1, inside 2..7
    b = a + length - 1
    dock_col = int(rng.integers(a, b + 1))
    transform = int(rng.integers(8))
    wall = [(row, c) for c in range(a, b + 1)]
    for (r, c), s in zip(wall, _assign_skus(rng, len(wall))):
        grid[r][c] = s
    spawns = [(r, c) for r in SPAWN_ROWS for c in range(a, b + 1)]
    layout = _finalize(grid, (DOCK_ROW, dock_col), spawns, family="detour", seed=seed, transform=transform)
    return None if layout.playability_problems() else layout


# --------------------------------------------------------------------- fixture
def fixtures_dir() -> Path:
    env = os.environ.get("WEIGHTSHIFT_FIXTURES")
    if env:
        return Path(env)
    # src/weightshift/env/generate.py -> repository root
    return Path(__file__).resolve().parents[3] / "fixtures"


@lru_cache(maxsize=None)
def load_fixed_layout(name: str) -> Layout:
    if name not in FIXED_LAYOUTS:
        raise ValueError(f"unknown fixed layout {name!r}; known: {FIXED_LAYOUTS}")
    path = fixtures_dir() / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (set WEIGHTSHIFT_FIXTURES to the fixtures directory)")
    layout = Layout.from_ascii(path.read_text(encoding="utf-8"), name=name, family=name)
    problems = layout.playability_problems()
    if problems:
        raise LayoutError(f"fixture {name} is not playable: {problems}")
    return layout


# ------------------------------------------------------------------- frontend
@lru_cache(maxsize=4096)
def _generate_cached(family: str, seed: int) -> Layout:
    rng = np.random.default_rng([seed, 0x1A7])
    for _ in range(_MAX_ATTEMPTS):
        if family == "open":
            layout = _try_open(rng, seed)
        elif family == "trap":
            layout = _try_u_shape(rng, seed, door=False)
        elif family == "detour":
            layout = _try_detour(rng, seed)
        else:
            layout = _try_u_shape(rng, seed, door=True)
        if layout is not None:
            return layout
    raise LayoutError(f"could not generate a playable {family} layout for seed {seed}")


def generate_layout(family: str, seed: int = 0, role: SeedRole | str | None = None) -> Layout:
    """The layout for (family, seed). `family` is open/trap/probe or a fixed name such as T0."""
    if family in FIXED_LAYOUTS:
        return load_fixed_layout(family)
    if family not in FAMILIES:
        raise ValueError(f"unknown layout family {family!r}; expected one of {FAMILIES + FIXED_LAYOUTS}")
    if role is not None:
        validate_seed(role, seed)
    return _generate_cached(family, int(seed))
