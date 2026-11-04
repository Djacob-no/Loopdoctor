"""The 600-number observation and its decoder (DELIVERY_1_ACCEPTANCE.md §2 "Observation").

Index of (row, col, channel) is interleaved, as in the MIT Exp 1 / Exp 5 grids:

    i = (row * 10 + col) * 6 + channel

Channels: 0 robot, 1 wall or rack, 2 rack whose SKU is still needed
(remaining[s] - carried[s] > 0), 3 dock, 4 previous robot cell, 5 gauge.

Gauge (channel 5 only): row 0 cols 0-2 hold the carried count as a bar; row
1+s cols 0-3 hold remaining[s] for SKU s as a bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .constants import (
    CH_DOCK,
    CH_GAUGE,
    CH_NEEDED,
    CH_OBSTACLE,
    CH_PREV,
    CH_ROBOT,
    GAUGE_CARRY_COLS,
    GAUGE_CARRY_ROW,
    GAUGE_NEED_COLS,
    GAUGE_NEED_ROW0,
    GRID,
    N_CELLS,
    N_CHANNELS,
    N_SKUS,
    OBS_SIZE,
)
from .layout import Layout


def obs_index(row: int, col: int, channel: int) -> int:
    if not (0 <= row < GRID and 0 <= col < GRID and 0 <= channel < N_CHANNELS):
        raise IndexError(f"({row}, {col}, {channel}) is outside the 10x10x6 observation")
    return (row * GRID + col) * N_CHANNELS + channel


def cell_channel_index(p: int, channel: int) -> int:
    """Index for flat cell p and a channel."""
    return p * N_CHANNELS + channel


_GAUGE_CARRY = np.array([obs_index(GAUGE_CARRY_ROW, c, CH_GAUGE) for c in range(GAUGE_CARRY_COLS)], dtype=np.intp)
_GAUGE_NEED = [
    np.array([obs_index(GAUGE_NEED_ROW0 + s, c, CH_GAUGE) for c in range(GAUGE_NEED_COLS)], dtype=np.intp)
    for s in range(N_SKUS)
]


class ObservationBuilder:
    """Per-layout cache of the static channels; builds one observation in a few array writes."""

    def __init__(self, layout: Layout):
        self.layout = layout
        base = np.zeros(OBS_SIZE, dtype=np.float32)
        for p in range(N_CELLS):
            if layout.is_obstacle(p):
                base[cell_channel_index(p, CH_OBSTACLE)] = 1.0
        base[cell_channel_index(layout.dock, CH_DOCK)] = 1.0
        base.setflags(write=False)
        self.base = base
        self.rack_idx = [
            np.array([cell_channel_index(p, CH_NEEDED) for p in layout.rack_cells if layout.cells[p] == s], dtype=np.intp)
            for s in range(N_SKUS)
        ]

    def build(self, game, out: np.ndarray | None = None) -> np.ndarray:
        obs = self.base.copy() if out is None else out
        if out is not None:
            obs[:] = self.base
        obs[game.p * N_CHANNELS + CH_ROBOT] = 1.0
        if game.prev >= 0:
            obs[game.prev * N_CHANNELS + CH_PREV] = 1.0
        req, dlv, car = game.required, game.delivered, game.carried
        for s in range(N_SKUS):
            rem = req[s] - dlv[s]
            if rem - car[s] > 0:
                obs[self.rack_idx[s]] = 1.0
            if rem > 0:
                obs[_GAUGE_NEED[s][: min(rem, GAUGE_NEED_COLS)]] = 1.0
        if game.n_carried:
            obs[_GAUGE_CARRY[: game.n_carried]] = 1.0
        return obs


@lru_cache(maxsize=512)
def builder_for(layout: Layout) -> ObservationBuilder:
    return ObservationBuilder(layout)


def encode_observation(game) -> np.ndarray:
    """Observation of a `Game` as a float32 vector of shape (600,)."""
    return builder_for(game.layout).build(game)


# ----------------------------------------------------------------- decoding
@dataclass(frozen=True)
class DecodedObservation:
    robot: tuple[int, int]
    prev: tuple[int, int] | None
    dock: tuple[int, int]
    obstacles: np.ndarray  # (10, 10) bool, walls and racks
    needed_racks: np.ndarray  # (10, 10) bool
    carried_total: int
    remaining: tuple[int, ...]

    def to_ascii(self) -> str:
        """A drawable grid: # obstacle, * still-needed rack, X dock, R robot, o previous cell."""
        rows = []
        for r in range(GRID):
            line = []
            for c in range(GRID):
                if (r, c) == self.robot:
                    ch = "R"
                elif self.needed_racks[r, c]:
                    ch = "*"
                elif self.obstacles[r, c]:
                    ch = "#"
                elif (r, c) == self.dock:
                    ch = "X"
                elif self.prev is not None and (r, c) == self.prev:
                    ch = "o"
                else:
                    ch = "."
                line.append(ch)
            rows.append("".join(line))
        return "\n".join(rows)


class DecodeError(ValueError):
    """The vector is not a valid WeightShift observation."""


def _bar_length(values: np.ndarray, what: str) -> int:
    n = int(values.sum())
    if not np.array_equal(values[:n], np.ones(n)) or values[n:].any():
        raise DecodeError(f"{what} gauge is not a left-aligned bar: {values.tolist()}")
    return n


def _single_cell(plane: np.ndarray, what: str, optional: bool = False) -> tuple[int, int] | None:
    cells = np.argwhere(plane > 0.5)
    if len(cells) == 0 and optional:
        return None
    if len(cells) != 1:
        raise DecodeError(f"expected exactly one {what} cell, found {len(cells)}")
    return int(cells[0][0]), int(cells[0][1])


def decode_observation(obs: np.ndarray) -> DecodedObservation:
    x = np.asarray(obs)
    if x.shape != (OBS_SIZE,):
        raise DecodeError(f"expected shape ({OBS_SIZE},), got {x.shape}")
    if not np.isin(x, (0.0, 1.0)).all():
        raise DecodeError("observation values must be 0 or 1")
    planes = x.reshape(GRID, GRID, N_CHANNELS)  # [row, col, channel] thanks to the interleaved index
    gauge = planes[:, :, CH_GAUGE]
    carried_total = _bar_length(gauge[GAUGE_CARRY_ROW, :GAUGE_CARRY_COLS], "carried")
    remaining = tuple(
        _bar_length(gauge[GAUGE_NEED_ROW0 + s, :GAUGE_NEED_COLS], f"SKU {s}") for s in range(N_SKUS)
    )
    return DecodedObservation(
        robot=_single_cell(planes[:, :, CH_ROBOT], "robot"),
        prev=_single_cell(planes[:, :, CH_PREV], "previous", optional=True),
        dock=_single_cell(planes[:, :, CH_DOCK], "dock"),
        obstacles=planes[:, :, CH_OBSTACLE] > 0.5,
        needed_racks=planes[:, :, CH_NEEDED] > 0.5,
        carried_total=carried_total,
        remaining=remaining,
    )
