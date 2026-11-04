"""Game constants. Every number here is a locked rule from DELIVERY_1_ACCEPTANCE.md §2."""

from __future__ import annotations

# Grid
GRID = 10
N_CELLS = GRID * GRID

# Items
N_SKUS = 4
SKU_LETTERS = "ABCD"
MAX_CARRY = 3

# Orders
ORDER_MIN_ITEMS = 3
ORDER_MAX_ITEMS = 4
ORDER_COMPLETE_BONUS = 5

# Episode
EPISODE_LEN = 200

# Actions
NOOP, UP, DOWN, LEFT, RIGHT, PICK, DROP = range(7)
N_ACTIONS = 7
ACTION_NAMES = ("noop", "up", "down", "left", "right", "pick", "drop")
MOVE_ACTIONS = (UP, DOWN, LEFT, RIGHT)
MOVE_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
# When several racks are adjacent, pick takes from the first one in this order.
PICK_ORDER = (UP, DOWN, LEFT, RIGHT)

# Cell codes in Layout.cells: racks are 0..N_SKUS-1 (the SKU they hold).
WALL = -2
FLOOR = -1

# Observation
N_CHANNELS = 6
OBS_SIZE = N_CELLS * N_CHANNELS  # 600
CH_ROBOT, CH_OBSTACLE, CH_NEEDED, CH_DOCK, CH_PREV, CH_GAUGE = range(N_CHANNELS)
GAUGE_CARRY_ROW = 0
GAUGE_CARRY_COLS = MAX_CARRY  # row 0, cols 0..2
GAUGE_NEED_ROW0 = 1  # SKU s lives on row 1 + s
GAUGE_NEED_COLS = ORDER_MAX_ITEMS  # cols 0..3
