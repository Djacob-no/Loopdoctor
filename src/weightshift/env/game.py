"""The game rules (DELIVERY_1_ACCEPTANCE.md §2). Pure Python, no Gymnasium.

One `Game` is one episode on one `Layout`. Randomness (spawn cell and the order
stream) comes only from the episode seed, and the order stream does not depend
on the actions taken, so two runs with the same seed see the same orders.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import (
    DROP,
    EPISODE_LEN,
    MAX_CARRY,
    N_ACTIONS,
    N_SKUS,
    ORDER_COMPLETE_BONUS,
    ORDER_MAX_ITEMS,
    ORDER_MIN_ITEMS,
    PICK,
    RIGHT,
    SKU_LETTERS,
    UP,
)
from .layout import Layout, idx, rc
from .seeds import SeedRole, validate_seed

# Sub-stream ids for np.random.default_rng([episode_seed, stream]).
RNG_STREAM_ENV = 0  # spawn + orders (owned by the game)
RNG_STREAM_PLAYER = 1  # action sampling (owned by players / policies)


def player_rng(episode_seed: int) -> np.random.Generator:
    """The action-sampling RNG that belongs to an episode seed."""
    return np.random.default_rng([episode_seed, RNG_STREAM_PLAYER])


@dataclass(frozen=True, slots=True)
class StepResult:
    reward: int  # unshaped score delta this step
    truncated: bool
    delivered: int  # items delivered this step
    order_completed: bool
    bump: bool  # tried to move into a wall or rack
    picked: int  # SKU picked this step, or -1


class Game:
    def __init__(self, layout: Layout, episode_seed: int, role: SeedRole | str | None = None):
        if role is not None:
            validate_seed(role, episode_seed)
        self.layout = layout
        self.episode_seed = int(episode_seed)
        self._rng = np.random.default_rng([self.episode_seed, RNG_STREAM_ENV])

        self.p: int = layout.spawns[int(self._rng.integers(len(layout.spawns)))]
        self.prev: int = -1  # cell at the previous time step; -1 before the first step
        self.t: int = 0
        self.carried: list[int] = [0] * N_SKUS
        self.n_carried: int = 0
        self.required: list[int] = [0] * N_SKUS
        self.delivered: list[int] = [0] * N_SKUS
        self.score: int = 0
        self.orders_completed: int = 0
        self.items_delivered: int = 0
        self._new_order()

    # ------------------------------------------------------------------ orders
    def _new_order(self) -> None:
        n_items = int(self._rng.integers(ORDER_MIN_ITEMS, ORDER_MAX_ITEMS + 1))
        items = self._rng.integers(0, N_SKUS, size=n_items)
        self.required = [int((items == s).sum()) for s in range(N_SKUS)]
        self.delivered = [0] * N_SKUS

    # ------------------------------------------------------------------- views
    @property
    def pos(self) -> tuple[int, int]:
        return rc(self.p)

    @property
    def prev_pos(self) -> tuple[int, int] | None:
        return None if self.prev < 0 else rc(self.prev)

    @property
    def done(self) -> bool:
        return self.t >= EPISODE_LEN

    @property
    def remaining(self) -> tuple[int, ...]:
        """remaining[s] = required[s] - delivered[s] for the current order."""
        return tuple(self.required[s] - self.delivered[s] for s in range(N_SKUS))

    @property
    def need(self) -> tuple[int, ...]:
        """Items still to be picked: max(0, remaining[s] - carried[s])."""
        return tuple(max(0, self.required[s] - self.delivered[s] - self.carried[s]) for s in range(N_SKUS))

    @property
    def deliverable(self) -> int:
        """How many carried items the current order would accept right now."""
        return sum(min(self.carried[s], self.required[s] - self.delivered[s]) for s in range(N_SKUS))

    def order_text(self) -> str:
        return " ".join(SKU_LETTERS[s] for s in range(N_SKUS) for _ in range(self.required[s]))

    # -------------------------------------------------------------------- step
    def step(self, action: int) -> StepResult:
        if self.t >= EPISODE_LEN:
            raise RuntimeError("episode is over; create a new Game")
        a = int(action)
        if not 0 <= a < N_ACTIONS or a != action:
            raise ValueError(f"invalid action {action!r}")

        old = self.p
        bump = False
        picked = -1
        delivered = 0
        completed = False

        if UP <= a <= RIGHT:
            new = self.layout.move_table[old][a]
            if new == old:
                bump = True
            else:
                self.p = new
        elif a == PICK:
            sku = self.layout.pick_sku_at[old]
            if sku >= 0 and self.n_carried < MAX_CARRY:
                self.carried[sku] += 1
                self.n_carried += 1
                picked = sku
        elif a == DROP:
            if old == self.layout.dock:
                for s in range(N_SKUS):
                    d = min(self.carried[s], self.required[s] - self.delivered[s])
                    if d:
                        self.carried[s] -= d
                        self.delivered[s] += d
                        delivered += d
                if delivered:
                    self.n_carried -= delivered
                    self.items_delivered += delivered
                    if self.delivered == self.required:
                        completed = True
                        self.orders_completed += 1
                        self._new_order()

        reward = delivered + (ORDER_COMPLETE_BONUS if completed else 0)
        self.score += reward
        self.prev = old
        self.t += 1
        return StepResult(
            reward=reward,
            truncated=self.t >= EPISODE_LEN,
            delivered=delivered,
            order_completed=completed,
            bump=bump,
            picked=picked,
        )

    # ------------------------------------------------------------ test hooks
    def debug_set(
        self,
        *,
        pos: tuple[int, int] | None = None,
        prev: tuple[int, int] | None = None,
        carried: list[int] | tuple[int, ...] | None = None,
        required: list[int] | tuple[int, ...] | None = None,
        delivered: list[int] | tuple[int, ...] | None = None,
        t: int | None = None,
    ) -> None:
        """Put the game in a hand-built state. For tests and diagnostics only."""
        if pos is not None:
            p = idx(*pos)
            if self.layout.is_obstacle(p):
                raise ValueError(f"{pos} is not a floor cell")
            self.p = p
        if prev is not None:
            self.prev = idx(*prev)
        if carried is not None:
            if len(carried) != N_SKUS or sum(carried) > MAX_CARRY or min(carried) < 0:
                raise ValueError("carried must be 4 non-negative counts summing to at most 3")
            self.carried = list(carried)
            self.n_carried = sum(carried)
        if required is not None:
            self.required = list(required)
        if delivered is not None:
            self.delivered = list(delivered)
        if t is not None:
            self.t = t
