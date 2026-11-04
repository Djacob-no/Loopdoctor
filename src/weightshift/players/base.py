"""The player interface shared by scripted players, humans and (later) policies."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..env.constants import NOOP


@runtime_checkable
class Player(Protocol):
    name: str

    def reset(self, game) -> None:
        """Called once with the freshly created Game, before the first act()."""

    def act(self, game) -> int:
        """Return the next action (0..6) for the current game state."""


class ActionListPlayer:
    """Plays a fixed list of actions, then no-ops. Used by tests and replay."""

    name = "scripted"

    def __init__(self, actions: Sequence[int]):
        self.actions = [int(a) for a in actions]
        self._i = 0

    def reset(self, game) -> None:
        self._i = 0

    def act(self, game) -> int:
        a = self.actions[self._i] if self._i < len(self.actions) else NOOP
        self._i += 1
        return a
