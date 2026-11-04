"""Players: anything with `name`, `reset(game)` and `act(game) -> action`."""

from .base import ActionListPlayer, Player
from .scripted import PLAYERS, GreedyStraightPlayer, OraclePlayer, RandomPlayer, make_player

__all__ = [
    "Player",
    "ActionListPlayer",
    "RandomPlayer",
    "GreedyStraightPlayer",
    "OraclePlayer",
    "PLAYERS",
    "make_player",
]
