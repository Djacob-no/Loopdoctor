"""A player driven by an actor network (sampled or greedy actions)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..env.constants import OBS_SIZE
from ..env.game import Game, player_rng
from ..env.observation import builder_for
from .actor import forward, greedy_action, sample_action, softmax

MODES = ("sampled", "greedy")


class PolicyPlayer:
    """Plays from the 600-number observation, like the trained policy will.

    Sampled mode draws one uniform per step from the episode's action-sampling
    RNG (`player_rng(episode_seed)`), so the same actor and seeds always give the
    same episode, and before/after comparisons are paired.
    """

    def __init__(self, params: Mapping[str, np.ndarray], mode: str = "sampled", name: str | None = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.params = params
        self.mode = mode
        self.name = name or f"policy:{mode}"
        self._obs = np.zeros(OBS_SIZE, dtype=np.float32)

    def reset(self, game: Game) -> None:
        self._rng = player_rng(game.episode_seed)
        self._builder = builder_for(game.layout)

    def act(self, game: Game) -> int:
        obs = self._builder.build(game, out=self._obs)
        logits = forward(self.params, obs)
        if self.mode == "greedy":
            return greedy_action(logits)
        return sample_action(softmax(logits), float(self._rng.random()))
