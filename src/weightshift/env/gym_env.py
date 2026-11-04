"""Gymnasium wrapper around `Game` (acceptance H1).

Reward is the unshaped score delta. Training-time shaping (FINAL_PLAN §6.2)
belongs in a separate wrapper in the PPO delivery, never in this class.

Reset options (all optional):
    layout       a Layout to play (overrides the env's family)
    layout_seed  seed for the env's family (defaults to the episode seed)
    episode_seed seed for spawn and orders (defaults to `seed`)
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .constants import N_ACTIONS, OBS_SIZE
from .game import Game
from .generate import FAMILIES, FIXED_LAYOUTS, generate_layout
from .layout import Layout
from .observation import builder_for
from .render import render_frame
from .seeds import SEED_RANGES, SeedRole


class WeightShiftEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        family: str = "open",
        layout: Layout | None = None,
        role: SeedRole | str | None = None,
        render_mode: str | None = None,
    ):
        if layout is None and family not in FAMILIES + FIXED_LAYOUTS:
            raise ValueError(f"unknown family {family!r}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.family = family
        self.fixed_layout = layout
        self.role = role
        self.render_mode = render_mode
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.game: Game | None = None
        self._builder = None
        self._last_action: int | None = None
        self._last_reward: int | None = None

    # ------------------------------------------------------------------ api
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = dict(options or {})
        if seed is None:
            # Draw inside the role's range when there is one, so an unseeded reset never
            # trips the role check (and never strays into another role's seeds).
            lo, hi = SEED_RANGES[SeedRole(self.role)] if self.role is not None else (0, 2**31 - 1)
            seed = int(self.np_random.integers(lo, hi))
        episode_seed = int(options.get("episode_seed", seed))
        layout = options.get("layout") or self.fixed_layout
        if layout is None:
            layout = generate_layout(self.family, int(options.get("layout_seed", seed)), role=self.role)
        self.game = Game(layout, episode_seed, role=self.role)
        self._builder = builder_for(layout)
        self._last_action = self._last_reward = None
        return self._builder.build(self.game), self._info(None)

    def step(self, action):
        if self.game is None:
            raise RuntimeError("call reset() before step()")
        r = self.game.step(action)  # Game.step validates; do not coerce 1.9 into 1 here
        self._last_action, self._last_reward = int(action), r.reward
        obs = self._builder.build(self.game)
        return obs, float(r.reward), False, r.truncated, self._info(r)

    def render(self):
        if self.render_mode == "ansi" and self.game is not None:
            return render_frame(self.game, self._last_action, self._last_reward)
        return None

    # -------------------------------------------------------------- helpers
    def _info(self, r) -> dict[str, Any]:
        g = self.game
        info: dict[str, Any] = {
            "t": g.t,
            "score": g.score,
            "orders_completed": g.orders_completed,
            "items_delivered": g.items_delivered,
            "layout": g.layout.name,
            "episode_seed": g.episode_seed,
        }
        if r is not None:
            info.update(delivered=r.delivered, order_completed=r.order_completed, bump=r.bump, picked=r.picked)
        return info
