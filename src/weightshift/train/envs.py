"""Training environments: the family mix and the disclosed shaping (FINAL_PLAN §6.2).

Shaping, training only:

    r_train = r + beta * (d_prev - d)

where d is the **Manhattan** distance from the robot to the dock (never the
BFS distance, which would reward the detour and remove the trap), applied on
steps that start with the robot carrying at least one item. `r` is the
unshaped score delta, which is what every evaluation reports.

`TrainVecEnv` steps N games in one process. It is deliberately simpler than a
Gymnasium vector env, and the whole object pickles, which makes checkpoints
resumable bit for bit. The end of an episode (step 200, or a shorter
practice length) is treated as terminal; the observation carries no clock.

Two per-family options, both **training only** (DELIVERY_3_ACCEPTANCE §2, §8):

- `episode_steps`: shorter practice episodes for a family.
- `fence`: entering a rectangle of cells in a practice episode of that family
  ends the episode with -penalty.

Evaluation never uses shaping, either option, or anything else here: it
always plays the full 200 steps, unshaped and unfenced.
"""

from __future__ import annotations

from collections.abc import Mapping

import gymnasium as gym
import numpy as np

from ..env.constants import EPISODE_LEN, OBS_SIZE
from ..env.game import Game
from ..env.generate import FIXED_LAYOUTS, generate_layout
from ..env.layout import manhattan
from ..env.observation import builder_for
from ..env.seeds import SEED_RANGES, SeedRole


def shaping_bonus(beta: float, p_prev: int, p_now: int, dock: int, carrying: bool) -> float:
    """beta * (d_prev - d) with Manhattan distance to the dock, only while carrying."""
    if not carrying or beta == 0.0:
        return 0.0
    return beta * (manhattan(p_prev, dock) - manhattan(p_now, dock))


class ShapingWrapper(gym.Wrapper):
    """Adds the shaping bonus to a WeightShiftEnv. info['unshaped_reward'] keeps the score delta."""

    def __init__(self, env, beta: float):
        super().__init__(env)
        self.beta = float(beta)

    def step(self, action):
        game = self.env.unwrapped.game
        p_prev, carrying = game.p, game.n_carried > 0
        obs, reward, terminated, truncated, info = self.env.step(action)
        bonus = shaping_bonus(self.beta, p_prev, game.p, game.layout.dock, carrying)
        info = dict(info, unshaped_reward=reward, shaping=bonus)
        return obs, reward + bonus, terminated, truncated, info


class TrainVecEnv:
    """N games on train-range seeds, with families drawn from a mix."""

    def __init__(
        self,
        n_envs: int,
        mix: Mapping[str, float],
        beta: float,
        seed: int,
        episode_steps: Mapping[str, int] | None = None,
        fence: Mapping[str, Mapping] | None = None,
    ):
        self.n = int(n_envs)
        self.families = sorted(mix)
        w = np.array([mix[f] for f in self.families], dtype=np.float64)
        self.weights = w / w.sum()
        self.beta = float(beta)
        self.episode_steps = dict(episode_steps or {})
        self.fence_cells: dict[str, frozenset[int]] = {}
        self.fence_penalty: dict[str, float] = {}
        for fam, f in (fence or {}).items():
            (r0, r1), (c0, c1) = f["rows"], f["cols"]
            self.fence_cells[fam] = frozenset(r * 10 + c for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
            self.fence_penalty[fam] = float(f["penalty"])
        self.rng = np.random.default_rng([int(seed), 0x7EA1])
        self.train_lo, self.train_hi = SEED_RANGES[SeedRole.TRAIN]
        self.games: list[Game] = [None] * self.n  # type: ignore[list-item]
        self.builders = [None] * self.n
        self.ep_family = [""] * self.n
        self.obs = np.zeros((self.n, OBS_SIZE), dtype=np.float32)
        for i in range(self.n):
            self._new_episode(i)

    def _new_episode(self, i: int) -> None:
        fam = self.families[int(self.rng.choice(len(self.families), p=self.weights))]
        episode_seed = int(self.rng.integers(self.train_lo, self.train_hi))
        if fam in FIXED_LAYOUTS:
            layout = generate_layout(fam)
        else:
            layout = generate_layout(fam, int(self.rng.integers(self.train_lo, self.train_hi)), role="train")
        self.games[i] = Game(layout, episode_seed, role="train")
        self.builders[i] = builder_for(layout)
        self.ep_family[i] = fam
        self.builders[i].build(self.games[i], out=self.obs[i])

    def step(self, actions: np.ndarray):
        """Returns obs (N, 600), shaped reward (N,), done (N,), and a list of finished episodes."""
        rewards = np.zeros(self.n, dtype=np.float32)
        dones = np.zeros(self.n, dtype=bool)
        finished: list[dict] = []
        for i in range(self.n):
            g = self.games[i]
            p_prev, carrying = g.p, g.n_carried > 0
            r = g.step(int(actions[i]))
            rewards[i] = r.reward + shaping_bonus(self.beta, p_prev, g.p, g.layout.dock, carrying)
            fam = self.ep_family[i]
            fenced = fam in self.fence_cells and g.p in self.fence_cells[fam]
            if fenced:
                rewards[i] -= self.fence_penalty[fam]
            if fenced or r.truncated or g.t >= self.episode_steps.get(fam, EPISODE_LEN):
                dones[i] = True
                finished.append(
                    {"family": fam, "score": g.score, "delivered": g.items_delivered,
                     "orders": g.orders_completed, "fenced": fenced}
                )
                self._new_episode(i)
            else:
                self.builders[i].build(g, out=self.obs[i])
        return self.obs, rewards, dones, finished

    def __getstate__(self):
        d = dict(self.__dict__)
        d["builders"] = None  # cached per layout; rebuilt on load
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.builders = [builder_for(g.layout) for g in self.games]
