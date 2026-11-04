"""Acceptance B3: invariants checked on every step of 1,000 random-action episodes."""

from __future__ import annotations

import numpy as np

from weightshift.env import Game, generate_layout, player_rng
from weightshift.env.constants import EPISODE_LEN, MAX_CARRY, N_ACTIONS, OBS_SIZE
from weightshift.env.observation import encode_observation

N_EPISODES = 1000


def test_invariants_hold_for_1000_random_episodes():
    families = ("open", "trap", "probe", "T0")
    for ep in range(N_EPISODES):
        layout = generate_layout(families[ep % 4], ep)
        g = Game(layout, ep)
        rng = player_rng(ep)
        last_score = 0
        actions = rng.integers(0, N_ACTIONS, size=EPISODE_LEN)
        for t in range(EPISODE_LEN):
            r = g.step(int(actions[t]))
            assert not layout.is_obstacle(g.p), f"robot inside an obstacle at {g.pos}"
            assert 0 <= g.n_carried <= MAX_CARRY
            assert g.n_carried == sum(g.carried) and min(g.carried) >= 0
            assert g.score >= last_score
            assert g.score - last_score == r.reward
            last_score = g.score
            assert all(0 <= g.delivered[s] <= g.required[s] for s in range(4))
            obs = encode_observation(g)
            assert obs.shape == (OBS_SIZE,) and obs.dtype == np.float32
            assert np.isin(obs, (0.0, 1.0)).all()
            assert r.truncated == (t == EPISODE_LEN - 1)
        assert g.done
