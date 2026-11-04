"""Acceptance H1: the Gymnasium wrapper."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from weightshift.env import SeedRangeError, generate_layout
from weightshift.env.gym_env import WeightShiftEnv


@pytest.mark.parametrize("family", ["open", "trap", "probe", "T0"])
def test_check_env_passes(family):
    check_env(WeightShiftEnv(family=family), skip_render_check=True)


def test_check_env_passes_with_ansi_render():
    check_env(WeightShiftEnv(family="T0", render_mode="ansi"))


def test_reset_with_seed_is_reproducible_and_episode_runs_200_steps():
    env = WeightShiftEnv(family="trap")
    o1, i1 = env.reset(seed=11)
    o2, i2 = env.reset(seed=11)
    assert np.array_equal(o1, o2) and i1 == i2
    total, steps, truncated = 0.0, 0, False
    while not truncated:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert not terminated
        total += reward
        steps += 1
    assert steps == 200 and total == info["score"]


def test_reset_options_choose_layout_and_seeds():
    env = WeightShiftEnv(family="open")
    lay = generate_layout("probe", 3)
    env.reset(seed=5, options={"layout": lay})
    assert env.game.layout is lay and env.game.episode_seed == 5
    env.reset(seed=5, options={"layout_seed": 9, "episode_seed": 21})
    assert env.game.layout is generate_layout("open", 9) and env.game.episode_seed == 21


def test_role_is_enforced_by_the_wrapper():
    env = WeightShiftEnv(family="open", role="eval")
    env.reset(seed=2_000_000)
    with pytest.raises(SeedRangeError):
        env.reset(seed=1)


def test_render_and_errors():
    env = WeightShiftEnv(family="T0", render_mode="ansi")
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset(seed=1)
    env.step(2)
    assert "last     down" in env.render()
    assert WeightShiftEnv(family="T0").render() is None
    with pytest.raises(ValueError):
        WeightShiftEnv(family="nope")
    with pytest.raises(ValueError):
        WeightShiftEnv(render_mode="human")
