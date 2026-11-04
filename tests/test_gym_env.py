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


@pytest.mark.parametrize("bad", [1.9, -1, 7, 2.5])
def test_step_rejects_invalid_actions_instead_of_truncating(bad):
    """Review finding: int(1.9) used to turn into action 1 (up)."""
    env = WeightShiftEnv(family="T0")
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(bad)
    assert env.game.t == 0  # nothing happened


def test_step_accepts_numpy_integer_actions():
    env = WeightShiftEnv(family="T0")
    env.reset(seed=1)
    env.step(np.int64(2))
    env.step(env.action_space.sample())
    assert env.game.t == 2


@pytest.mark.parametrize("role", ["train", "diagnosis", "eval"])
def test_reset_without_seed_draws_inside_the_role_range(role):
    """Review finding: an unseeded reset drew from [0, 2**31) and then failed the role check."""
    from weightshift.env import SEED_RANGES, SeedRole

    lo, hi = SEED_RANGES[SeedRole(role)]
    env = WeightShiftEnv(family="open", role=role)
    for _ in range(20):
        env.reset()
        assert lo <= env.game.episode_seed < hi
        assert lo <= env.game.layout.layout_seed < hi
