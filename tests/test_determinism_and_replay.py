"""Acceptance C1, C2 (determinism) and I2 (record + replay)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from weightshift.env import Game, generate_layout, load_fixed_layout
from weightshift.env.constants import EPISODE_LEN, N_ACTIONS
from weightshift.players.base import ActionListPlayer
from weightshift.trajectory import ReplayMismatch, Trajectory, code_version, replay, run_episode, verify_replay

FAMILIES = ("open", "trap", "probe", "T0")


def _random_actions(seed: int) -> list[int]:
    return [int(a) for a in np.random.default_rng([seed, 99]).integers(0, N_ACTIONS, size=EPISODE_LEN)]


# ---------------------------------------------------------------------- C1
def test_c1_same_seeds_and_actions_give_byte_identical_trajectories():
    for seed in range(1000):
        layout = generate_layout(FAMILIES[seed % 4], seed)
        actions = _random_actions(seed)
        a = run_episode(layout, seed, ActionListPlayer(actions)).to_jsonl()
        b = run_episode(generate_layout(FAMILIES[seed % 4], seed), seed, ActionListPlayer(actions)).to_jsonl()
        assert a == b, f"seed {seed} not deterministic"


# ---------------------------------------------------------------------- C2
@pytest.mark.parametrize("family", FAMILIES)
def test_c2_different_episode_seeds_give_different_orders_or_spawns(family):
    layout = generate_layout(family, 0)
    same = 0
    for s in range(1000):
        g1, g2 = Game(layout, 2 * s), Game(layout, 2 * s + 1)
        if g1.pos == g2.pos and g1.required == g2.required:
            same += 1
    assert same <= 10, f"{same}/1000 seed pairs produced the same spawn and first order"


# ---------------------------------------------------------------------- I2
def test_i2_recorded_episode_replays_identically(tmp_path):
    for i, family in enumerate(FAMILIES):
        layout = generate_layout(family, 40 + i)
        traj = run_episode(layout, 1_000_000 + i, ActionListPlayer(_random_actions(i)))
        path = traj.write(tmp_path / f"{family}.jsonl")
        back = Trajectory.read(path)
        assert back.to_jsonl() == traj.to_jsonl()
        again = verify_replay(back)
        assert again.score == traj.score
        assert again.steps == traj.steps


def test_trajectory_file_layout(tmp_path):
    traj = run_episode(load_fixed_layout("T0"), 1_000_005, ActionListPlayer([2, 2, 5]))
    lines = [json.loads(x) for x in traj.to_jsonl().splitlines()]
    head, first, end = lines[0], lines[1], lines[-1]
    assert head["type"] == "header" and head["format"] == 1
    for key in ("layout_seed", "episode_seed", "layout", "code_version", "spawn", "first_order", "player"):
        assert key in head
    assert head["layout"] == load_fixed_layout("T0").to_ascii().splitlines()
    assert head["code_version"] == code_version()
    assert first["type"] == "step" and set(first) >= {"action", "pos", "carried", "delta"}
    assert end["type"] == "end" and end["steps"] == 200 and len(lines) == 202


def test_tampered_trajectory_fails_replay():
    traj = run_episode(load_fixed_layout("T0"), 1_000_001, ActionListPlayer(_random_actions(3)))
    text = traj.to_jsonl().splitlines()
    step = json.loads(text[5])
    step["action"] = (step["action"] + 1) % N_ACTIONS
    text[5] = json.dumps(step, sort_keys=True, separators=(",", ":"))
    bad = Trajectory.from_jsonl("\n".join(text))
    with pytest.raises(ReplayMismatch):
        verify_replay(bad)

    end_bad = Trajectory.from_jsonl(traj.to_jsonl())
    end_bad.end["score"] += 1
    with pytest.raises(ReplayMismatch, match="score"):
        verify_replay(end_bad)


def test_replay_of_a_short_trajectory():
    traj = run_episode(load_fixed_layout("T0"), 1_000_002, ActionListPlayer([1, 2, 3]), max_steps=3)
    assert len(traj.steps) == 3 and traj.end["steps"] == 3
    assert replay(traj).steps == traj.steps


@pytest.mark.parametrize(
    "text",
    [
        "",
        '{"type":"step"}\n',
        '{"type":"header","format":99}\n{"type":"end"}\n',
        '{"type":"header","format":1}\n',
    ],
)
def test_malformed_trajectory_files_are_rejected(text):
    with pytest.raises(ValueError):
        Trajectory.from_jsonl(text)
