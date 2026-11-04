"""Acceptance D1-D3: the 600-number observation."""

from __future__ import annotations

import numpy as np
import pytest

from weightshift.env import Game, generate_layout, rc
from weightshift.env.constants import (
    CH_DOCK,
    CH_GAUGE,
    CH_NEEDED,
    CH_OBSTACLE,
    CH_PREV,
    CH_ROBOT,
    N_ACTIONS,
    OBS_SIZE,
)
from weightshift.env.observation import DecodeError, decode_observation, encode_observation, obs_index


def _plane(obs, ch):
    return obs.reshape(10, 10, 6)[:, :, ch]


# --------------------------------------------------------------------- D2
def test_interleaved_index_formula():
    assert obs_index(0, 0, 0) == 0
    assert obs_index(0, 0, 5) == 5
    assert obs_index(0, 1, 0) == 6
    assert obs_index(1, 0, 0) == 60
    assert obs_index(9, 9, 5) == 599
    for r in range(10):
        for c in range(10):
            for ch in range(6):
                assert obs_index(r, c, ch) == (r * 10 + c) * 6 + ch
    with pytest.raises(IndexError):
        obs_index(10, 0, 0)


def test_static_channels_cell_by_cell(game, rules_layout):
    obs = encode_observation(game)
    assert obs.shape == (OBS_SIZE,) and obs.dtype == np.float32
    for p in range(100):
        r, c = rc(p)
        assert obs[obs_index(r, c, CH_OBSTACLE)] == (1.0 if rules_layout.is_obstacle(p) else 0.0)
        assert obs[obs_index(r, c, CH_DOCK)] == (1.0 if p == rules_layout.dock else 0.0)


def test_robot_and_previous_cell(game):
    game.debug_set(pos=(1, 1))
    obs = encode_observation(game)
    assert _plane(obs, CH_ROBOT).sum() == 1 and obs[obs_index(1, 1, CH_ROBOT)] == 1
    assert _plane(obs, CH_PREV).sum() == 0  # step 0: no previous cell
    game.step(4)  # right
    obs = encode_observation(game)
    assert obs[obs_index(1, 2, CH_ROBOT)] == 1 and obs[obs_index(1, 1, CH_PREV)] == 1
    assert _plane(obs, CH_PREV).sum() == 1


def test_needed_channel_marks_only_racks_of_still_needed_skus(game):
    # rules layout racks: A (2,3), B (2,6), C (6,3), D (6,6)
    game.debug_set(required=[1, 0, 2, 0], delivered=[0, 0, 0, 0], carried=[0, 0, 0, 0])
    plane = _plane(encode_observation(game), CH_NEEDED)
    assert {tuple(x) for x in np.argwhere(plane)} == {(2, 3), (6, 3)}


@pytest.mark.parametrize("carried_total", [0, 1, 2, 3])
def test_gauge_carried_bar(game, carried_total):
    game.debug_set(carried=[carried_total, 0, 0, 0], required=[4, 0, 0, 0], delivered=[0, 0, 0, 0])
    g = _plane(encode_observation(game), CH_GAUGE)
    assert g[0, :3].tolist() == [1.0] * carried_total + [0.0] * (3 - carried_total)
    assert g[0, 3:].sum() == 0


@pytest.mark.parametrize("remaining", [0, 1, 2, 3, 4])
def test_gauge_remaining_bars(game, remaining):
    game.debug_set(required=[remaining, 4 - remaining, 0, 4], delivered=[0, 0, 0, 0], carried=[0, 0, 0, 0])
    g = _plane(encode_observation(game), CH_GAUGE)
    assert g[1, :4].tolist() == [1.0] * remaining + [0.0] * (4 - remaining)
    assert g[2, :4].sum() == 4 - remaining
    assert g[3, :4].sum() == 0
    assert g[4, :4].tolist() == [1.0] * 4
    # Nothing else in the gauge plane is ever set.
    mask = np.zeros((10, 10), bool)
    mask[0, :3] = True
    mask[1:5, :4] = True
    assert g[~mask].sum() == 0


def test_gauge_does_not_touch_other_channels(game, rules_layout):
    game.debug_set(required=[4, 0, 0, 0], delivered=[0, 0, 0, 0], carried=[3, 0, 0, 0])
    obs = encode_observation(game)
    assert _plane(obs, CH_OBSTACLE).sum() == sum(1 for p in range(100) if rules_layout.is_obstacle(p))


# --------------------------------------------------------------------- D3
def test_needed_channel_turns_off_when_sku_is_covered_by_delivered_plus_carried(game):
    a_rack = obs_index(2, 3, CH_NEEDED)
    game.debug_set(required=[2, 0, 1, 0], delivered=[0, 0, 0, 0], carried=[0, 0, 0, 0])
    assert encode_observation(game)[a_rack] == 1
    game.debug_set(carried=[1, 0, 0, 0])  # one A carried, one still to pick
    assert encode_observation(game)[a_rack] == 1
    game.debug_set(carried=[2, 0, 0, 0])  # both A carried
    assert encode_observation(game)[a_rack] == 0
    game.debug_set(carried=[1, 0, 0, 0], delivered=[1, 0, 0, 0])  # one delivered, one carried
    assert encode_observation(game)[a_rack] == 0
    assert encode_observation(game)[obs_index(6, 3, CH_NEEDED)] == 1  # C is still needed


# --------------------------------------------------------------------- D1
def test_decode_encode_round_trip_on_1000_random_states():
    rng = np.random.default_rng(0)
    for i in range(1000):
        family = ("open", "trap", "probe", "T0")[i % 4]
        layout = generate_layout(family, i)
        g = Game(layout, i)
        for _ in range(int(rng.integers(0, 120))):
            g.step(int(rng.integers(N_ACTIONS)))
        dec = decode_observation(encode_observation(g))
        assert dec.robot == g.pos
        assert dec.prev == g.prev_pos
        assert dec.dock == rc(layout.dock)
        assert dec.carried_total == g.n_carried
        assert dec.remaining == g.remaining
        expected_obstacles = np.array([layout.is_obstacle(p) for p in range(100)]).reshape(10, 10)
        assert np.array_equal(dec.obstacles, expected_obstacles)
        need = g.need
        expected_needed = np.array(
            [layout.cells[p] >= 0 and need[layout.cells[p]] > 0 for p in range(100)]
        ).reshape(10, 10)
        assert np.array_equal(dec.needed_racks, expected_needed)


def test_decoded_grid_is_drawable(game):
    game.debug_set(pos=(1, 1), required=[1, 0, 0, 0], delivered=[0, 0, 0, 0])
    text = decode_observation(encode_observation(game)).to_ascii()
    lines = text.splitlines()
    assert len(lines) == 10 and all(len(ln) == 10 for ln in lines)
    assert lines[1][1] == "R" and lines[2][3] == "*" and lines[8][4] == "X" and lines[2][6] == "#"


def test_decoder_rejects_invalid_vectors(game):
    obs = encode_observation(game)
    with pytest.raises(DecodeError):
        decode_observation(obs[:-1])
    bad = obs.copy()
    bad[0] = 0.5
    with pytest.raises(DecodeError):
        decode_observation(bad)
    bad = obs.copy()
    bad[obs_index(5, 5, CH_ROBOT)] = 1  # two robots
    with pytest.raises(DecodeError):
        decode_observation(bad)
    bad = obs.copy()
    bad[obs_index(0, 2, CH_GAUGE)] = 1  # gap in the carried bar
    bad[obs_index(0, 0, CH_GAUGE)] = 0
    with pytest.raises(DecodeError):
        decode_observation(bad)


def test_builder_can_write_into_a_preallocated_buffer(game):
    from weightshift.env.observation import builder_for

    out = np.full(OBS_SIZE, 7.0, dtype=np.float32)
    builder_for(game.layout).build(game, out=out)
    assert np.array_equal(out, encode_observation(game))
