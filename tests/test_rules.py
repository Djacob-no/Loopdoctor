"""Acceptance B1: one hand-built test per locked rule (DELIVERY_1_ACCEPTANCE.md §2)."""

from __future__ import annotations

import pytest

from weightshift.env import Game, layout_from_spec
from weightshift.env.constants import (
    DOWN,
    DROP,
    EPISODE_LEN,
    LEFT,
    NOOP,
    PICK,
    RIGHT,
    UP,
)

A, B, C, D = range(4)


# ---------------------------------------------------------------- movement
def test_move_into_outer_wall_is_a_bump(game):
    game.debug_set(pos=(1, 1))
    r = game.step(UP)
    assert r.bump and game.pos == (1, 1)
    r = game.step(LEFT)
    assert r.bump and game.pos == (1, 1)


def test_move_into_rack_is_a_bump(game):
    game.debug_set(pos=(1, 3))  # rack A is directly below at (2,3)
    r = game.step(DOWN)
    assert r.bump and game.pos == (1, 3)


def test_free_move_changes_position_and_is_not_a_bump(game):
    game.debug_set(pos=(1, 1))
    r = game.step(RIGHT)
    assert not r.bump and game.pos == (1, 2)
    r = game.step(DOWN)
    assert not r.bump and game.pos == (2, 2)


def test_noop_is_not_a_bump(game):
    r = game.step(NOOP)
    assert not r.bump and r.reward == 0


def test_previous_cell_is_the_position_one_step_ago(game):
    game.debug_set(pos=(1, 1))
    assert game.prev_pos is None  # nothing before step 0
    game.step(RIGHT)
    assert game.prev_pos == (1, 1)
    game.step(NOOP)
    assert game.prev_pos == (1, 2)  # stationary: previous cell equals current cell


def test_invalid_action_raises(game):
    with pytest.raises(ValueError):
        game.step(7)
    with pytest.raises(ValueError):
        game.step(-1)
    with pytest.raises(ValueError):
        game.step(1.5)


# -------------------------------------------------------------------- pick
def test_pick_from_4_adjacent_rack(game):
    game.debug_set(pos=(1, 3))  # A below
    r = game.step(PICK)
    assert r.picked == A
    assert game.carried == [1, 0, 0, 0] and game.n_carried == 1


@pytest.mark.parametrize("pos", [(1, 2), (1, 4), (3, 2), (3, 4)])
def test_pick_diagonal_to_rack_does_nothing(game, pos):
    game.debug_set(pos=pos)  # each is diagonal to rack A at (2,3) and 4-adjacent to no rack
    r = game.step(PICK)
    assert r.picked == -1 and game.n_carried == 0


def test_pick_with_several_adjacent_racks_uses_up_down_left_right_order():
    # (3,3) has A above and B to the right: "up" wins.
    lay = layout_from_spec({(2, 3): "A", (3, 4): "B", (6, 3): "C", (6, 6): "D"}, (8, 4), [(1, 1)])
    g = Game(lay, 1)
    g.debug_set(pos=(3, 3))
    assert g.step(PICK).picked == A
    # (4,4) has B above and nothing else: B.
    g.debug_set(pos=(4, 4))
    assert g.step(PICK).picked == B
    # (3,5) has B to the left only.
    g.debug_set(pos=(3, 5))
    assert g.step(PICK).picked == B


def test_carry_cap_is_three(game):
    game.debug_set(pos=(1, 3))
    for _ in range(3):
        assert game.step(PICK).picked == A
    r = game.step(PICK)
    assert r.picked == -1
    assert game.carried == [3, 0, 0, 0] and game.n_carried == 3


def test_wrong_pick_still_succeeds_and_occupies_a_slot(game):
    game.debug_set(pos=(1, 6), required=[1, 0, 0, 0], delivered=[0, 0, 0, 0])  # B rack below, order wants only A
    r = game.step(PICK)
    assert r.picked == B
    assert game.carried == [0, 1, 0, 0]
    assert game.need == (1, 0, 0, 0)  # the wrong item does not reduce what is still needed
    assert game.deliverable == 0


# -------------------------------------------------------------------- drop
def test_drop_away_from_dock_does_nothing(game):
    game.debug_set(pos=(7, 4), carried=[1, 0, 0, 0], required=[1, 1, 1, 0], delivered=[0, 0, 0, 0])
    r = game.step(DROP)
    assert r.delivered == 0 and r.reward == 0
    assert game.carried == [1, 0, 0, 0]


def test_drop_on_dock_delivers_one_point_per_needed_item(game):
    game.debug_set(pos=(8, 4), carried=[1, 0, 1, 0], required=[1, 1, 1, 0], delivered=[0, 0, 0, 0])
    r = game.step(DROP)
    assert r.delivered == 2 and r.reward == 2 and not r.order_completed
    assert game.carried == [0, 0, 0, 0] and game.n_carried == 0
    assert game.delivered == [1, 0, 1, 0]
    assert game.score == 2 and game.items_delivered == 2


def test_partial_delivery_leaves_unneeded_items_in_inventory(game):
    # Order wants one A; robot carries A and B. Only A is delivered, B stays (cannot be discarded).
    game.debug_set(pos=(8, 4), carried=[1, 1, 0, 0], required=[2, 0, 0, 0], delivered=[0, 0, 0, 0])
    r = game.step(DROP)
    assert r.delivered == 1
    assert game.carried == [0, 1, 0, 0] and game.n_carried == 1
    assert game.remaining == (1, 0, 0, 0)


def test_delivery_is_capped_at_remaining_per_sku(game):
    game.debug_set(pos=(8, 4), carried=[3, 0, 0, 0], required=[2, 1, 0, 0], delivered=[1, 0, 0, 0])
    r = game.step(DROP)
    assert r.delivered == 1  # only one more A was wanted
    assert game.carried == [2, 0, 0, 0]


def test_completing_an_order_gives_bonus_and_next_order_in_same_step(game):
    game.debug_set(pos=(8, 4), carried=[1, 0, 0, 0], required=[1, 1, 1, 0], delivered=[0, 1, 1, 0])
    r = game.step(DROP)
    assert r.delivered == 1 and r.order_completed
    assert r.reward == 1 + 5
    assert game.orders_completed == 1 and game.score == 6
    # The next order is already active after this same step.
    assert 3 <= sum(game.required) <= 4
    assert game.delivered == [0, 0, 0, 0]


def test_one_drop_serves_only_the_current_order(game):
    # Two A carried, the order needs one A. The second A is not applied to the new order in the same drop.
    game.debug_set(pos=(8, 4), carried=[2, 0, 0, 0], required=[1, 0, 0, 0], delivered=[0, 0, 0, 0])
    r = game.step(DROP)
    assert r.order_completed and r.delivered == 1
    assert game.carried == [1, 0, 0, 0]
    assert game.delivered == [0, 0, 0, 0]


def test_drop_with_nothing_deliverable_does_nothing(game):
    game.debug_set(pos=(8, 4), carried=[0, 0, 0, 0])
    r = game.step(DROP)
    assert r.delivered == 0 and r.reward == 0 and not r.order_completed


# ----------------------------------------------------------------- orders
def test_orders_have_three_or_four_items_from_four_skus(rules_layout):
    sizes = set()
    for seed in range(300):
        g = Game(rules_layout, seed)
        assert len(g.required) == 4 and min(g.required) >= 0
        sizes.add(sum(g.required))
    assert sizes == {3, 4}


def test_order_stream_does_not_depend_on_actions(rules_layout):
    """Two games with the same seed see the same next order even if they act differently first."""
    g1, g2 = Game(rules_layout, 7), Game(rules_layout, 7)
    assert g1.required == g2.required
    g2.step(RIGHT)
    g2.step(DOWN)
    for g in (g1, g2):  # force-complete the current order the same way in both
        g.debug_set(pos=(8, 4), carried=[1, 0, 0, 0], required=[1, 0, 0, 0], delivered=[0, 0, 0, 0])
        assert g.step(DROP).order_completed
    assert g1.required == g2.required


def test_spawn_is_one_of_the_candidates():
    lay = layout_from_spec(
        {(2, 3): "A", (2, 6): "B", (6, 3): "C", (6, 6): "D"}, (8, 4), [(1, 1), (1, 8), (4, 4)]
    )
    seen = {Game(lay, s).pos for s in range(200)}
    assert seen == {(1, 1), (1, 8), (4, 4)}


# ---------------------------------------------------------------- episode
def test_episode_truncates_at_exactly_200_steps(game):
    for t in range(EPISODE_LEN - 1):
        assert not game.step(NOOP).truncated
    assert game.t == EPISODE_LEN - 1 and not game.done
    r = game.step(NOOP)
    assert r.truncated and game.t == EPISODE_LEN and game.done
    with pytest.raises(RuntimeError):
        game.step(NOOP)


def test_role_validation_on_game(rules_layout):
    Game(rules_layout, 2_000_000, role="eval")
    from weightshift.env import SeedRangeError

    with pytest.raises(SeedRangeError):
        Game(rules_layout, 5, role="eval")
