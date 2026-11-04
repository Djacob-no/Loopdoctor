"""Renderer snapshot (supports E4 and I1)."""

from __future__ import annotations

from weightshift.env import Game, load_fixed_layout
from weightshift.env.constants import DOWN, PICK
from weightshift.env.render import LEGEND, render_frame, render_grid

T0_FRAME_STEP0 = """\
step   0/200   score 0   orders done 0   items delivered 0
order    A B B      still to deliver: A:1 B:2 C:0 D:0
carried  [ ][ ][ ]
##########
#...R....#
#........#
#.ABCDAB.#
#.A....C.#
#.D....B.#
#.C....A.#
#........#
#...X....#
##########
""" + LEGEND


def test_t0_frame_snapshot():
    g = Game(load_fixed_layout("T0"), 1_000_000)
    g.debug_set(pos=(1, 4), required=[1, 2, 0, 0], delivered=[0, 0, 0, 0])
    assert render_frame(g) == T0_FRAME_STEP0


def test_frame_shows_carried_items_and_last_action():
    g = Game(load_fixed_layout("T0"), 1_000_000)
    g.debug_set(pos=(1, 4))
    g.step(DOWN)
    r = g.step(PICK)  # rack C below (3,4)
    text = render_frame(g, PICK, r.reward)
    assert "carried  [C][ ][ ]" in text
    assert "last     pick" in text


def test_robot_is_drawn_on_the_dock():
    g = Game(load_fixed_layout("T0"), 1_000_000)
    g.debug_set(pos=(8, 4))
    assert render_grid(g).splitlines()[8] == "#...R....#"
