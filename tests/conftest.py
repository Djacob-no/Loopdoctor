from __future__ import annotations

import pytest

from weightshift.env import Game, layout_from_spec

# A small hand-built level used by the rule tests:
#
#   ##########
#   #s.......#   s = spawn (1,1)
#   #..A..B..#   A at (2,3), B at (2,6)
#   #........#
#   #........#
#   #........#
#   #..C..D..#   C at (6,3), D at (6,6)
#   #........#
#   #...X....#   dock at (8,4)
#   ##########
RULES_RACKS = {(2, 3): "A", (2, 6): "B", (6, 3): "C", (6, 6): "D"}
RULES_DOCK = (8, 4)


@pytest.fixture
def rules_layout():
    return layout_from_spec(RULES_RACKS, RULES_DOCK, [(1, 1)], name="rules")


@pytest.fixture
def game(rules_layout):
    return Game(rules_layout, episode_seed=123)
