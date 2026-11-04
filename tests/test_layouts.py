"""Acceptance E1-E4: generated layouts, run over 10,000 layouts per family."""

from __future__ import annotations

from pathlib import Path

import pytest

from weightshift.env import Layout, LayoutError, SeedRangeError, idx, layout_from_spec, manhattan, rc
from weightshift.env.constants import FLOOR, N_SKUS, WALL
from weightshift.env.generate import FAMILIES, generate_layout, load_fixed_layout
from weightshift.env.layout import monotone_path_exists, shortest_path

N_LAYOUTS = 10_000

T0_ASCII = """\
##########
#..ssss..#
#........#
#.ABCDAB.#
#.A....C.#
#.D....B.#
#.C....A.#
#........#
#...X....#
##########"""


def _layouts(family: str, n: int = N_LAYOUTS):
    return (generate_layout(family, s) for s in range(n))


# ------------------------------------------------------------------------- E1
def _check_e1(layout: Layout) -> None:
    assert layout.structural_problems() == [], layout.name
    assert layout.playability_problems() == [], layout.name
    # Exactly one dock, checked on the rendered grid (the form that is saved and parsed back).
    assert layout.to_ascii().count("X") == 1, layout.name
    assert layout.cells[layout.dock] == FLOOR
    reach = layout.reachable_from(layout.dock)
    for s in layout.spawns:
        assert s in reach
    for sku in range(N_SKUS):
        assert any(p in reach for p in layout.pick_cells[sku])
    for p in layout.rack_cells:
        assert any(n in reach for n in layout.neighbours(p))


@pytest.mark.parametrize("family", FAMILIES)
def test_e1_every_family_is_playable(family):
    for layout in _layouts(family):
        _check_e1(layout)


def test_e1_t0_is_playable():
    _check_e1(load_fixed_layout("T0"))


# ------------------------------------------------------------------------- E2
def _check_trap(layout: Layout) -> None:
    for s in layout.spawns:
        assert not monotone_path_exists(layout, s, layout.dock), f"{layout.name}: straight line open from {rc(s)}"
        path = shortest_path(layout, s, layout.dock)
        assert path is not None, f"{layout.name}: dock unreachable from {rc(s)}"
        dists = [manhattan(p, layout.dock) for p in path]
        # The route has to move away from the dock at least once.
        assert any(b > a for a, b in zip(dists, dists[1:])), f"{layout.name}: route from {rc(s)} never backs off"


def test_e2_trap_layouts_block_the_straight_line():
    for layout in _layouts("trap"):
        _check_trap(layout)


def test_e2_t0_blocks_the_straight_line():
    _check_trap(load_fixed_layout("T0"))


# ------------------------------------------------------------------------- E3
def test_e3_probe_layouts_allow_the_straight_line():
    for layout in _layouts("probe"):
        for s in layout.spawns:
            assert monotone_path_exists(layout, s, layout.dock), f"{layout.name}: blocked from {rc(s)}"


# ------------------------------------------------------------------------- E4
def test_e4_t0_loads_from_fixture_file_and_matches_snapshot():
    path = Path(__file__).resolve().parents[1] / "fixtures" / "T0.txt"
    assert path.exists()
    t0 = load_fixed_layout("T0")
    assert t0.to_ascii() == T0_ASCII
    assert generate_layout("T0") is t0
    assert t0.dock == idx(8, 4)
    assert [rc(s) for s in t0.spawns] == [(1, 3), (1, 4), (1, 5), (1, 6)]


# ---------------------------------------------------------- generator basics
@pytest.mark.parametrize("family", FAMILIES)
def test_generation_is_deterministic_and_varied(family):
    a = [generate_layout(family, s).cells for s in range(50)]
    b = [generate_layout(family, s).cells for s in range(50)]
    assert a == b
    assert len(set(a)) > 40  # seeds give genuinely different layouts


@pytest.mark.parametrize("family", ["trap", "probe"])
def test_u_shapes_use_all_eight_orientations(family):
    docks = {rc(generate_layout(family, s).dock) for s in range(400)}
    sides = {
        "top" if r == 1 else "bottom" if r == 8 else "left" if c == 1 else "right"
        for r, c in docks
    }
    assert sides == {"top", "bottom", "left", "right"}


def test_open_racks_have_a_free_corridor_ring():
    for layout in _layouts("open", 2000):
        for p in layout.rack_cells:
            r, c = rc(p)
            assert 2 <= r <= 7 and 2 <= c <= 7


def test_generate_layout_validates_role_and_family():
    generate_layout("open", 2_000_000, role="eval")
    with pytest.raises(SeedRangeError):
        generate_layout("open", 3, role="eval")
    with pytest.raises(ValueError):
        generate_layout("maze", 0)


# --------------------------------------------------------- ASCII and checks
def test_ascii_round_trip_for_generated_layouts():
    for family in FAMILIES:
        for s in range(200):
            layout = generate_layout(family, s)
            back = Layout.from_ascii(layout.to_ascii())
            assert back.cells == layout.cells and back.dock == layout.dock
            assert back.spawns == layout.spawns


@pytest.mark.parametrize(
    "text,msg",
    [
        ("#" * 10, "10 lines"),
        ("\n".join(["##########"] + ["#........#"] * 8 + ["##########"]), "exactly one dock"),
        ("\n".join(["##########", "#X..Z....#"] + ["#........#"] * 7 + ["##########"]), "unknown character"),
    ],
)
def test_from_ascii_rejects_malformed_text(text, msg):
    with pytest.raises(LayoutError, match=msg):
        Layout.from_ascii(text)


def test_structural_rules_are_enforced():
    racks = {(2, 3): "A", (2, 6): "B", (6, 3): "C", (6, 6): "D"}
    with pytest.raises(LayoutError, match="not directly adjacent"):
        layout_from_spec(racks, (4, 4), [(1, 1)])  # dock in the middle of the room
    with pytest.raises(LayoutError, match="missing"):
        layout_from_spec({(2, 3): "A", (2, 6): "B", (6, 3): "C"}, (8, 4), [(1, 1)])
    with pytest.raises(LayoutError, match="spawn"):
        layout_from_spec(racks, (8, 4), [(2, 3)])  # spawn on a rack
    with pytest.raises(LayoutError, match="no spawn"):
        layout_from_spec(racks, (8, 4), [])
    cells = list(layout_from_spec(racks, (8, 4), [(1, 1)]).cells)
    cells[idx(0, 4)] = FLOOR  # hole in the outer wall
    with pytest.raises(LayoutError, match="outer ring"):
        Layout(tuple(cells), idx(8, 4), (idx(1, 1),))
    cells[idx(0, 4)] = WALL
    cells[idx(4, 4)] = WALL  # interior wall
    with pytest.raises(LayoutError, match="interior"):
        Layout(tuple(cells), idx(8, 4), (idx(1, 1),))


def test_playability_detects_a_walled_off_sku():
    # SKU D is boxed in by other racks: no floor cell can pick it.
    racks = {(2, 3): "A", (2, 6): "B", (5, 5): "C", (6, 6): "D", (6, 5): "C", (7, 6): "C", (6, 7): "C", (5, 6): "C"}
    lay = layout_from_spec(racks, (8, 4), [(1, 1)])
    assert any("SKU D" in p for p in lay.playability_problems())


def test_path_helpers_on_t0():
    t0 = load_fixed_layout("T0")
    s = idx(1, 4)
    assert not monotone_path_exists(t0, s, t0.dock)
    path = shortest_path(t0, s, t0.dock)
    assert path[0] == s and path[-1] == t0.dock
    assert len(path) - 1 == t0.distances_from(t0.dock)[s]
    assert t0.dist_matrix[s][t0.dock] == len(path) - 1
    assert monotone_path_exists(t0, idx(7, 1), t0.dock)
    assert not monotone_path_exists(t0, idx(3, 3), t0.dock)  # obstacle source
