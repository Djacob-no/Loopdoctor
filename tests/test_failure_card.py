"""Delivery 3, acceptance F: the failure card."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from weightshift.diagnose.failure_card import build_card
from weightshift.policy import random_actor, save_actor

ROOT = Path(__file__).resolve().parents[1]
CARDS = sorted((ROOT / "artifacts" / "d3").glob("*/failure_card.md"))


@pytest.fixture(scope="module")
def card(tmp_path_factory):
    path = save_actor(tmp_path_factory.mktemp("card") / "a.npz", random_actor(2))
    return path, *build_card(path, n=16)


def test_f1_contents(card):
    _, text, data = card
    assert data["episodes"] == 16 and data["trace_seed"] == 1_000_000
    assert set(data) >= {"T0", "open"}
    assert "| T0 (the layout above) |" in text and "| open (normal rooms) |" in text
    trace = text.split("ONE RECORDED EPISODE ON T0", 1)[1].splitlines()
    steps = [ln for ln in trace if re.match(r"^\s*\d+ \| ", ln)]
    assert len(steps) == 201  # the start line plus 200 steps
    assert steps[-1].lstrip().startswith("200 |")


def test_f2_deterministic(card):
    path, text, data = card
    again, data2 = build_card(path, n=16)
    assert again == text and data2 == data


def _assert_nothing_hidden(text: str) -> None:
    lowered = text.lower()
    for banned in ("prob", "unit", "neuron", "heatmap", "activation", "capacity", "l1.", "l2.", "l3."):
        assert banned not in lowered, banned


def test_f3_card_excludes_probabilities_and_units(card):
    _assert_nothing_hidden(card[1])


@pytest.mark.gates
@pytest.mark.skipif(not CARDS, reason="no committed Delivery 3 artifacts yet")
@pytest.mark.parametrize("card_path", CARDS, ids=lambda p: p.parent.name)
def test_f2_committed_cards_regenerate_exactly(card_path):
    text, _ = build_card(card_path.parent / "actor.npz")
    assert text == card_path.read_text(encoding="utf-8")
    _assert_nothing_hidden(text)
