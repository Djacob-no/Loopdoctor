"""Acceptance C3: seed ranges are disjoint and enforced."""

from __future__ import annotations

import itertools

import pytest

from weightshift.env import SEED_RANGES, SeedRangeError, SeedRole, role_of, seeds_for, validate_seed


def test_declared_ranges_match_the_acceptance_doc():
    assert SEED_RANGES[SeedRole.TRAIN] == (0, 1_000_000)
    assert SEED_RANGES[SeedRole.DIAGNOSIS] == (1_000_000, 1_001_000)
    assert SEED_RANGES[SeedRole.EVAL] == (2_000_000, 2_010_000)


def test_ranges_are_pairwise_disjoint():
    for (ra, (a0, a1)), (rb, (b0, b1)) in itertools.combinations(SEED_RANGES.items(), 2):
        assert a1 <= b0 or b1 <= a0, f"{ra} overlaps {rb}"


@pytest.mark.parametrize(
    "role,ok,bad",
    [
        ("train", [0, 999_999], [-1, 1_000_000, 2_000_000]),
        ("diagnosis", [1_000_000, 1_000_999], [999_999, 1_001_000, 2_000_000]),
        ("eval", [2_000_000, 2_009_999], [0, 1_000_000, 1_999_999, 2_010_000]),
    ],
)
def test_validate_seed_accepts_inside_and_rejects_outside(role, ok, bad):
    for s in ok:
        assert validate_seed(role, s) == s
    for s in bad:
        with pytest.raises(SeedRangeError):
            validate_seed(role, s)


def test_validate_seed_rejects_non_int_and_unknown_role():
    with pytest.raises(SeedRangeError):
        validate_seed("train", 1.0)
    with pytest.raises(SeedRangeError):
        validate_seed("train", True)
    with pytest.raises(ValueError):
        validate_seed("test", 1)


def test_role_of_and_seeds_for():
    assert role_of(5) is SeedRole.TRAIN
    assert role_of(1_000_500) is SeedRole.DIAGNOSIS
    assert role_of(2_000_001) is SeedRole.EVAL
    assert role_of(1_500_000) is None
    assert seeds_for("eval", 3) == [2_000_000, 2_000_001, 2_000_002]
    assert seeds_for("diagnosis", 2, offset=998) == [1_000_998, 1_000_999]
    with pytest.raises(SeedRangeError):
        seeds_for("diagnosis", 2, offset=999)
