"""Seed ranges per role, enforced in code (DELIVERY_1_ACCEPTANCE.md §2 "Seeds").

The experiment's credibility depends on the model never seeing evaluation
episodes, so every caller that knows which role a seed is for should pass the
role and let this module reject seeds outside its range.
"""

from __future__ import annotations

from enum import Enum


class SeedRole(str, Enum):
    TRAIN = "train"
    DIAGNOSIS = "diagnosis"
    EVAL = "eval"


# Half-open ranges [start, stop).
SEED_RANGES: dict[SeedRole, tuple[int, int]] = {
    SeedRole.TRAIN: (0, 1_000_000),
    SeedRole.DIAGNOSIS: (1_000_000, 1_001_000),
    SeedRole.EVAL: (2_000_000, 2_010_000),
}


class SeedRangeError(ValueError):
    """A seed was used outside the range declared for its role."""


def _as_role(role: SeedRole | str) -> SeedRole:
    try:
        return SeedRole(role)
    except ValueError as exc:
        raise ValueError(f"unknown seed role {role!r}; expected one of {[r.value for r in SeedRole]}") from exc


def validate_seed(role: SeedRole | str, seed: int) -> int:
    """Return `seed` unchanged if it is inside `role`'s range, otherwise raise SeedRangeError."""
    r = _as_role(role)
    start, stop = SEED_RANGES[r]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise SeedRangeError(f"seed must be an int, got {type(seed).__name__}")
    if not start <= seed < stop:
        raise SeedRangeError(f"seed {seed} is outside the {r.value} range [{start}, {stop})")
    return seed


def role_of(seed: int) -> SeedRole | None:
    """The role whose range contains `seed`, or None."""
    for role, (start, stop) in SEED_RANGES.items():
        if start <= seed < stop:
            return role
    return None


def seeds_for(role: SeedRole | str, n: int, offset: int = 0) -> list[int]:
    """The first `n` seeds of a role's range, starting at `offset` within it."""
    r = _as_role(role)
    start, stop = SEED_RANGES[r]
    first = start + offset
    if offset < 0 or first + n > stop:
        raise SeedRangeError(f"{n} seeds from offset {offset} do not fit in the {r.value} range")
    return list(range(first, first + n))
