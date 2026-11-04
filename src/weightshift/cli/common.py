"""Helpers shared by the command-line tools."""

from __future__ import annotations

from ..env.generate import FAMILIES, FIXED_LAYOUTS, generate_layout
from ..env.layout import Layout
from ..metrics import EpisodeMetrics

LAYOUT_CHOICES = FIXED_LAYOUTS + FAMILIES


def resolve_layout(name: str, seed: int, role: str | None = None) -> Layout:
    """T0 (fixed) or a family name; for families the seed is also the layout seed."""
    return generate_layout(name, seed, role=role)


def format_metrics(m: EpisodeMetrics) -> str:
    return "\n".join(
        [
            f"score {m.score}   orders completed {m.orders_completed}   items delivered {m.items_delivered}",
            f"picks {m.picks} (wrong {m.wrong_picks})   bumps {m.bumps}   steps {m.steps}",
            f"loop index {m.loop_index:.3f}   idle fraction {m.idle_fraction:.3f}"
            f"   full-but-not-delivering {m.full_not_delivering:.3f}",
        ]
    )
