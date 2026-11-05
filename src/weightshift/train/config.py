"""Training configuration: one TOML file per attempt (DELIVERY_3_ACCEPTANCE §2).

    name = "attempt01"
    beta = 0.1                 # shaping strength, FINAL_PLAN §6.2
    total_steps = 10_000_000
    seeds = [0, 1, 2]
    [mix]                      # training family weights (normalised)
    open = 0.5
    detour = 0.2
    trap = 0.15
    T0 = 0.15
    [train_episode_steps]      # optional: shorter practice episodes for some families
    T0 = 12
    [fence.T0]                 # optional: the disclosed failure induction (DELIVERY_3 §2)
    rows = [4, 8]              # inclusive cell rectangle; entering it in a practice
    cols = [3, 6]              # episode ends the episode with -penalty
    penalty = 5.0
    [ppo]                      # optional overrides of the CleanRL defaults
    ent_coef = 0.01

The config hash covers everything that changes training (not `name`, not
`seeds`), so two files with the same settings share a hash.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

from ..env.constants import EPISODE_LEN
from ..env.generate import FAMILIES, FIXED_LAYOUTS
from ..policy.actor import ALLOWED_WIDTHS


@dataclass(frozen=True)
class PPOParams:
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    num_envs: int = 64
    num_steps: int = 128  # rollout length per env
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    ent_coef_final: float | None = None  # if set, ent_coef anneals linearly to this (like the learning rate)
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 32  # actor width (FINAL_PLAN D5: widen to 64 only if capacity fails)
    critic_hidden: int = 64

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


@dataclass(frozen=True)
class TrainConfig:
    name: str
    beta: float
    total_steps: int
    mix: dict[str, float]
    seeds: tuple[int, ...] = (0, 1, 2)
    checkpoint_every: int = 250_000
    eval_episodes: int = 128  # per family, select seeds, at each checkpoint
    train_episode_steps: dict[str, int] = field(default_factory=dict)  # training only; evaluation is always 200
    fence: dict[str, dict] = field(default_factory=dict)  # training only; family -> {rows, cols, penalty}
    ppo: PPOParams = field(default_factory=PPOParams)
    # Fine-tuning (DELIVERY_3_ACCEPTANCE §8 note 8): start from another run's full.pt
    # ("{seed}" is replaced by the seed), and switch off `prune_units` units per hidden
    # layer of the actor (the lowest outgoing-L2 units of the rounded starting actor,
    # the same choice as gate E4). They stay exactly zero throughout training.
    init_from: str | None = None
    prune_units: int = 0

    def __post_init__(self):
        if self.beta < 0:
            raise ValueError("beta must be >= 0")
        if self.total_steps <= 0 or self.checkpoint_every <= 0:
            raise ValueError("total_steps and checkpoint_every must be positive")
        bad = sorted(set(self.mix) - set(FAMILIES + FIXED_LAYOUTS))
        if bad:
            raise ValueError(f"unknown families in mix: {bad}")
        if not self.mix or any(w < 0 for w in self.mix.values()) or sum(self.mix.values()) <= 0:
            raise ValueError("mix weights must be >= 0 with a positive sum")
        for fam, n in self.train_episode_steps.items():
            if fam not in FAMILIES + FIXED_LAYOUTS:
                raise ValueError(f"unknown family in train_episode_steps: {fam!r}")
            if not 1 <= n <= EPISODE_LEN:
                raise ValueError(f"train_episode_steps[{fam}] must be 1..{EPISODE_LEN}, got {n}")
        for fam, f in self.fence.items():
            if fam not in FAMILIES + FIXED_LAYOUTS:
                raise ValueError(f"unknown family in fence: {fam!r}")
            if set(f) != {"rows", "cols", "penalty"}:
                raise ValueError(f"fence.{fam} needs exactly rows, cols and penalty")
            for k in ("rows", "cols"):
                lo, hi = f[k]
                if not 0 <= lo <= hi <= 9:
                    raise ValueError(f"fence.{fam}.{k} must be [lo, hi] within 0..9")
            if f["penalty"] < 0:
                raise ValueError(f"fence.{fam}.penalty must be >= 0")
        if self.ppo.hidden not in ALLOWED_WIDTHS:
            raise ValueError(f"ppo.hidden must be one of {ALLOWED_WIDTHS}, got {self.ppo.hidden}")
        if self.prune_units and not self.init_from:
            raise ValueError("prune_units needs init_from (units are chosen on a trained actor)")
        if not 0 <= self.prune_units < self.ppo.hidden:
            raise ValueError(f"prune_units must be 0..{self.ppo.hidden - 1}")
        if self.ppo.batch_size % self.ppo.num_minibatches:
            raise ValueError("num_envs * num_steps must divide by num_minibatches")

    @property
    def mix_normalised(self) -> dict[str, float]:
        total = sum(self.mix.values())
        return {k: v / total for k, v in sorted(self.mix.items()) if v > 0}

    def hashed_fields(self) -> dict:
        d = asdict(self)
        d.pop("name")
        d.pop("seeds")
        d["mix"] = self.mix_normalised
        if not self.train_episode_steps:
            d.pop("train_episode_steps")  # keeps the hashes of earlier attempts unchanged
        if not self.fence:
            d.pop("fence")  # likewise
        if self.ppo.ent_coef_final is None:
            d["ppo"].pop("ent_coef_final")  # likewise
        if self.init_from is None:
            d.pop("init_from")  # likewise
        if not self.prune_units:
            d.pop("prune_units")  # likewise
        return d

    @property
    def config_hash(self) -> str:
        blob = json.dumps(self.hashed_fields(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def to_json(self) -> dict:
        d = asdict(self)
        d["config_hash"] = self.config_hash
        return d


def config_from_dict(d: dict) -> TrainConfig:
    d = dict(d)
    ppo_d = dict(d.pop("ppo", {}) or {})
    known = {f.name for f in fields(PPOParams)}
    unknown = sorted(set(ppo_d) - known)
    if unknown:
        raise ValueError(f"unknown [ppo] keys: {unknown}")
    known_top = {f.name for f in fields(TrainConfig)} - {"ppo"}
    unknown = sorted(set(d) - known_top)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    if "seeds" in d:
        d["seeds"] = tuple(int(s) for s in d["seeds"])
    d["mix"] = {str(k): float(v) for k, v in d["mix"].items()}
    if "fence" in d:
        d["fence"] = {
            str(k): {"rows": [int(x) for x in v["rows"]], "cols": [int(x) for x in v["cols"]], "penalty": float(v["penalty"])}
            if isinstance(v, dict) and {"rows", "cols", "penalty"} <= set(v)
            else dict(v)
            for k, v in d["fence"].items()
        }
    if "train_episode_steps" in d:
        d["train_episode_steps"] = {str(k): int(v) for k, v in d["train_episode_steps"].items()}
    return TrainConfig(**d, ppo=replace(PPOParams(), **ppo_d))


def load_config(path: str | Path) -> TrainConfig:
    with open(path, "rb") as f:
        return config_from_dict(tomllib.load(f))
