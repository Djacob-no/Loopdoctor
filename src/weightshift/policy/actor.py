"""The actor (policy network) as plain numpy arrays.

An actor is an ordered dict of named float64 tensors. Weight matrices are
stored **(inputs, outputs)**: row i of `L1.W` holds the weights leaving
input i (the Exp 4 convention, FINAL_PLAN §7.1). PyTorch's `nn.Linear.weight`
is the transpose; converting it is the trainer's job (Delivery 3).

The serializer, applier and analysis code work on any such dict (for example
Exp 4's `embed.W ... actor_head.b`); only `forward` assumes our 600->32->32->7
tanh network.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ..env.constants import N_ACTIONS, OBS_SIZE

Params = dict[str, np.ndarray]

HIDDEN = 32

# FINAL_PLAN D5: width 32, widened to 64 only if the capacity check fails.
ALLOWED_WIDTHS = (32, 64)

Spec = tuple[tuple[str, tuple[int, ...]], ...]


def actor_spec(hidden: int = HIDDEN) -> Spec:
    """Our actor: name -> shape, in dump order."""
    return (
        ("L1.W", (OBS_SIZE, hidden)),
        ("L1.b", (hidden,)),
        ("L2.W", (hidden, hidden)),
        ("L2.b", (hidden,)),
        ("L3.W", (hidden, N_ACTIONS)),
        ("L3.b", (N_ACTIONS,)),
    )


ACTOR_SPEC: Spec = actor_spec(HIDDEN)
N_PARAMS = sum(int(np.prod(s)) for _, s in ACTOR_SPEC)
assert N_PARAMS == 20_519, N_PARAMS


class ActorError(ValueError):
    """An actor file or dict does not match the expected format."""


def n_params(params: Mapping[str, np.ndarray]) -> int:
    return sum(int(a.size) for a in params.values())


def layer_names(params: Mapping[str, np.ndarray]) -> list[str]:
    """Layer prefixes in order, from the `.W` tensors (e.g. ['L1', 'L2', 'L3'])."""
    return [k[:-2] for k in params if k.endswith(".W")]


def _infer_spec(params: Mapping[str, np.ndarray]) -> Spec:
    b = params.get("L1.b")
    width = int(np.shape(b)[0]) if b is not None and np.ndim(b) == 1 else None
    if width is None:
        return actor_spec(HIDDEN)  # the name/shape checks report what is wrong
    if width not in ALLOWED_WIDTHS:
        raise ActorError(f"L1.b has shape ({width},): hidden width {width} is not one of {ALLOWED_WIDTHS}")
    return actor_spec(width)


def validate(params: Mapping[str, np.ndarray], spec: Spec | None = None) -> Params:
    """Check names, shapes and finiteness against `spec`; return a float64 copy in spec order.

    With no spec, the width is read from `L1.b` and must be one of ALLOWED_WIDTHS.
    """
    if spec is None:
        spec = _infer_spec(params)
    want = [n for n, _ in spec]
    got = list(params)
    if sorted(got) != sorted(want):
        missing = sorted(set(want) - set(got))
        extra = sorted(set(got) - set(want))
        raise ActorError(f"tensor names differ: missing {missing}, unexpected {extra}")
    out: Params = {}
    for name, shape in spec:
        a = np.asarray(params[name])
        if a.shape != tuple(shape):
            hint = ""
            if a.ndim == 2 and a.shape == tuple(shape)[::-1]:
                hint = " (looks transposed: W must be (inputs, outputs))"
            raise ActorError(f"{name} has shape {a.shape}, expected {tuple(shape)}{hint}")
        if not np.issubdtype(a.dtype, np.number):
            raise ActorError(f"{name} is not numeric ({a.dtype})")
        a = a.astype(np.float64, copy=True)
        if not np.all(np.isfinite(a)):
            raise ActorError(f"{name} contains NaN or infinite values")
        out[name] = a
    return out


def copy_params(params: Mapping[str, np.ndarray]) -> Params:
    return {k: np.array(v, dtype=np.float64, copy=True) for k, v in params.items()}


def save_actor(path: str | Path, params: Mapping[str, np.ndarray], spec: Spec | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    checked = validate(params, spec)
    with open(p, "wb") as f:
        np.savez(f, **checked)
    return p


def load_actor(path: str | Path, spec: Spec | None = None) -> Params:
    with np.load(Path(path), allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    return validate(raw, spec)


def actor_sha256(path: str | Path) -> str:
    """SHA-256 of an actor file's bytes (the experiment log's checkpoint_sha256)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


_CACHE_MAX = 32
_cache_by_digest: dict[str, Params] = {}


def load_actor_cached(path: str | Path) -> Params:
    """For processes that play many episodes with one actor file. Do not mutate the result.

    The file is read on every call and the parsed actor is cached by the
    SHA-256 of its bytes, so a rewritten file is always reloaded, even when
    its size and modification time are unchanged (two writes within one
    filesystem timestamp tick).
    """
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    actor = _cache_by_digest.get(digest)
    if actor is None:
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            actor = validate({k: z[k] for k in z.files})
        if len(_cache_by_digest) >= _CACHE_MAX:
            _cache_by_digest.pop(next(iter(_cache_by_digest)))
        _cache_by_digest[digest] = actor
    return actor


def random_actor(seed: int, scale: float = 1.0) -> Params:
    """A randomly initialised actor (tests and benchmarks). Fan-in scaled normal weights."""
    rng = np.random.default_rng([seed, 0xAC7])
    out: Params = {}
    for name, shape in ACTOR_SPEC:
        fan_in = shape[0] if len(shape) == 2 else HIDDEN
        out[name] = rng.normal(0.0, scale / np.sqrt(fan_in), size=shape)
    return out


# ---------------------------------------------------------------- forward
def forward(params: Mapping[str, np.ndarray], obs: np.ndarray) -> np.ndarray:
    """Action logits for one observation (600,) or a batch (n, 600), in float64."""
    x = np.asarray(obs, dtype=np.float64)
    h1 = np.tanh(x @ params["L1.W"] + params["L1.b"])
    h2 = np.tanh(h1 @ params["L2.W"] + params["L2.b"])
    return h2 @ params["L3.W"] + params["L3.b"]


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def action_probs(params: Mapping[str, np.ndarray], obs: np.ndarray) -> np.ndarray:
    return softmax(forward(params, obs))


def sample_action(probs: np.ndarray, u: float) -> int:
    """Inverse-CDF sample from one uniform u in [0, 1)."""
    cdf = np.cumsum(probs)
    a = int(np.searchsorted(cdf, u * cdf[-1], side="right"))
    return min(a, len(probs) - 1)


def greedy_action(probs_or_logits: np.ndarray) -> int:
    """Argmax; ties go to the lowest action index."""
    return int(np.argmax(probs_or_logits))
