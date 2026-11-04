"""Delivery 2, acceptance B: actor file, forward pass, policy player."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from weightshift.env import generate_layout, seeds_for
from weightshift.env.game import Game, player_rng
from weightshift.evaluation import evaluate
from weightshift.players.scripted import make_player
from weightshift.policy import (
    ACTOR_SPEC,
    N_PARAMS,
    ActorError,
    PolicyPlayer,
    action_probs,
    forward,
    load_actor,
    random_actor,
    save_actor,
)
from weightshift.policy.actor import greedy_action, layer_names, n_params, sample_action, softmax
from weightshift.serialize import dump_actor
from weightshift.trajectory import run_episode

ROOT = Path(__file__).resolve().parents[1]


def zero_actor():
    return {n: np.zeros(s) for n, s in ACTOR_SPEC}


# ------------------------------------------------------------------ B1
def test_param_count_and_layers():
    a = random_actor(0)
    assert N_PARAMS == 20_519 == n_params(a)
    assert layer_names(a) == ["L1", "L2", "L3"]


def test_save_load_bit_exact(tmp_path):
    a = random_actor(1)
    p = save_actor(tmp_path / "a.npz", a)
    b = load_actor(p)
    assert list(b) == [n for n, _ in ACTOR_SPEC]
    for k in a:
        assert b[k].dtype == np.float64
        assert np.array_equal(a[k], b[k])


def test_float32_input_is_accepted_and_widened(tmp_path):
    a = {k: v.astype(np.float32) for k, v in random_actor(2).items()}
    b = load_actor(save_actor(tmp_path / "a.npz", a))
    assert all(b[k].dtype == np.float64 and np.array_equal(b[k], a[k].astype(np.float64)) for k in a)


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda a: a.pop("L2.b"), "missing"),
        (lambda a: a.__setitem__("L4.W", np.zeros((7, 7))), "unexpected"),
        (lambda a: a.__setitem__("L1.b", np.zeros(33)), "shape"),
        (lambda a: a.__setitem__("L1.W", np.zeros((32, 600))), "transposed"),
        (lambda a: a["L2.W"].__setitem__((3, 4), np.nan), "NaN"),
        (lambda a: a["L3.b"].__setitem__(0, np.inf), "NaN or infinite"),
    ],
)
def test_bad_actor_rejected(tmp_path, mutate, msg):
    a = random_actor(3)
    mutate(a)
    with pytest.raises(ActorError, match=msg):
        save_actor(tmp_path / "bad.npz", a)
    # and on load, for a file written without validation
    path = tmp_path / "raw.npz"
    np.savez(path, **a)
    with pytest.raises(ActorError, match=msg):
        load_actor(path)


# ------------------------------------------------------------------ B2
def test_forward_matches_hand_computation():
    a = zero_actor()
    a["L1.W"][0, 0] = 2.0
    a["L1.W"][1, 0] = -1.0
    a["L1.b"][0] = 0.5
    a["L2.W"][0, 3] = 1.5
    a["L2.b"][3] = -0.25
    a["L3.W"][3, 6] = 4.0
    a["L3.b"][2] = 1.0
    x = np.zeros(600)
    x[0], x[1] = 1.0, 1.0
    h1 = np.tanh(2.0 - 1.0 + 0.5)
    h2 = np.tanh(1.5 * h1 - 0.25)
    want = np.zeros(7)
    want[6] = 4.0 * h2
    want[2] = 1.0
    assert np.allclose(forward(a, x), want, rtol=0, atol=1e-15)
    p = action_probs(a, x)
    assert p.shape == (7,) and abs(p.sum() - 1.0) < 1e-12
    e = np.exp(want - want.max())
    assert np.allclose(p, e / e.sum(), rtol=0, atol=1e-15)


def test_batch_forward_matches_single():
    a = random_actor(4)
    xs = (np.random.default_rng(0).random((50, 600)) < 0.1).astype(np.float32)
    batch = action_probs(a, xs)
    assert np.allclose(batch.sum(axis=1), 1.0)
    for i in range(0, 50, 7):
        assert np.allclose(batch[i], action_probs(a, xs[i]), rtol=0, atol=1e-14)


def test_sampling_and_greedy_helpers():
    p = np.array([0.1, 0.0, 0.6, 0.3, 0.0, 0.0, 0.0])
    assert sample_action(p, 0.0) == 0
    assert sample_action(p, 0.0999) == 0
    assert sample_action(p, 0.1) == 2
    assert sample_action(p, 0.6999) == 2
    assert sample_action(p, 0.7) == 3
    assert sample_action(p, 0.999999999) == 3
    assert greedy_action(np.array([1.0, 3.0, 3.0, 0.0])) == 1  # ties -> lowest index
    assert np.allclose(softmax(np.array([1000.0, 1000.0])), [0.5, 0.5])  # no overflow


# ------------------------------------------------------------------ B3
@pytest.mark.parametrize("i, j", [(0, 0), (37, 5), (599, 31), (300, 17)])
def test_row_is_input(i, j):
    """Only input i moves hidden unit j when L1.W[i, j] is the only nonzero weight."""
    a = zero_actor()
    a["L1.W"][i, j] = 1.0
    for k in range(600):
        x = np.zeros(600)
        x[k] = 1.0
        h1 = np.tanh(x @ a["L1.W"] + a["L1.b"])
        if k == i:
            assert h1[j] == pytest.approx(np.tanh(1.0)) and np.count_nonzero(h1) == 1
        else:
            assert not h1.any()


def test_dump_rejects_out_in_layout():
    a = random_actor(5)
    a["L1.W"] = a["L1.W"].T.copy()  # PyTorch's nn.Linear.weight layout
    with pytest.raises(ActorError, match="transposed"):
        dump_actor(a)


# ------------------------------------------------------------------ B4
def test_sampled_episodes_are_deterministic():
    a = random_actor(6)
    lay = generate_layout("T0")
    for seed in seeds_for("eval", 200):
        t1 = run_episode(lay, seed, PolicyPlayer(a, "sampled")).to_jsonl()
        t2 = run_episode(lay, seed, PolicyPlayer(a, "sampled")).to_jsonl()
        assert t1 == t2


def test_greedy_episodes_are_deterministic_and_differ_from_sampled():
    a = random_actor(7)
    lay = generate_layout("open", 2_000_001, role="eval")
    g1 = run_episode(lay, 2_000_001, PolicyPlayer(a, "greedy"))
    g2 = run_episode(lay, 2_000_001, PolicyPlayer(a, "greedy"))
    s = run_episode(lay, 2_000_001, PolicyPlayer(a, "sampled"))
    assert g1.to_jsonl() == g2.to_jsonl()
    assert g1.actions != s.actions


def test_sampled_player_uses_the_episode_rng():
    """The first sampled action is the inverse-CDF of player_rng(seed)'s first draw."""
    a = random_actor(8)
    lay = generate_layout("T0")
    game = Game(lay, 1_000_005)
    pl = PolicyPlayer(a, "sampled")
    pl.reset(game)
    from weightshift.env.observation import encode_observation

    u = player_rng(1_000_005).random()
    assert pl.act(game) == sample_action(action_probs(a, encode_observation(game)), u)


def test_policy_player_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        PolicyPlayer(random_actor(0), mode="argmax")


# ------------------------------------------------------------------ B5
def test_make_player_policy_spec(tmp_path):
    path = save_actor(tmp_path / "a.npz", random_actor(9))
    p = make_player(("policy", str(path), "greedy"))
    assert isinstance(p, PolicyPlayer) and p.mode == "greedy" and p.name == "policy:greedy"
    with pytest.raises(ValueError):
        make_player(("policy", str(path)))
    with pytest.raises(ValueError):
        make_player(("oracle", str(path), "greedy"))


def test_policy_evaluation_identical_serial_and_parallel(tmp_path):
    path = save_actor(tmp_path / "a.npz", random_actor(10))
    seeds = seeds_for("eval", 64)
    spec = ("policy", str(path), "sampled")
    a = evaluate(spec, "T0", seeds, workers=1)
    b = evaluate(spec, "T0", seeds, workers=4)
    assert a == b
    assert [r.seed for r in a] == seeds


# ------------------------------------------------------------------ B6
def test_policy_benchmark_runs():
    spec = importlib.util.spec_from_file_location("bench_steps", ROOT / "scripts" / "bench_steps.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    rate, steps = bench.bench_policy("T0", seconds=0.2)
    assert steps >= 200 and rate > 0


def test_rewritten_actor_file_is_reloaded(tmp_path):
    """Save, evaluate, overwrite, evaluate again in one process: the new weights are used."""
    path = save_actor(tmp_path / "a.npz", random_actor(20))
    first = make_player(("policy", str(path), "greedy")).params
    save_actor(path, random_actor(21))
    second = make_player(("policy", str(path), "greedy")).params
    assert np.array_equal(second["L3.b"], random_actor(21)["L3.b"])
    assert not np.array_equal(first["L3.b"], second["L3.b"])
    seeds = seeds_for("eval", 8)
    spec = ("policy", str(path), "sampled")
    np.savez(path, **random_actor(22))  # rewritten by something other than save_actor
    a = evaluate(spec, "T0", seeds, workers=1)
    b = [r.metrics for r in evaluate(spec, "T0", seeds, workers=1)]
    fresh = [
        run_episode(generate_layout("T0"), s, PolicyPlayer(load_actor(path), "sampled")) for s in seeds
    ]
    from weightshift.metrics import compute_metrics

    assert [r.metrics for r in a] == b == [compute_metrics(t) for t in fresh]
