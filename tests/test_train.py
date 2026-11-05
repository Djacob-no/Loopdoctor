"""Delivery 3, acceptance C: training code (shaping, export/transpose, determinism, resume, smoke)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from weightshift.env import WeightShiftEnv, layout_from_spec
from weightshift.env.constants import DOWN, PICK, RIGHT, UP
from weightshift.env.game import Game
from weightshift.policy import forward, load_actor
from weightshift.policy.actor import actor_sha256
from weightshift.train.config import config_from_dict, load_config
from weightshift.train.envs import ShapingWrapper, TrainVecEnv, shaping_bonus
from weightshift.train.export import actor_to_params, assert_no_shared_parameters, save_actor_checkpoint, torch_logits
from weightshift.train.ppo import Agent, setup_torch, train


def tiny_cfg(**over):
    d = {
        "name": "tiny",
        "beta": 0.1,
        "total_steps": 20_480,
        "checkpoint_every": 4096,
        "mix": {"open": 1.0, "T0": 1.0},
        "ppo": {"num_envs": 16, "num_steps": 128, "num_minibatches": 4, "update_epochs": 2},
    }
    d.update(over)
    return config_from_dict(d)


# ------------------------------------------------------------------ config
def test_config_hash_ignores_name_and_seeds_and_normalises_mix():
    a = tiny_cfg()
    b = tiny_cfg(name="other", seeds=[5, 6, 7], mix={"open": 2.0, "T0": 2.0})
    c = tiny_cfg(beta=0.2)
    assert a.config_hash == b.config_hash != c.config_hash


@pytest.mark.parametrize(
    "over, msg",
    [
        ({"mix": {"nowhere": 1.0}}, "unknown families"),
        ({"mix": {"open": 0.0}}, "positive sum"),
        ({"beta": -1.0}, "beta"),
        ({"bogus": 1}, "unknown config keys"),
        ({"ppo": {"lr": 1.0}}, "unknown \\[ppo\\] keys"),
        ({"ppo": {"num_envs": 3, "num_steps": 3, "num_minibatches": 4}}, "divide"),
        ({"ppo": {"hidden": 48}}, "ppo.hidden"),
        ({"prune_units": 4}, "needs init_from"),
    ],
)
def test_config_rejects_bad_values(over, msg):
    with pytest.raises(ValueError, match=msg):
        tiny_cfg(**over)


def test_config_files_load(tmp_path):
    from pathlib import Path

    for path in sorted(Path(__file__).resolve().parents[1].glob("configs/*.toml")):
        cfg = load_config(path)
        assert cfg.name == path.stem and len(cfg.config_hash) == 12


# ------------------------------------------------------------------ C1
def test_actor_and_critic_share_nothing():
    assert_no_shared_parameters(Agent())


def test_shared_parameters_are_detected():
    agent = Agent()
    agent.critic[0] = agent.actor[0]  # the critic reuses the actor's first layer
    with pytest.raises(AssertionError, match="share"):
        assert_no_shared_parameters(agent)


# ------------------------------------------------------------------ C2: transpose
@pytest.mark.parametrize("hidden", [32, 64])
def test_exported_actor_matches_torch(tmp_path, hidden):
    setup_torch(0)
    agent = Agent(hidden=hidden)
    with torch.no_grad():  # make every weight non-trivial, so a transpose cannot hide
        for p in agent.actor.parameters():
            p.normal_(0, 0.3)
    path = save_actor_checkpoint(tmp_path / "a.npz", agent)
    params = load_actor(path)
    assert params["L1.W"].shape == (600, hidden) and params["L3.W"].shape == (hidden, 7)
    obs = (np.random.default_rng(0).random((1000, 600)) < 0.1).astype(np.float32)
    assert np.max(np.abs(forward(params, obs) - torch_logits(agent, obs))) <= 1e-5


def test_export_is_a_transpose_of_nn_linear_weight():
    agent = Agent()
    p = actor_to_params(agent)
    assert np.array_equal(p["L1.W"], agent.actor[0].weight.detach().numpy().astype(np.float64).T)
    assert np.array_equal(p["L3.b"], agent.actor[4].bias.detach().numpy().astype(np.float64))


# ------------------------------------------------------------------ C3: shaping
# A layout where Manhattan and BFS distance to the dock differ: a wall of racks
# between the robot and the dock.
#   row 1: robot at (1,4); row 3: racks at cols 2..7; dock (8,4)
WALL = {(3, c): "ABCDAB"[c - 2] for c in range(2, 8)}


@pytest.fixture
def wall_layout():
    return layout_from_spec(WALL, (8, 4), [(1, 4)], name="wall")


def test_bonus_formula():
    from weightshift.env.layout import idx

    dock = idx(8, 4)
    assert shaping_bonus(0.5, idx(1, 4), idx(2, 4), dock, carrying=True) == 0.5  # one step closer
    assert shaping_bonus(0.5, idx(2, 4), idx(2, 3), dock, carrying=True) == -0.5  # one step away
    assert shaping_bonus(0.5, idx(1, 4), idx(2, 4), dock, carrying=False) == 0.0
    assert shaping_bonus(0.0, idx(1, 4), idx(2, 4), dock, carrying=True) == 0.0


def test_wrapper_shapes_only_while_carrying_and_uses_manhattan(wall_layout):
    env = ShapingWrapper(WeightShiftEnv(layout=wall_layout), beta=0.25)
    env.reset(seed=1)
    g = env.unwrapped.game
    # empty-handed: moving towards the dock earns nothing
    _, r, _, _, info = env.step(DOWN)  # (1,4) -> (2,4)
    assert g.pos == (2, 4) and r == 0.0 and info["shaping"] == 0.0 and info["unshaped_reward"] == 0.0
    _, r, _, _, info = env.step(PICK)  # rack (3,4) is below
    assert g.n_carried == 1 and r == 0.0
    # carrying: a sideways step away from the dock column is penalised by Manhattan distance,
    # even though it is the first step of the shortest (BFS) route around the wall
    _, r, _, _, info = env.step(RIGHT)
    assert info["shaping"] == pytest.approx(-0.25) and info["unshaped_reward"] == 0.0
    _, r, _, _, info = env.step(UP)
    assert info["shaping"] == pytest.approx(-0.25)
    assert g.pos == (1, 5)
    _, r, _, _, info = env.step(DOWN)  # back towards the dock
    assert g.pos == (2, 5) and info["shaping"] == pytest.approx(0.25)
    # a bump does not move: no shaping
    _, r, _, _, info = env.step(DOWN)  # (3,5) is a rack
    assert g.pos == (2, 5) and info["shaping"] == 0.0


def test_vec_env_reward_is_score_delta_plus_bonus(wall_layout):
    """Recompute every shaped reward of a random rollout from the games' own records."""
    env = TrainVecEnv(8, {"open": 1.0, "T0": 1.0}, beta=0.3, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(400):
        before = [(g.p, g.n_carried > 0, g.score, g.layout.dock) for g in env.games]
        acts = rng.integers(0, 7, size=8)
        games = list(env.games)
        _, rew, done, _ = env.step(acts)
        for i in range(8):
            g = games[i]  # the game that took the step (replaced afterwards if done)
            p0, carrying, s0, dock = before[i]
            want = (g.score - s0) + shaping_bonus(0.3, p0, g.p, dock, carrying)
            assert rew[i] == pytest.approx(want, abs=1e-6)


def test_vec_env_uses_train_seeds_and_the_mix():
    env = TrainVecEnv(64, {"open": 1.0, "detour": 1.0}, beta=0.1, seed=3)
    assert set(env.ep_family) == {"open", "detour"}
    assert all(0 <= g.episode_seed < 1_000_000 for g in env.games)


# ------------------------------------------------------------------ C4 / C5
def test_training_is_deterministic(tmp_path):
    cfg = tiny_cfg()
    a = train(cfg, 0, tmp_path / "a", log=lambda *_: None)
    b = train(cfg, 0, tmp_path / "b", log=lambda *_: None)
    ca = sorted((a / "checkpoints").glob("*.npz"))
    cb = sorted((b / "checkpoints").glob("*.npz"))
    assert [p.name for p in ca] == [p.name for p in cb] and len(ca) == 5
    assert [actor_sha256(p) for p in ca] == [actor_sha256(p) for p in cb]
    c = train(cfg, 1, tmp_path / "c", log=lambda *_: None)
    assert actor_sha256(sorted((c / "checkpoints").glob("*.npz"))[-1]) != actor_sha256(ca[-1])


def test_resume_gives_the_same_result(tmp_path):
    cfg = tiny_cfg()
    full = train(cfg, 0, tmp_path / "full", log=lambda *_: None)
    part = train(cfg, 0, tmp_path / "part", stop_after_updates=4, log=lambda *_: None)
    assert len(list((part / "checkpoints").glob("*.npz"))) == 2
    part = train(cfg, 0, tmp_path / "part", log=lambda *_: None)  # resumes from full.pt
    last = lambda d: sorted((d / "checkpoints").glob("*.npz"))[-1]  # noqa: E731
    assert last(full).name == last(part).name
    assert actor_sha256(last(full)) == actor_sha256(last(part))
    rows = [json.loads(x) for x in (part / "metrics.jsonl").read_text().splitlines()]
    assert [r["global_step"] for r in rows] == [4096, 8192, 12288, 16384, 20480]


def test_resume_refuses_another_config(tmp_path):
    train(tiny_cfg(), 0, tmp_path, stop_after_updates=2, log=lambda *_: None)
    other = tiny_cfg(beta=0.5, name="tiny")
    # a different config has a different run directory, so nothing is resumed by accident
    from weightshift.train.ppo import load_full, run_dir

    full = run_dir(tmp_path, tiny_cfg(), 0) / "full.pt"
    with pytest.raises(ValueError, match="different config"):
        load_full(full, other, 0)


# ------------------------------------------------------------------ C7: smoke
def test_ppo_learns_to_deliver_on_open(tmp_path):
    from weightshift.env import seeds_for
    from weightshift.evaluation import evaluate

    cfg = config_from_dict(
        {
            "name": "smoke",
            "beta": 0.1,
            "total_steps": 600_000,
            "checkpoint_every": 600_000,
            "mix": {"open": 1.0},
            "ppo": {"learning_rate": 1e-3},  # the fastest setting in the sweeps (TRAINING_LOG.md)
        }
    )
    rd = train(cfg, 0, tmp_path, log=lambda *_: None)
    actor = sorted((rd / "checkpoints").glob("*.npz"))[-1]
    res = evaluate(("policy", str(actor), "sampled"), "open", seeds_for("select", 256), role="select", workers=1)
    success = sum(r.metrics.items_delivered > 0 for r in res) / len(res)
    assert success >= 0.30, success  # random play: 12.4%


# ------------------------------------------------------------------ the short practice clock on T0
def test_short_practice_episodes_only_for_the_named_family():
    env = TrainVecEnv(32, {"open": 1.0, "T0": 1.0}, beta=0.1, seed=4, episode_steps={"T0": 12})
    lengths: dict[str, set[int]] = {"open": set(), "T0": set()}
    rng = np.random.default_rng(0)
    for _ in range(450):
        games, fams = list(env.games), list(env.ep_family)
        _, _, done, fin = env.step(rng.integers(0, 7, size=32))
        for i in np.flatnonzero(done):
            lengths[fams[i]].add(games[i].t)
    assert lengths["T0"] == {12} and lengths["open"] == {200}


def test_fastest_t0_delivery_is_longer_than_the_practice_clock():
    """The disclosed failure induction relies on this: no delivery fits in 12 steps on T0."""
    from weightshift.env import generate_layout

    t0 = generate_layout("T0")
    fastest = min(
        t0.distances_from(s)[p] + 1 + t0.distances_from(p)[t0.dock] + 1
        for s in t0.spawns
        for p in range(100)
        if t0.pick_sku_at[p] >= 0 and t0.distances_from(s)[p] >= 0
    )
    assert fastest == 14


@pytest.mark.parametrize("steps, msg", [({"nowhere": 10}, "unknown family"), ({"T0": 0}, "1..200"), ({"T0": 201}, "1..200")])
def test_train_episode_steps_validated(steps, msg):
    with pytest.raises(ValueError, match=msg):
        tiny_cfg(train_episode_steps=steps)


def test_train_episode_steps_change_the_hash_only_when_set():
    assert tiny_cfg().config_hash == tiny_cfg(train_episode_steps={}).config_hash
    assert tiny_cfg().config_hash != tiny_cfg(train_episode_steps={"T0": 12}).config_hash


def test_entropy_annealing_option(tmp_path):
    assert tiny_cfg().config_hash == tiny_cfg(ppo={"num_envs": 16, "num_steps": 128, "num_minibatches": 4, "update_epochs": 2}).config_hash
    annealed = tiny_cfg(ppo={"num_envs": 16, "num_steps": 128, "num_minibatches": 4, "update_epochs": 2, "ent_coef_final": 0.0})
    assert annealed.config_hash != tiny_cfg().config_hash
    rd = train(annealed, 0, tmp_path, log=lambda *_: None)
    rows = [json.loads(x) for x in (rd / "metrics.jsonl").read_text().splitlines()]
    coefs = [r["train"]["ent_coef"] for r in rows]
    assert coefs == sorted(coefs, reverse=True) and coefs[0] < 0.01 and coefs[-1] < 0.001


# ------------------------------------------------------------------ the electric fence on T0
T0_FENCE = {"T0": {"rows": [4, 8], "cols": [3, 6], "penalty": 5.0}}


def test_fence_ends_t0_practice_episodes_with_a_penalty_and_leaves_others_alone():
    env = TrainVecEnv(32, {"open": 1.0, "T0": 1.0}, beta=0.0, seed=5, fence=T0_FENCE)
    rng = np.random.default_rng(1)
    seen_fenced = 0
    for _ in range(3000):
        games, fams = list(env.games), list(env.ep_family)
        _, rew, done, fin = env.step(rng.integers(0, 7, size=32))
        by_game = iter(fin)
        for i in np.flatnonzero(done):
            rec = next(by_game)
            g = games[i]
            r, c = divmod(g.p, 10)
            inside = fams[i] == "T0" and 4 <= r <= 8 and 3 <= c <= 6
            assert rec["fenced"] == inside
            if inside:
                seen_fenced += 1
                assert rew[i] <= -5.0 + 1e-6 and g.t < 200
            else:
                assert g.t == 200
        # a T0 game that is still running is never inside the fence
        for g, f in zip(env.games, env.ep_family):
            if f == "T0":
                r, c = divmod(g.p, 10)
                assert not (4 <= r <= 8 and 3 <= c <= 6)
    assert seen_fenced > 0


def test_fence_covers_the_t0_pocket_and_dock_but_no_spawn_or_rack():
    from weightshift.env import generate_layout

    t0 = generate_layout("T0")
    cells = {r * 10 + c for r in range(4, 9) for c in range(3, 7)}
    assert t0.dock in cells
    assert not cells & set(t0.spawns)
    assert not cells & set(t0.rack_cells)


@pytest.mark.parametrize(
    "fence, msg",
    [
        ({"nowhere": {"rows": [0, 1], "cols": [0, 1], "penalty": 1.0}}, "unknown family"),
        ({"T0": {"rows": [4, 8], "cols": [3, 6]}}, "exactly rows, cols and penalty"),
        ({"T0": {"rows": [8, 4], "cols": [3, 6], "penalty": 1.0}}, "within 0..9"),
        ({"T0": {"rows": [4, 8], "cols": [3, 6], "penalty": -1.0}}, ">= 0"),
    ],
)
def test_fence_validated(fence, msg):
    with pytest.raises(ValueError, match=msg):
        tiny_cfg(fence=fence)


def test_fence_changes_the_hash_only_when_set():
    assert tiny_cfg().config_hash == tiny_cfg(fence={}).config_hash != tiny_cfg(fence=T0_FENCE).config_hash


def test_actor_with_unsupported_width_is_rejected_clearly():
    from weightshift.policy.actor import ActorError, validate

    params = actor_to_params(Agent(hidden=64))
    validate(params)  # 64 is allowed
    bad = {k: v[..., :48] if k in ("L1.W", "L1.b") else v for k, v in params.items()}
    with pytest.raises(ActorError, match="hidden width 48"):
        validate(bad)


def test_fine_tune_keeps_pruned_units_at_exactly_zero(tmp_path):
    from weightshift.gates import ablate_capacity
    from weightshift.serialize import rounded

    base = tiny_cfg(total_steps=4096, checkpoint_every=4096)
    rd = train(base, 0, tmp_path / "base", log=lambda *_: None)
    cfg = tiny_cfg(
        name="ft", total_steps=8192, checkpoint_every=4096,
        init_from=str(rd / "full.pt").replace("-s0", "-s{seed}"), prune_units=4,
    )
    assert cfg.config_hash != base.config_hash
    ft = train(cfg, 0, tmp_path / "ft", log=lambda *_: None)
    info = json.loads((ft / "init.json").read_text(encoding="utf-8"))
    frozen = info["frozen_units"]
    assert {k: len(v) for k, v in frozen.items()} == {"L1": 4, "L2": 4}
    actor = load_actor(sorted((ft / "checkpoints").glob("*.npz"))[-1])
    for layer, nxt in (("L1", "L2"), ("L2", "L3")):
        u = frozen[layer]
        assert not actor[f"{layer}.W"][:, u].any() and not actor[f"{layer}.b"][u].any()
        assert not actor[f"{nxt}.W"][u, :].any()
    # gate E4 would zero exactly these units, so they are genuinely spare
    _, zeroed = ablate_capacity(rounded(actor))
    assert zeroed == frozen
    # ...and the other units did train
    start = load_actor(sorted((rd / "checkpoints").glob("*.npz"))[-1])
    assert not np.array_equal(start["L1.W"], actor["L1.W"])
