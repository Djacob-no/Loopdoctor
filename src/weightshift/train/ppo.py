"""Single-file PPO for WeightShift, after CleanRL's ppo.py (FINAL_PLAN §5, §6.1).

- Actor 600 -> hidden -> hidden -> 7 (hidden = `ppo.hidden`, 32 or 64), tanh on
  the hidden layers and linear logits, and a **separate** critic
  600 -> critic_hidden -> critic_hidden -> 1. They share no parameters
  (asserted at every save).
- CPU only, one thread per run, seeded: the same config and seed give the
  same actor bit for bit, and resuming from a checkpoint gives the same result
  as an uninterrupted run (DELIVERY_3_ACCEPTANCE C4, C5).
- Optionally starts from another run's weights and keeps some actor units
  switched off (exactly zero) throughout: `init_from` / `prune_units`.
- Every `checkpoint_every` steps: actor.npz (Delivery 2 format), full.pt (for
  resume) and one metrics.jsonl row.
"""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical

from ..env.constants import N_ACTIONS, OBS_SIZE
from .config import TrainConfig
from .envs import TrainVecEnv
from .export import assert_no_shared_parameters, save_actor_checkpoint

CKPT_FORMAT = 1


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, hidden: int = 32, critic_hidden: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(OBS_SIZE, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, N_ACTIONS), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(OBS_SIZE, critic_hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(critic_hidden, critic_hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(critic_hidden, 1), std=1.0),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def setup_torch(seed: int) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


@dataclass
class TrainState:
    agent: Agent
    optimizer: torch.optim.Optimizer
    env: TrainVecEnv
    next_obs: torch.Tensor
    next_done: torch.Tensor
    update: int  # updates completed
    global_step: int
    next_checkpoint: int
    wall_s: float
    frozen: dict[str, list[int]] = field(default_factory=dict)  # actor units held at zero


# torch actor modules: actor[0] = L1, actor[2] = L2, actor[4] = L3 (logits)
_LAYER_MODULES = {"L1": (0, 2), "L2": (2, 4)}  # hidden layer -> (its Linear, the next Linear)


@torch.no_grad()
def apply_frozen(agent: Agent, frozen: dict[str, list[int]]) -> None:
    """Zero the incoming weights, bias and outgoing weights of the frozen units."""
    for layer, units in frozen.items():
        if not units:
            continue
        own, nxt = _LAYER_MODULES[layer]
        idx = torch.tensor(units, dtype=torch.long)
        agent.actor[own].weight[idx, :] = 0.0
        agent.actor[own].bias[idx] = 0.0
        agent.actor[nxt].weight[:, idx] = 0.0


def choose_frozen(agent: Agent, n_units: int) -> dict[str, list[int]]:
    """The units gate E4 would zero: lowest outgoing L2 on the rounded actor."""
    from ..gates import ablate_capacity  # local: gates pulls in evaluation
    from ..serialize import rounded
    from .export import actor_to_params

    _, zeroed = ablate_capacity(rounded(actor_to_params(agent)), n_units)
    return zeroed


def init_path(cfg: TrainConfig, seed: int) -> Path:
    return Path(cfg.init_from.format(seed=seed))


def new_state(cfg: TrainConfig, seed: int) -> TrainState:
    setup_torch(seed)
    p = cfg.ppo
    agent = Agent(p.hidden, p.critic_hidden)
    frozen: dict[str, list[int]] = {}
    if cfg.init_from:
        blob = torch.load(init_path(cfg, seed), weights_only=False)  # our own file
        agent.load_state_dict(blob["agent"])  # fails loudly on a width mismatch
        if cfg.prune_units:
            frozen = choose_frozen(agent, cfg.prune_units)
            apply_frozen(agent, frozen)
    assert_no_shared_parameters(agent)
    optimizer = torch.optim.Adam(agent.parameters(), lr=p.learning_rate, eps=1e-5)
    env = TrainVecEnv(p.num_envs, cfg.mix_normalised, cfg.beta, seed, cfg.train_episode_steps, cfg.fence)
    return TrainState(
        agent=agent,
        optimizer=optimizer,
        env=env,
        next_obs=torch.from_numpy(env.obs.copy()),
        next_done=torch.zeros(p.num_envs),
        update=0,
        global_step=0,
        next_checkpoint=cfg.checkpoint_every,
        wall_s=0.0,
        frozen=frozen,
    )


def num_updates(cfg: TrainConfig) -> int:
    return cfg.total_steps // cfg.ppo.batch_size


# --------------------------------------------------------------- checkpoints
def save_full(path: Path, st: TrainState, cfg: TrainConfig, seed: int) -> None:
    blob = {
        "format": CKPT_FORMAT,
        "config": cfg.to_json(),
        "seed": seed,
        "agent": st.agent.state_dict(),
        "optimizer": st.optimizer.state_dict(),
        "env": pickle.dumps(st.env),
        "next_obs": st.next_obs,
        "next_done": st.next_done,
        "update": st.update,
        "global_step": st.global_step,
        "next_checkpoint": st.next_checkpoint,
        "wall_s": st.wall_s,
        "frozen": st.frozen,
        "torch_rng": torch.get_rng_state(),
    }
    buf = io.BytesIO()
    torch.save(blob, buf)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(buf.getvalue())
    tmp.replace(path)


def load_full(path: Path, cfg: TrainConfig, seed: int) -> TrainState:
    blob = torch.load(path, weights_only=False)  # our own file
    if blob["format"] != CKPT_FORMAT or blob["config"]["config_hash"] != cfg.config_hash or blob["seed"] != seed:
        raise ValueError(f"{path} was written by a different config or seed")
    setup_torch(seed)
    p = cfg.ppo
    agent = Agent(p.hidden, p.critic_hidden)
    agent.load_state_dict(blob["agent"])
    optimizer = torch.optim.Adam(agent.parameters(), lr=p.learning_rate, eps=1e-5)
    optimizer.load_state_dict(blob["optimizer"])
    torch.set_rng_state(blob["torch_rng"])
    return TrainState(
        agent=agent,
        optimizer=optimizer,
        env=pickle.loads(blob["env"]),  # our own file
        next_obs=blob["next_obs"],
        next_done=blob["next_done"],
        update=blob["update"],
        global_step=blob["global_step"],
        next_checkpoint=blob["next_checkpoint"],
        wall_s=blob["wall_s"],
        frozen=blob.get("frozen", {}),
    )


# --------------------------------------------------------------- training
def train_update(st: TrainState, cfg: TrainConfig) -> dict:
    """One rollout of num_steps x num_envs, then the PPO update. Returns stats."""
    p = cfg.ppo
    agent, optimizer, env = st.agent, st.optimizer, st.env
    n_up = num_updates(cfg)
    frac = 1.0 - st.update / n_up
    if p.anneal_lr:
        optimizer.param_groups[0]["lr"] = frac * p.learning_rate
    ent_coef = p.ent_coef if p.ent_coef_final is None else p.ent_coef_final + frac * (p.ent_coef - p.ent_coef_final)

    obs = torch.zeros((p.num_steps, p.num_envs, OBS_SIZE))
    actions = torch.zeros((p.num_steps, p.num_envs), dtype=torch.long)
    logprobs = torch.zeros((p.num_steps, p.num_envs))
    rewards = torch.zeros((p.num_steps, p.num_envs))
    dones = torch.zeros((p.num_steps, p.num_envs))
    values = torch.zeros((p.num_steps, p.num_envs))
    finished: list[dict] = []

    next_obs, next_done = st.next_obs, st.next_done
    for step in range(p.num_steps):
        obs[step] = next_obs
        dones[step] = next_done
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
        values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob
        o, r, d, fin = env.step(action.numpy())
        rewards[step] = torch.from_numpy(r)
        next_obs = torch.from_numpy(o.copy())
        next_done = torch.from_numpy(d.astype(np.float32))
        finished.extend(fin)
    st.global_step += p.batch_size

    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards)
        lastgaelam = torch.zeros(p.num_envs)
        for t in reversed(range(p.num_steps)):
            if t == p.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + p.gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + p.gamma * p.gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values

    b_obs = obs.reshape((-1, OBS_SIZE))
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    b_inds = np.arange(p.batch_size)
    perm_rng = np.random.default_rng([st.update, 0x5EED])  # minibatch order: deterministic per update
    clipfracs = []
    for _ in range(p.update_epochs):
        perm_rng.shuffle(b_inds)
        for start in range(0, p.batch_size, p.minibatch_size):
            mb = b_inds[start : start + p.minibatch_size]
            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb], b_actions[mb])
            logratio = newlogprob - b_logprobs[mb]
            ratio = logratio.exp()
            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs.append(((ratio - 1.0).abs() > p.clip_coef).float().mean().item())
            mb_adv = b_advantages[mb]
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
            pg_loss = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 1 - p.clip_coef, 1 + p.clip_coef)).mean()
            newvalue = newvalue.view(-1)
            v_unclipped = (newvalue - b_returns[mb]) ** 2
            v_clipped = b_values[mb] + torch.clamp(newvalue - b_values[mb], -p.clip_coef, p.clip_coef)
            v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - b_returns[mb]) ** 2).mean()
            entropy_loss = entropy.mean()
            loss = pg_loss - ent_coef * entropy_loss + v_loss * p.vf_coef
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), p.max_grad_norm)
            optimizer.step()
            if st.frozen:
                apply_frozen(agent, st.frozen)

    st.next_obs, st.next_done = next_obs, next_done
    st.update += 1
    by_family: dict[str, list[float]] = defaultdict(list)
    for f in finished:
        by_family[f["family"]].append(f["score"])
    return {
        "update": st.update,
        "global_step": st.global_step,
        "lr": optimizer.param_groups[0]["lr"],
        "ent_coef": ent_coef,
        "pg_loss": pg_loss.item(),
        "v_loss": v_loss.item(),
        "entropy": entropy_loss.item(),
        "approx_kl": approx_kl.item(),
        "clipfrac": float(np.mean(clipfracs)),
        "train_score": {k: float(np.mean(v)) for k, v in sorted(by_family.items())},
        "episodes": len(finished),
    }


def run_dir(out: Path, cfg: TrainConfig, seed: int) -> Path:
    return out / f"{cfg.name}-{cfg.config_hash}-s{seed}"


def train(
    cfg: TrainConfig,
    seed: int,
    out: Path,
    *,
    resume: bool = True,
    stop_after_updates: int | None = None,
    eval_fn=None,
    log=print,
) -> Path:
    """Train (or resume) one seed. Returns the run directory.

    `eval_fn(actor_path, run_dir) -> dict` is called at each checkpoint and its
    result stored in metrics.jsonl (None skips evaluation, e.g. in tests).
    `stop_after_updates` stops early (tests of resume).
    """
    rd = run_dir(Path(out), cfg, seed)
    (rd / "checkpoints").mkdir(parents=True, exist_ok=True)
    (rd / "config.json").write_text(
        json.dumps({**cfg.to_json(), "seed": seed}, indent=2), encoding="utf-8", newline="\n"
    )
    full = rd / "full.pt"
    if resume and full.exists():
        st = load_full(full, cfg, seed)
        log(f"resumed {rd.name} at step {st.global_step:,}")
    else:
        st = new_state(cfg, seed)
        if cfg.init_from:
            src = init_path(cfg, seed)
            (rd / "init.json").write_text(
                json.dumps(
                    {"init_from": str(src), "init_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                     "frozen_units": st.frozen},
                    indent=2, sort_keys=True,
                ) + "\n",
                encoding="utf-8", newline="\n",
            )
    n_up = num_updates(cfg)
    t_last = time.perf_counter()
    while st.update < n_up:
        stats = train_update(st, cfg)
        now = time.perf_counter()
        st.wall_s += now - t_last
        t_last = now
        last = st.update == n_up
        if st.global_step >= st.next_checkpoint or last:
            while st.next_checkpoint <= st.global_step:
                st.next_checkpoint += cfg.checkpoint_every
            _checkpoint(rd, st, cfg, seed, stats, eval_fn, log)
            t_last = time.perf_counter()  # checkpoint evaluation is not training time
        if stop_after_updates is not None and st.update >= stop_after_updates:
            break
    return rd


def _checkpoint(rd: Path, st: TrainState, cfg: TrainConfig, seed: int, stats: dict, eval_fn, log) -> None:
    assert_no_shared_parameters(st.agent)
    name = f"step{st.global_step:010d}"
    actor_path = save_actor_checkpoint(rd / "checkpoints" / f"{name}.npz", st.agent)
    row = {
        "checkpoint": name,
        "global_step": st.global_step,
        "update": st.update,
        "wall_s": round(st.wall_s, 3),
        "steps_per_s": round(st.global_step / st.wall_s, 1) if st.wall_s else None,
        "train": stats,
    }
    if eval_fn is not None:
        row["eval"] = eval_fn(actor_path, rd)
    with open(rd / "metrics.jsonl", "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    save_full(rd / "full.pt", st, cfg, seed)
    short = ""
    if "eval" in row:
        short = " ".join(f"{k}:{v['success']:.2f}" for k, v in row["eval"].items())
    log(f"{rd.name} step {st.global_step:>10,} ({row['steps_per_s']} steps/s)  success {short}")

