"""PyTorch actor -> Delivery 2 actor file.

`nn.Linear.weight` is (outputs, inputs); our actor files store W as
(inputs, outputs) (FINAL_PLAN §7.1), so every weight is transposed here and
only here. The float32 values are widened to float64 exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..policy.actor import Params, save_actor


def assert_no_shared_parameters(agent: nn.Module) -> None:
    """The actor and critic must not share a single tensor (FINAL_PLAN §5)."""
    actor_ids = {id(p) for p in agent.actor.parameters()}
    critic_ids = {id(p) for p in agent.critic.parameters()}
    actor_storage = {p.untyped_storage().data_ptr() for p in agent.actor.parameters()}
    critic_storage = {p.untyped_storage().data_ptr() for p in agent.critic.parameters()}
    if actor_ids & critic_ids or actor_storage & critic_storage:
        raise AssertionError("actor and critic share parameters")


def actor_to_params(agent: nn.Module) -> Params:
    linears = [m for m in agent.actor if isinstance(m, nn.Linear)]
    if len(linears) != 3:
        raise ValueError(f"expected 3 Linear layers in the actor, found {len(linears)}")
    out: Params = {}
    for i, lin in enumerate(linears, start=1):
        w = lin.weight.detach().cpu().numpy().astype(np.float64)
        b = lin.bias.detach().cpu().numpy().astype(np.float64)
        out[f"L{i}.W"] = w.T.copy()  # (out, in) -> (in, out)
        out[f"L{i}.b"] = b
    return out


def save_actor_checkpoint(path: Path, agent: nn.Module) -> Path:
    assert_no_shared_parameters(agent)
    return save_actor(path, actor_to_params(agent))


@torch.no_grad()
def torch_logits(agent: nn.Module, obs: np.ndarray) -> np.ndarray:
    return agent.actor(torch.as_tensor(obs, dtype=torch.float32)).numpy()
