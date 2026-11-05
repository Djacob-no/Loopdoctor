"""Gate A checks on a trained actor (DELIVERY_3_ACCEPTANCE §4 E, FINAL_PLAN §6.3-6.4).

All gates run on the **rounded** network (the one the model will edit), over
1,024 paired `select`-seed episodes per condition, sampled actions unless
noted. Success = at least one needed item delivered.

    E1 failure      T0 success <= 2%, sampled and greedy
    E2 pocket       >= 80% of T0 episodes pick a needed item; T0 loop index >= 0.80;
                    T0 full-but-not-delivering >= 0.25 and >= 3x the policy's own
                    open value (rule changed 2026-11-04: loop index is a fraction
                    <= 1, so "3x open" was unreachable once open looped at 0.6)
    E3 competence   open success >= 95% and mean score >= 40% of the oracle's;
                    probe success >= 90%; detour success >= 80%
    E4 capacity     zeroing the 4 lowest outgoing-L2 units per hidden layer
                    drops open mean score by < 2% relative
    E5 rounding     the float network also meets E1 (T0 sampled and greedy), and the 95% paired
                    bootstrap CI of (rounded - float) open score lies within +-3% of the
                    float open score (rule changed 2026-11-05 from "contains 0": with
                    1,024 paired episodes a 1% effect was enough to fail)

E6 (second seed) is decided across seeds by the CLI. Everything here is
deterministic, so a report can be recomputed bit for bit (E7).
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .env.seeds import seeds_for
from .evaluation import EpisodeResult, evaluate
from .metrics import EpisodeMetrics
from .policy.actor import actor_sha256, layer_names, load_actor, save_actor
from .serialize.analysis import rank_units_by_outgoing_l2, zero_units
from .serialize.quantize import rounded
from .stats import clopper_pearson, paired_bootstrap_ci

N_GATE = 1024
ROLE = "select"
EVAL_FAMILIES = ("open", "detour", "probe", "trap", "T0")

T0_MAX_SUCCESS = 0.02
POCKET_MIN_PICK_FRAC = 0.80
POCKET_MIN_RATIO = 3.0
POCKET_MIN_LOOP = 0.80
POCKET_MIN_FULL_NOT_DELIVERING = 0.25
OPEN_MIN_SUCCESS = 0.95
OPEN_MIN_SCORE_FRAC_OF_ORACLE = 0.40
PROBE_MIN_SUCCESS = 0.90
DETOUR_MIN_SUCCESS = 0.80
CAPACITY_UNITS = 4
CAPACITY_MAX_REL_DROP = 0.02
ROUNDING_MAX_REL_DIFF = 0.03


# ------------------------------------------------------------------ summaries
def needed_picks(m: EpisodeMetrics) -> int:
    return m.picks - m.wrong_picks


def summarize(results: Sequence[EpisodeResult]) -> dict:
    ms = [r.metrics for r in results]
    n = len(ms)
    k = sum(m.items_delivered > 0 for m in ms)
    ci = clopper_pearson(k, n)
    return {
        "episodes": n,
        "success": k / n,
        "success_ci95": [ci.low, ci.high],
        "mean_score": float(np.mean([m.score for m in ms])),
        "mean_delivered": float(np.mean([m.items_delivered for m in ms])),
        "mean_orders": float(np.mean([m.orders_completed for m in ms])),
        "frac_picked_needed": sum(needed_picks(m) > 0 for m in ms) / n,
        "loop_index": float(np.mean([m.loop_index for m in ms])),
        "idle_fraction": float(np.mean([m.idle_fraction for m in ms])),
        "full_not_delivering": float(np.mean([m.full_not_delivering for m in ms])),
    }


def run_family(actor_path: Path, family: str, seeds: list[int], mode: str = "sampled") -> list[EpisodeResult]:
    return evaluate(("policy", str(actor_path), mode), family, seeds, role=ROLE)


def quick_eval(actor_path: Path, n: int = 128) -> dict:
    """Checkpoint-time metrics: n select seeds per family, sampled, float network."""
    seeds = seeds_for(ROLE, n)
    out = {}
    for fam in EVAL_FAMILIES:
        s = summarize(run_family(Path(actor_path), fam, seeds))
        out[fam] = {k: s[k] for k in ("success", "mean_score", "loop_index", "idle_fraction")}
    return out


def ablate_capacity(params: dict, n_units: int = CAPACITY_UNITS) -> tuple[dict, dict[str, list[int]]]:
    """Zero the n lowest outgoing-L2 units of every hidden layer (ranked on `params`)."""
    out = params
    zeroed: dict[str, list[int]] = {}
    for layer in layer_names(params)[:-1]:
        units = rank_units_by_outgoing_l2(params, layer)[:n_units]
        zeroed[layer] = sorted(units)
        out = zero_units(out, layer, units)
    return out, zeroed


# ---------------------------------------------------------------------- gates
def e5_rounding_pass(
    float_t0_sampled: float, float_t0_greedy: float, rounded_minus_float_ci, float_open_mean: float
) -> bool:
    """E5: the float network meets E1 in both modes, and rounding moves the open score by
    less than ROUNDING_MAX_REL_DIFF (the whole 95% CI lies inside the margin)."""
    margin = ROUNDING_MAX_REL_DIFF * float_open_mean
    return (
        float_t0_sampled <= T0_MAX_SUCCESS
        and float_t0_greedy <= T0_MAX_SUCCESS
        and -margin <= rounded_minus_float_ci.low
        and rounded_minus_float_ci.high <= margin
    )


def compute_gates(actor_path: str | Path, n: int = N_GATE) -> dict:
    actor_path = Path(actor_path)
    seeds = seeds_for(ROLE, n)
    float_params = load_actor(actor_path)
    round_params = rounded(float_params)
    ablated, zeroed = ablate_capacity(round_params)

    with tempfile.TemporaryDirectory(prefix="ws-gates-") as tmp:
        p_round = save_actor(Path(tmp) / "rounded.npz", round_params)
        p_ablated = save_actor(Path(tmp) / "ablated.npz", ablated)
        res = {fam: run_family(p_round, fam, seeds) for fam in EVAL_FAMILIES}
        res_greedy = {fam: run_family(p_round, fam, seeds, "greedy") for fam in EVAL_FAMILIES}
        res_float = {fam: run_family(actor_path, fam, seeds) for fam in ("open", "T0")}
        res_float_greedy_t0 = run_family(actor_path, "T0", seeds, "greedy")
        res_ablated_open = run_family(p_ablated, "open", seeds)
    oracle_open = evaluate("oracle", "open", seeds, role=ROLE)

    s = {fam: summarize(r) for fam, r in res.items()}
    sg = {fam: summarize(r) for fam, r in res_greedy.items()}
    sf = {fam: summarize(r) for fam, r in res_float.items()}
    sf_greedy_t0 = summarize(res_float_greedy_t0)
    s_abl = summarize(res_ablated_open)
    oracle_mean = float(np.mean([r.metrics.score for r in oracle_open]))

    open_scores = [r.metrics.score for r in res["open"]]
    float_open_scores = [r.metrics.score for r in res_float["open"]]
    abl_scores = [r.metrics.score for r in res_ablated_open]
    base = float(np.mean(open_scores))
    rel_drop = (base - float(np.mean(abl_scores))) / base if base > 0 else float("inf")
    drop_ci = paired_bootstrap_ci(open_scores, abl_scores)
    round_ci = paired_bootstrap_ci(open_scores, float_open_scores)

    t0, op = s["T0"], s["open"]
    gates = {
        "E1_failure": {
            "pass": t0["success"] <= T0_MAX_SUCCESS and sg["T0"]["success"] <= T0_MAX_SUCCESS,
            "T0_success_sampled": t0["success"],
            "T0_success_greedy": sg["T0"]["success"],
            "max": T0_MAX_SUCCESS,
        },
        "E2_pocket": {
            "pass": (
                t0["frac_picked_needed"] >= POCKET_MIN_PICK_FRAC
                and t0["loop_index"] >= POCKET_MIN_LOOP
                and t0["full_not_delivering"] >= POCKET_MIN_FULL_NOT_DELIVERING
                and t0["full_not_delivering"] >= POCKET_MIN_RATIO * op["full_not_delivering"]
            ),
            "T0_frac_picked_needed": t0["frac_picked_needed"],
            "T0_loop_index": t0["loop_index"],
            "open_loop_index": op["loop_index"],
            "T0_full_not_delivering": t0["full_not_delivering"],
            "open_full_not_delivering": op["full_not_delivering"],
            "min_pick_frac": POCKET_MIN_PICK_FRAC,
            "min_ratio": POCKET_MIN_RATIO,
            "min_loop_index": POCKET_MIN_LOOP,
            "min_full_not_delivering": POCKET_MIN_FULL_NOT_DELIVERING,
        },
        "E3_competence": {
            "pass": (
                op["success"] >= OPEN_MIN_SUCCESS
                and op["mean_score"] >= OPEN_MIN_SCORE_FRAC_OF_ORACLE * oracle_mean
                and s["probe"]["success"] >= PROBE_MIN_SUCCESS
                and s["detour"]["success"] >= DETOUR_MIN_SUCCESS
            ),
            "open_success": op["success"],
            "open_mean_score": op["mean_score"],
            "oracle_open_mean_score": oracle_mean,
            "open_score_frac_of_oracle": op["mean_score"] / oracle_mean,
            "probe_success": s["probe"]["success"],
            "detour_success": s["detour"]["success"],
            "min": {
                "open_success": OPEN_MIN_SUCCESS,
                "open_score_frac_of_oracle": OPEN_MIN_SCORE_FRAC_OF_ORACLE,
                "probe_success": PROBE_MIN_SUCCESS,
                "detour_success": DETOUR_MIN_SUCCESS,
            },
        },
        "E4_capacity": {
            "pass": rel_drop < CAPACITY_MAX_REL_DROP,
            "zeroed_units": zeroed,
            "open_mean_score": base,
            "ablated_open_mean_score": float(np.mean(abl_scores)),
            "rel_drop": rel_drop,
            "drop_ci95": drop_ci.to_json(),
            "max_rel_drop": CAPACITY_MAX_REL_DROP,
        },
        "E5_rounding": {
            "pass": e5_rounding_pass(sf["T0"]["success"], sf_greedy_t0["success"], round_ci, sf["open"]["mean_score"]),
            "max_rel_diff": ROUNDING_MAX_REL_DIFF,
            "float_T0_success_sampled": sf["T0"]["success"],
            "float_T0_success_greedy": sf_greedy_t0["success"],
            "float_open_mean_score": sf["open"]["mean_score"],
            "rounded_minus_float_open_score_ci95": round_ci.to_json(),
        },
    }
    return {
        "actor": actor_path.name,
        "actor_sha256": actor_sha256(actor_path),
        "episodes_per_condition": n,
        "seed_role": ROLE,
        "seeds": [seeds[0], seeds[-1]],
        "pass": all(g["pass"] for g in gates.values()),
        "gates": gates,
        "rounded_sampled": s,
        "rounded_greedy": sg,
        "float_sampled": sf,
        "float_greedy_T0": sf_greedy_t0,
        "ablated_open": s_abl,
    }


# --------------------------------------------------------------------- output
def report_json(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def report_markdown(report: dict) -> str:
    g = report["gates"]
    s = report["rounded_sampled"]
    sg = report["rounded_greedy"]
    lines = [
        f"# Gate report: `{report['actor']}`",
        "",
        f"SHA-256 `{report['actor_sha256']}` · {report['episodes_per_condition']} paired `{report['seed_role']}`-seed "
        f"episodes per condition · rounded network · **{'PASS' if report['pass'] else 'FAIL'}**",
        "",
        "| Gate | Result | Key numbers |",
        "|---|---|---|",
        f"| E1 failure | {'pass' if g['E1_failure']['pass'] else 'FAIL'} | T0 success {_pct(g['E1_failure']['T0_success_sampled'])} sampled, "
        f"{_pct(g['E1_failure']['T0_success_greedy'])} greedy (max 2%) |",
        f"| E2 pocket | {'pass' if g['E2_pocket']['pass'] else 'FAIL'} | picked a needed item in {_pct(g['E2_pocket']['T0_frac_picked_needed'])} of T0 episodes; "
        f"loop index {g['E2_pocket']['T0_loop_index']:.3f} vs open {g['E2_pocket']['open_loop_index']:.3f}; "
        f"full-not-delivering {g['E2_pocket']['T0_full_not_delivering']:.3f} vs open {g['E2_pocket']['open_full_not_delivering']:.3f} |",
        f"| E3 competence | {'pass' if g['E3_competence']['pass'] else 'FAIL'} | open success {_pct(g['E3_competence']['open_success'])}, "
        f"score {g['E3_competence']['open_mean_score']:.1f} = {_pct(g['E3_competence']['open_score_frac_of_oracle'])} of oracle; "
        f"probe {_pct(g['E3_competence']['probe_success'])}; detour {_pct(g['E3_competence']['detour_success'])} |",
        f"| E4 capacity | {'pass' if g['E4_capacity']['pass'] else 'FAIL'} | open score {g['E4_capacity']['open_mean_score']:.2f} → "
        f"{g['E4_capacity']['ablated_open_mean_score']:.2f} with 4 units/layer zeroed ({_pct(g['E4_capacity']['rel_drop'])} drop, max 2%) |",
        f"| E5 rounding | {'pass' if g['E5_rounding']['pass'] else 'FAIL'} | float T0 success {_pct(g['E5_rounding']['float_T0_success_sampled'])} sampled, {_pct(g['E5_rounding']['float_T0_success_greedy'])} greedy (max 2%); "
        f"rounded − float open score CI [{g['E5_rounding']['rounded_minus_float_open_score_ci95']['low']:.2f}, "
        f"{g['E5_rounding']['rounded_minus_float_open_score_ci95']['high']:.2f}] "
        f"(must lie within ±{g['E5_rounding']['max_rel_diff'] * g['E5_rounding']['float_open_mean_score']:.2f}, 3% of {g['E5_rounding']['float_open_mean_score']:.2f}) |",
        "",
        "| Layouts | Success (sampled) | Success (greedy) | Mean score | Loop index | Idle fraction | Full, not delivering |",
        "|---|---|---|---|---|---|---|",
    ]
    for fam in EVAL_FAMILIES:
        lines.append(
            f"| {fam} | {_pct(s[fam]['success'])} | {_pct(sg[fam]['success'])} | {s[fam]['mean_score']:.2f} | "
            f"{s[fam]['loop_index']:.3f} | {s[fam]['idle_fraction']:.3f} | {s[fam]['full_not_delivering']:.3f} |"
        )
    return "\n".join(lines) + "\n"
