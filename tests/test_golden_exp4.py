"""Delivery 2, acceptance F (and A2): reproduce MIT CSAIL's Exp 4 exactly.

Needs the published files: `uv run python scripts/fetch_exp4.py` (CC BY 4.0,
see ATTRIBUTION.md). Without them these tests are skipped, but Delivery 2 is
only done when they have run green.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from weightshift.serialize import analyse_edit, dump_weights, edit_from_reply, quantize, read_dump, rounded

ROOT = Path(__file__).resolve().parents[1]
EXP4 = ROOT / "fixtures" / "exp4"


def _load_fetch_module():
    spec = importlib.util.spec_from_file_location("fetch_exp4", ROOT / "scripts" / "fetch_exp4.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fetch_exp4 = _load_fetch_module()
_missing = fetch_exp4.missing_or_bad(EXP4)
pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(
        bool(_missing), reason=f"Exp 4 files missing or changed ({_missing}); run `uv run python scripts/fetch_exp4.py`"
    ),
]

# Kinetix parameter names -> the names Exp 4's prompt uses (DELIVERY_2_ACCEPTANCE §8)
_P = "params/GeneralActorCriticRNN_0/"
KINETIX_NAMES = [("embed", "params/Dense_0")]
KINETIX_NAMES += [(f"actor{i}", f"{_P}DenseAndActivation_{i}/Dense_0") for i in range(5)]
KINETIX_NAMES += [("actor_head", f"{_P}Dense_0")]


@pytest.fixture(scope="module")
def ppo():
    with np.load(EXP4 / "ppo_final_seed0.npz", allow_pickle=False) as z:
        out = {}
        for name, prefix in KINETIX_NAMES:
            out[f"{name}.W"] = z[f"{prefix}/kernel"]
            out[f"{name}.b"] = z[f"{prefix}/bias"]
    return out


@pytest.fixture(scope="module")
def edited():
    with np.load(EXP4 / "edited_actor.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def prompt_weights() -> str:
    text = (EXP4 / "prompt.txt").read_text(encoding="utf-8")
    return text[text.index("embed.W (679 x 128), scale") :]


@pytest.fixture(scope="module")
def result():
    return json.loads((EXP4 / "oneshot_result.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def applied(ppo):
    r = rounded(ppo)
    parsed, after = edit_from_reply(r, (EXP4 / "answer.md").read_text(encoding="utf-8"))
    return r, parsed, after


def test_actor_structure(ppo, edited):
    assert sum(v.size for v in ppo.values()) == 171_664
    assert list(edited) == list(ppo)
    assert all(edited[k].shape == ppo[k].shape for k in ppo)


# ------------------------------------------------------------------ F1
def test_dump_matches_published_prompt_byte_for_byte(ppo, prompt_weights):
    assert dump_weights(ppo) == prompt_weights


def test_published_dump_reads_back_to_our_quantization(ppo, prompt_weights):
    ours = quantize(ppo)
    theirs = read_dump(prompt_weights)
    assert list(theirs) == list(ours)
    assert [t.k for t in theirs.values()] == [-2, -4, -3, -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3]
    assert all(np.array_equal(theirs[k].ints, ours[k].ints) for k in ours)


# ------------------------------------------------------------------ F2
def test_answer_parses_to_51_statements(applied, result):
    _, parsed, _ = applied
    assert parsed.n_blocks == 1 and not parsed.cut_off
    assert parsed.n_statements == 51 == result["edit_report"]["statements"]
    assert parsed.n_skipped == 0 and parsed.clean


# ------------------------------------------------------------------ F3
def test_edit_reproduces_published_network(applied, edited):
    _, _, after = applied
    for k in edited:
        assert after[k].dtype == edited[k].dtype == np.float64
        assert np.array_equal(after[k], edited[k]), k


def test_change_counts_match_published(applied, result):
    r, parsed, after = applied
    an = analyse_edit(r, after, parsed)
    assert an.n_changed == 6_073 == result["changes"]["changed"]
    assert an.n_params == 171_664 == result["changes"]["entries"]
    assert an.frac_changed == pytest.approx(result["changes"]["frac_changed"], abs=1e-15)
    for name, rec in result["changes"]["per_tensor"].items():
        assert an.changed_per_tensor[name] == rec["changed"], name
    assert sorted(an.tensors_changed) == sorted(result["edit_report"]["tensors_touched"])


def test_units_taken_over_match_the_post(applied):
    """FINAL_PLAN §18: embed 3, actor0 6, actor1 4, actor2-4 2 each, head rows 126-127."""
    r, parsed, after = applied
    an = analyse_edit(r, after, parsed)
    assert an.units_touched == {
        "embed": (6, 7, 8),
        "actor0": (122, 123, 124, 125, 126, 127),
        "actor1": (124, 125, 126, 127),
        "actor2": (126, 127),
        "actor3": (126, 127),
        "actor4": (126, 127),
    }
    assert an.outputs_touched == (0, 1, 2)
    changed_head_rows = np.flatnonzero((after["actor_head.W"] != r["actor_head.W"]).any(axis=1)).tolist()
    assert changed_head_rows == [126, 127]


# ------------------------------------------------------------------ A2
def test_all_files_verified():
    assert fetch_exp4.missing_or_bad(EXP4) == []
