# Delivery 2: weights as text, the edit language, and the Exp 4 golden test
*Defines scope, locked rules, and acceptance criteria for the second implementation delivery. Parent plan: `FINAL_PLAN.md` (§7 serialization and edits, §6.4 capacity check, §10 edit analysis, §15 Days 1–2). Previous delivery: `DELIVERY_1_ACCEPTANCE.md`. Date: 2026-11-04.*

**One-line goal:** by the end of this delivery, a policy stored as a numpy file can be written out as Exp 4-style text, edited with the Exp 4 edit language, analysed, ablated and played in the game, and the whole pipeline **reproduces MIT's Exp 4 exactly**. Still no PyTorch and no LLM calls.

---

## 1. Scope

### In
1. **Actor file format** (`.npz`) with load, save and validation, plus a **numpy forward pass** (tanh MLP, softmax).
2. **`PolicyPlayer`**, sampled or greedy, which plugs into the Delivery 1 evaluation runner.
3. **Quantizer**: the per-tensor power-of-ten scale rule, rounding and dequantizing.
4. **Weight dump** in Exp 4's text format, and a **dump reader** that parses it back.
5. **Edit parser** for the Exp 4 edit language, with a skip log.
6. **Applier**, which applies the parsed statements in order to the rounded network.
7. **Edit analysis**: fraction changed, L2, units touched, statement and skip counts.
8. **Unit ablation helpers**: zero hidden units, and rank units by outgoing weight norm.
9. **`scripts/fetch_exp4.py`**, which downloads the published Exp 4 files and checks their hashes.
10. Tests: golden, round-trip, transpose, parser fuzz, unit tests.
11. `ATTRIBUTION.md` (MIT logs repo, CC BY 4.0).

### Out (later deliveries)
- PyTorch, PPO, trap shaping, and the *runs* of the competence and capacity checks (Delivery 3). Delivery 3 uses this delivery's helpers.
- Prompt assembly beyond the weight section, `prereg.md`, LLM adapters, any paid call (Delivery 4).
- Matched random edits, control (b′) (Delivery 4 or 5).

---

## 2. Locked rules (tests depend on these)
Where the plan left a detail open, it is decided here. Change a rule here first, then in the code. **Where Exp 4's published files disagree with a rule below, Exp 4 wins**: update the rule here and record the finding in §8.

**Actor file**
- A `.npz` file with exactly these keys:

| Key | Shape | Meaning |
|---|---|---|
| `L1.W` | (600, 32) | input → hidden 1 |
| `L1.b` | (32,) | |
| `L2.W` | (32, 32) | hidden 1 → hidden 2 |
| `L2.b` | (32,) | |
| `L3.W` | (32, 7) | hidden 2 → 7 action logits |
| `L3.b` | (7,) | |

- **`W` is (inputs, outputs): row i = input i.** This is the Exp 4 convention. PyTorch stores the transpose; converting is Delivery 3's job.
- The loader rejects missing or extra keys, wrong shapes, and non-finite values. The total is **20,519** parameters, and that is asserted.
- The serializer, applier and analysis code are generic over a list of named tensors, so Exp 4's tensors (`embed`, `actor0..4`, `actor_head`) go through the same code. Exp 4's network is never *run*; only its weights are dumped and edited.

**Forward pass and actions**
- `h1 = tanh(x @ L1.W + L1.b)`, `h2 = tanh(h1 @ L2.W + L2.b)`, `logits = h2 @ L3.W + L3.b`, softmax over 7 actions, computed in float64.
- **Sampled:** one uniform draw per step from the episode's action-sampling RNG (`player_rng(episode_seed)` from Delivery 1), mapped through the cumulative probabilities.
- **Greedy:** argmax. Ties go to the lowest action index.

**Quantization (FINAL_PLAN §7.1)**
- Per tensor: `k = floor(log10(max|w|)) − 2`, `scale = 10^k`, displayed value `round(w / scale)` as an integer. The largest entry gets 3 significant digits.
- An all-zero tensor uses `k = 0`.
- Dequantize as `int × 10.0**k` in float64.
- The **rounded** network is what the model edits and what baseline (a) evaluates.
- Rounding is `np.rint` (half to even) on `w / 10.0**k` in float64. If the largest entry rounds up to 1000, that entry shows 4 digits and the scale is not bumped. Settled by the golden test (§8).

**Weight dump text**
- One header per tensor, then one line per input row (`i: ints…`); a bias is a single line. The header and row layout are copied **byte for byte** from Exp 4's `prompt.txt`.
- Our tensor names are `L1.W L1.b L2.W L2.b L3.W L3.b`.
- A dump reader parses the text back into integers and scales.

**Edit language (FINAL_PLAN §7.2)**
- Only the **last** code block opened with three backticks followed by `edits` is read.
- One statement per line. `#` starts a comment. A bracketed list may continue over several lines.
- Forms:

| Form | Example |
|---|---|
| single entry | `L2.W[3, 17] = 1.5` |
| column slice | `L1.W[120:126, 4] = -2` |
| whole column | `L1.W[:, 4] = 0` |
| row slice | `L2.W[5:8, :] *= 0.5` |
| 1-D entry or slice | `L1.b[4] = 3`, `L3.b[0:3] = [0.1, 0.2, 0.3]` |
| list right-hand side | one value per selected entry, row-major |
| zero a tensor | `ZERO L3.b` |

- Operators: `=`, `+=`, `-=`, `*=`. Slices are half-open (`a:b` means a up to b−1) and have no step. `:` alone means the whole axis.
- **Values are plain decimals in real weight units**, not display integers. They must be finite with |v| ≤ 1e6.
- Negative indices are rejected (`index_out_of_range`). Exp 4's prompt does not offer them.

**Skipped lines**
- An unreadable statement is **skipped and logged** with its line number, raw text and a reason. The rest still apply.
- Reasons: `syntax`, `unknown_tensor`, `wrong_rank`, `index_out_of_range`, `list_length`, `value_out_of_range`.
- A reply with no `edits` block gives 0 statements and the flag `no_block`.
- The parser **never** calls `eval` or `exec` and **never** raises, whatever the input string.

**Applier**
- Statements apply **in order** to a copy of the rounded network, so a later statement overwrites an earlier one. The input network is never modified.
- The result stays in float and is **not** re-quantized.

**Edit analysis (FINAL_PLAN §10)**
All of it is measured against the rounded network:
- `n_changed`: entries where the value differs (exact `!=`)
- `frac_changed`: `n_changed` divided by all actor parameters
- `l2_change`: L2 norm of the difference
- units touched per hidden layer ("taken over"): unit j of layer k counts if `Lk.b[j]` changed, or `Lk.W[:, j]` changed on a row that is not itself a touched unit of layer k−1. Layers are processed in order; for L1, every row is an input. (§8 explains why.)
- output (action) units touched, by the same rule
- `n_statements` and `n_skipped`

**Ablation (FINAL_PLAN §6.4, mechanism ablation)**
- `zero_units(actor, layer, units)` zeros each unit's incoming column, bias and outgoing row.
- `rank_units_by_outgoing_l2(actor, layer)` ranks units by the L2 norm of their outgoing weights, lowest first. Ties go to the lower index.

**Golden data**
- `scripts/fetch_exp4.py` downloads from a **pinned commit SHA** of `github.com/RikiyaT/lm_understand-logs` into `fixtures/exp4/`, which is gitignored:

| File (under `exp4_fix_car/`) | Used for |
|---|---|
| `networks/ppo_final_seed0.npz` | the network before the edit |
| `networks/edited_actor.npz` | the network after the edit |
| `runs/oneshot_repair/reply/prompt.txt` | the published weight dump |
| `runs/oneshot_repair/reply/answer.md` | the 51 edit statements |
| `eval/oneshot_result.json` | published change counts |

- The script checks a SHA-256 per file, with the hashes recorded in the script. A second run does nothing.
- The golden tests are marked `golden`. They are skipped with the message "run `uv run python scripts/fetch_exp4.py`" when the files are absent.
- **Known so far** (Exp 4 README, read 2026-11-04):
  - `ppo_final_seed0.npz` uses Kinetix names: `params/Dense_0` is embed, `DenseAndActivation_0..4` are actor0..4, and `GeneralActorCriticRNN_0/Dense_0` is actor_head. It also holds value-head keys, which are ignored.
  - `edited_actor.npz` uses `embed.W`, `embed.b`, `actor0..4` and `actor_head`.
  - The network is 679 → 128 (ReLU) → 5 × 128 (tanh) → 16, with 171,664 actor weights.
  - The exact sub-keys and dtypes are inspected first and recorded in §8. This closes FINAL_PLAN's "[UNVERIFIED] npz key names".

---

## 3. Deliverables (files)
```
src/weightshift/policy/actor.py        actor format, load/save/validate, forward, probabilities
src/weightshift/policy/player.py       PolicyPlayer (sampled | greedy)
src/weightshift/serialize/quantize.py  scale rule, rounding, dequantizing
src/weightshift/serialize/dump.py      weight dump text + reader
src/weightshift/serialize/edits.py     edit parser and skip log
src/weightshift/serialize/apply.py     applier
src/weightshift/serialize/analysis.py  edit analysis, unit ranking, zero_units
scripts/fetch_exp4.py
ATTRIBUTION.md
tests/test_policy.py, tests/test_serialize_*.py, tests/test_golden_exp4.py
```
Changes to Delivery 1 code:
- `evaluation.py`: `evaluate` and `run_one` also accept a picklable policy spec, `("policy", actor_path, "sampled" | "greedy")`. The existing scripted-player string API and its results do not change.
- `.gitignore` gains `fixtures/exp4/`. `pyproject.toml` registers the `golden` marker. The README gets this delivery's commands and numbers.
- **No new runtime dependencies.** The parser is hand-written, without `lark`.

---

## 4. Acceptance criteria
Each item is checkable. "Pass" means the stated command or inspection gives the stated result.

### A. Setup
- **A1.** `uv sync` then `uv run pytest` succeeds on Python 3.12, and `torch` is still not installed. The whole Delivery 1 suite is unchanged and green.
- **A2.** `uv run python scripts/fetch_exp4.py` then `uv run pytest -m golden` succeeds. Running the fetch again downloads nothing. A file altered by one byte fails the hash check.

### B. Actor and policy player
- **B1.** Save then load is bit-exact. Wrong key names, wrong shapes and NaN values are each rejected. The parameter count is 20,519.
- **B2.** The forward pass matches a hand-computed toy network, and the probabilities sum to 1.
- **B3. Row = input.** With every weight zero except `L1.W[i, j]`, only input i changes hidden unit j. Dumping an `(outputs, inputs)` array for a non-square layer is rejected on shape.
- **B4.** For 200 seeds, sampled episodes with the same actor and seeds are byte-identical when run twice. Greedy episodes are deterministic.
- **B5.** `evaluate` with a policy spec gives identical results serially and in parallel (`WEIGHTSHIFT_WORKERS`).
- **B6.** Policy-driven episodes run at **at least 3,000 environment steps per second** on one core. The figure is recorded in the README.

### C. Quantization and dump
- **C1.** Scale rule unit tests: a max of 0.0123 gives `1e-4`, 1.23 gives `1e-2`, 80 gives `1e-1`. A tensor whose largest magnitude is negative uses that magnitude. An all-zero tensor gives `1e0`.
- **C2.** Every dumped integer has at most 3 digits, except at the rounding edge recorded in §8.
- **C3.** For 100 random actors, dump → reader → dequantize equals quantize(actor) exactly.

### D. Edit parser
- **D1.** One test per form and operator in §2, including a multi-line list, comments, last-block-wins and a reply with no block.
- **D2.** One test per skip reason. In each, the other statements in the block still apply.
- **D3. Fuzz.** 10,000 seeded random and mutated strings never raise and never execute anything. A sentinel such as `__import__('os')` inside a statement is skipped as `syntax`.

### E. Applier
- **E1.** Statements apply in order: a later statement overwrites an earlier one, and `+=` after `=` adds to the new value.
- **E2.** Slice and list orientation are tested on non-square tensors, so a swapped row/column fails.
- **E3.** The input actor is unmodified after applying.

### F. Golden test (Exp 4)
- **F1.** Our dump of the rounded `ppo_final_seed0` actor equals the weight lines of `prompt.txt` **byte for byte**, after mapping tensor names.
- **F2.** Parsing `answer.md` gives **51 statements and 0 skipped**, from the last `edits` block.
- **F3.** Applying them to the rounded network equals `edited_actor.npz` **exactly** (both float64), with **6,073 changed weights** of 171,664 (3.54%), matching `oneshot_result.json` overall and per tensor.

### G. Round trip on our own network
- **G1.** For random actors, float → quantize → text → reader → actor gives action probabilities equal to the quantized actor's (within 1e-12) on 1,000 random observations, and identical sampled `T0` episodes for 100 seeds.
- **G2.** An empty `edits` block leaves the rounded network unchanged: 0 changed.

### H. Edit analysis and ablation
- **H1.** Hand-built edits give known `n_changed`, `frac_changed`, `l2_change`, units touched per layer and outputs touched. On Exp 4, the units touched are embed 6–8, actor0 122–127, actor1 124–127, actor2–4 126–127 and outputs 0–2.
- **H2.** After `zero_units`, the network output does not change whatever values the zeroed unit's incoming weights take. The ranking breaks ties by lower index.

### I. Manual acceptance (you)
- **I1.** You read a dump of a random actor next to an excerpt of Exp 4's `prompt.txt` and confirm they look alike.
- **I2.** You read the locked rules (§2) and the golden-test findings (§8), and confirm them or send changes.

---

## 5. Definition of done
All of A–H pass in one `uv run pytest` run **with the golden tests run, not skipped**. Line coverage of `src/weightshift/serialize/` and `src/weightshift/policy/` is at least **90%**. The README is updated, and you have signed off I1–I2. Delivery 3 (PPO, trap shaping, competence and capacity checks) starts only after that.

## 6. Dependencies on later deliveries
- **Delivery 3** (PPO, PyTorch) exports the trained actor in this delivery's `.npz` format, transposing `nn.Linear.weight`. It adds a PyTorch transpose test against B3, and runs the competence and capacity checks with `evaluate(policy spec)`, `zero_units` and `rank_units_by_outgoing_l2`.
- **Delivery 4** (prompt, `prereg.md`, LLM calls) uses `dump.py` for the weight section of the prompt, and `edits.py`, `apply.py` and `analysis.py` on the **stored** replies. A parser bug is fixed and the stored reply re-read; no call is repeated.

## 7. Risks and open items
- **Exp 4 rounding or format quirks** may differ from §2. The golden test is the authority: update §2 first, then the code.
- **npz key or dtype surprises.** Inspect the files before writing the golden test, and record what you find below.
- **`answer.md` may use a form missing from §2.** Add it, because it is Exp 4's grammar, and note it below.
- **The upstream repo may change.** The commit SHA and file hashes are pinned.

## 8. Implementation notes and clarifications (added during implementation)
Recorded here so reviewers can see where the code had to interpret this document.

1. **What the Exp 4 files contain (closes FINAL_PLAN's "[UNVERIFIED] npz key names").**
   - `ppo_final_seed0.npz` is float32 with Kinetix names: `params/Dense_0/{kernel,bias}` is embed (679×128), `params/GeneralActorCriticRNN_0/DenseAndActivation_{0..4}/Dense_0/{kernel,bias}` are actor0..4, and `params/GeneralActorCriticRNN_0/Dense_0/{kernel,bias}` is actor_head (128×16). `DenseAndActivation_5..9` and `Dense_1` are the value head, which is ignored.
   - Kinetix (flax) kernels are already (inputs, outputs), so they need no transpose.
   - `edited_actor.npz` is **float64** with keys `embed.W`, `embed.b`, `actor0.W` … `actor_head.b`.
   - Pinned commit: `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`.
2. **Quantization, settled by the golden test.**
   - Quantizing in float64 with `np.rint` reproduces every integer in `prompt.txt`.
   - On these weights, half-to-even and half-away-from-zero agree. Quantizing in float32 instead would get 1–2 integers wrong in actor3.W and actor4.W, so float64 is the rule.
   - Dequantizing as `int × 10.0**k` reproduces every unchanged entry of `edited_actor.npz` bit for bit. `int / 10**(-k)` does **not**: it differs in the last bit on many entries, up to a quarter of some tensors.
   - Our actors are therefore float64 throughout. A float32 file is widened when loaded.
3. **Dump format.** The header lines, the `i: ` row prefix, the single blank line between tensors and the final newline all match `prompt.txt` byte for byte (567,049 characters). The prompt mentions `embed.W (679 x 128)` once earlier, in its network description, so the golden test finds the dump by its full header, with `, scale`.
4. **"Units touched" needed a sharper definition than the one first written here.** The first definition counted a unit if its incoming column, its bias **or its outgoing row** changed.
   - On Exp 4 that counted **every** unit of every layer. Taking over a unit starts with zeroing its outgoing row (`actor0.W[6:9, :] = 0`), which crosses every column of the next matrix. Zeroing a column crosses every row in the same way.
   - The rule in §2 excludes changes on rows that belong to units already taken over in the layer before. On Exp 4 it gives exactly the structure described in the post (FINAL_PLAN §18): embed 3, actor0 6, actor1 4, actor2–4 2 each. Astra's head-row edits (rows 126–127) are rows of taken-over units, so they don't count as new output units. Only the bias edits do, which gives outputs 0–2.
5. **Extra skip reason `wrong_rank`**: the number of indices doesn't match the tensor (e.g. `L1.W[3] = 0`). It is separate from `index_out_of_range` so the logs say what went wrong.
6. **Leniencies beyond Exp 4's wording**, all harmless and accepted without a skip:
   - numbers may use an exponent (`1e-3`), a leading `+`, or a bare decimal point (`.5`, `3.`);
   - a list may end with a trailing comma;
   - text after a closing fence is ignored, and other code blocks (e.g. ```` ```python ````) are ignored.

   `nan`, `inf`, fractions (`1/3`) and anything else are `syntax` skips.
7. **Multi-line lists.** A list that is still open continues onto the next line, unless that line clearly starts a new statement (`NAME[` or `ZERO `). In that case the unfinished one is skipped as `syntax`, so one missing `]` cannot swallow the rest of the block.
8. **Fences.** Only a bare ```` ``` ```` line closes a block.
   - A new ```` ```edits ```` opener inside an open block ends that block (flagged `cut_off`) and starts the next one, which is then the last block. *(Fixed after review: at first any line starting with three backticks closed a block, so a second opener was swallowed and the earlier block won.)*
   - Another fence such as ```` ```python ```` inside a block is not a closer; it is skipped as `syntax`.
   - A block with no closing fence (a reply cut off mid-block) is still read and flagged `cut_off`. Exp 4's harness reports the same flag.
9. **Policy specs.** `evaluate` and `make_player` accept `("policy", actor_path, mode)`.
   - The loaded actor is cached by the **SHA-256 of the file's bytes**, and the file is re-read on every call (about 0.08 ms), so a rewritten file is always picked up.
   - *(Fixed after review: the cache was keyed by path only, so save → evaluate → overwrite → evaluate in one process scored the old network. Path + modification time + size is not enough either: two writes of a same-size `.npz` within one filesystem timestamp tick, measured here on both ext4 and `/mnt/c`, leave both unchanged.)*
   - `actor_sha256(path)` gives the same hash for the experiment log's `checkpoint_sha256`.
10. **The speed benchmark** is `scripts/bench_steps.py --policy`: 29,000 steps/s on `open` and 33,000 on `T0`, against a target of 3,000.
11. **Hardware crashes.** While this delivery was built, 7 of 9 full test runs crashed with segfaults (once inside pytest's own error reporting), as in Delivery 1's known issue. Every traceback was in Delivery 1 tests, including pure-Python code. Clean runs pass all 297 tests.
