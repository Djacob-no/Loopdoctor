# WeightShift (Loop Doctor): Final Build Plan
*Version: final (v2), 2026-11-04 · Supersedes `ORIGINAL_PLAN.md`, `weightshift_plan_review_recommendations.md` (Review #2), `REVIEW_3_recommendations.md`, `REVIEW_4_recommendations.md` · Audience: the builder agent in `C:\Projects\QuickWins\LoopDoctor`*

> This is the single source of truth. It merges v1 and the three reviews, settles every conflict between them, and re-checks the disputed facts against primary sources (§18). Where an earlier document disagrees with this one, this one wins. The review documents stay in the repo as the audit trail. Items marked **[UNVERIFIED]** must be confirmed before you rely on them. Code design is up to the builder. The things this plan fixes are the **protocol, the formats, the gates, and the controls**.

---

## 1. Goal and hypothesis
Reproduce, at hobby scale, **Experiment 4** of MIT CSAIL's *Sparks of Weight-Space Programming* (Takehi, Teoh, Zhang, Khattab, 2026-11-02) [S1].
- In that run, `gpt-6-astra` received a PPO policy that never succeeded (0/256), as text. It also got the scene rules and one recorded episode.
- In **one tool-free reply** it explained the failure and wrote 51 edit statements.
- Those statements changed 6,073 of 171,664 weights (3.54%) and produced **1,024/1,024 successes** [S2][S4][S5].

**Hypothesis.** A frontier LLM can read a small tanh PPO policy (~20.5k weights) for a single-robot warehouse grid game, diagnose a measured local-minimum failure, and repair it by writing weight edits. It does this with no gradients, no tools and no rollouts. The repair must significantly beat the rounded unedited policy and matched random edits on paired held-out episodes, without regressing elsewhere.

**The scientific question the controls must answer:** did the model *read* the weights, or did it just *write a patch* from the rules? Exp 5 shows frontier models can write working grid policies from scratch [S1][S6]. That makes the no-weights control (§9, e1) as important as the main result.

**Negative results are a result.** Every attempt is logged and reported, including parser skips, refusals and failed controls.

---

## 2. Locked decisions (read this if nothing else)
| # | Decision |
|---|---|
| D1 | **Primary condition = Exp 4's prompt contract** on our game: one tool-free reply, no feedback, no second try. |
| D2 | **Weights shown** as integers × a per-tensor power-of-ten scale (3 significant digits of the tensor's max), **rows = inputs** (`W` is `(inputs, outputs)`). |
| D3 | **Edits** use the Exp 4 edit language with **real decimal values**: `=`, `+=`, `-=`, `*=`, slices, lists, `ZERO`. Only the last block counts, unreadable lines are skipped and logged, and edits apply to the **rounded** weights. No JSON schema, no `eval`. |
| D4 | **Observation:** 6 binary channels × 10×10 = 600 inputs, **interleaved** index `i = (row*10 + col)*6 + ch`. No derived features in the primary condition. |
| D5 | **Network:** 600 → 32 → 32 → 7, all tanh, **20,519 parameters**. Widen to 64 only if the capacity check (§6.4) fails. |
| D6 | **Failure:** a "dock pocket" local minimum, induced with disclosed distance-to-dock shaping (the Kinetix analogue). The **competence check** and **capacity check** must pass before any paid call. |
| D7 | **Day-4 gate models:** `gpt-6-astra` (the replication) and `claude-opus-5-5` (strongest Claude), k = 5 each, with `claude-haiku-5-5` as the cheap control. Effort `high`. Sonnet 5.5 and GPT-6.1 Sol come in on Day 5. |
| D8 | **Headline metric:** first-attempt success rate over k independent attempts. Per attempt: success rate on 1,024 paired episodes, sampled actions, on the layout shown in the prompt. |
| D9 | **Controls:** (a) rounded-unedited vs float, (b′) matched random edits, (c) hand-written bypass, (d) more PPO, **(e1) no weights, no hints**, (e2) no weights with named free units, (f) transplant to another seed, and the mechanism ablation. |
| D10 | **No hard 10% weight cap.** Report the fraction changed, the L2 and the units touched. The optional unit cap is pre-registered. |
| D11 | **`prereg.md`**, including the prompt template, is hashed before the first paid call. Any prompt change after that is a **new condition**. |
| D12 | **Renderer:** ovrtx + ovstage after a Day-7 spike. Fallbacks: current-stable Isaac Sim, then three.js, then Blender. Driver target **595.97**. Rotate-then-translate keys. |

---

## 3. Name and framing
- **Name: WeightShift.** It reads as a warehouse *shift* and as a *shift* in weights. The working title is Loop Doctor.
- Credit line: "inspired by Grocery Bot (grocerybot.no)". The Grocery Bot name stays out of the title.
- Credit the MIT CSAIL post and link its logs repo.

---

## 4. Game spec: WeightShift v0
It is loosely inspired by Grocery Bot. **None of its code, assets, maps, protocol or branding is used** (§16).
- **Grid:** 10×10. The outer wall is obstacles, plus 2–3 rack rows. Racks hold one of **4 SKUs** and never run out. There is **one dock**.
- **Robot:** one robot that carries up to 3 items. It picks from a **4-adjacent** rack holding a needed SKU. Wrong picks are allowed and occupy a slot until delivered or the episode ends. That keeps Grocery Bot's "wrong pick takes a slot" pressure.
- **Orders:** an order of 3–4 items drawn from the 4 SKUs. When it completes, the next order appears.
- **Actions (7):** no-op, up, down, left, right, pick, drop. Drop works only on or adjacent to the dock (fix one rule and document it).
- **Reward / score:**
  - **+1 per delivered needed item**, **+5 per completed order**. That is Grocery Bot's shape, which awards +1 per item loaded and +5 per completed order [S13].
  - Plus the disclosed shaping term from §6.2 *during training only*.
  - The **score reported in evaluation is unshaped**.
- **Episode:** 200 steps.
- **Layouts:** generated from a seed. Three disjoint families:
  - `open`: normal layouts, where the policy should be competent
  - `trap`: dock-pocket layouts
  - `probe`: mirrored, rotated or opened-pocket variants, used for the competence check

  Fix a **headline trap layout**, `T0`, that the prompt describes. Exp 4 had one deterministic scene. Spawn positions and order sequences vary by seed.
- **Determinism:** identical seeds must give identical episodes under greedy actions, and identical sampled episodes under a fixed action-sampling RNG. Unit-test this.
- **Later scale-up (only after a GO):** 16×12 with 3 robots and a shared policy, adding collisions and deadlocks.

---

## 5. Observation and network
**Primary observation "G":** 6 binary channels × 10×10 = **600 inputs**, using the blog's **interleaved** grid convention, as in Exp 1 and Exp 5 [S6][S7]:
```
i = (row * 10 + col) * 6 + channel        # row 0 = top, col 0 = left
channels: 0 robot · 1 obstacles/racks · 2 racks holding a still-needed SKU · 3 dock · 4 trail (previous robot cell) · 5 gauge
```
- **Gauge channel (5):** a fixed block of cells encodes the carried count (0–3) and the needed count per SKU, as filled bars, in the spirit of Exp 5's "gauge" channel. Describe the encoding in one short paragraph of the prompt.
- **No derived features** in the primary condition. That rules out Review #2's `dx_to_dock`, stall timer, `is_full` and `can_pick`, which already contain the diagnosis.
- **Secondary labelled condition "G+E"** (Day 5–6, only after a GO): G plus 6 **raw** scalars at indices 600–605: robot x, robot y, dock x, dock y, items carried, time-left fraction. No relative vectors, no timers, no flags. It is reported as its own row.

**Network:** `h1 = tanh(x @ L1.W + L1.b)`, `h2 = tanh(h1 @ L2.W + L2.b)`, `logits = h2 @ L3.W + L3.b`, with a 7-way softmax and **sampled** actions.
- Shapes: 600→32→32→7, giving **20,519 parameters** (600·32+32 + 32·32+32 + 32·7+7). Exp 5's 600→32→32→6 has 20,486 [S6].
- With G+E (606 inputs): 20,711.
- All tanh, matching the Exp 1 and Exp 5 grid policies. The Kinetix first layer is ReLU, which makes hard off-switches easier. We deliberately don't copy that, and say so in the prereg.

**Actor/critic:** separate networks. Current SB3 is already separate by default, so Review #2's "shared backbone" claim is outdated [S12]. Use a **single-file CleanRL-style PPO** so the six actor tensors are explicit. Assert at save time that no actor parameter is shared with the critic. **Serialize the actor only.**

---

## 6. Training, failure induction and pre-call checks
### 6.1 PPO
- CleanRL-style `ppo.py` (discrete), Gymnasium vector env with 32–64 envs on **CPU**. A 20k-parameter MLP doesn't need the GPU. A batched torch env is optional; measure it before relying on it.
- Budget 5–50M steps. Train **3–5 seeds**, because control (f) needs a second seed with the same failure.
- Checkpoint every 250k steps. Log open, trap and probe metrics at each checkpoint.

### 6.2 Inducing the dock-pocket failure (Kinetix analogue)
- In Kinetix, the reward includes `0.2 × (d_prev − d)` toward the goal. That shaping pays the car for driving straight at the ramp, which is the local minimum [S3].
- **Our analogue:**
  - On `trap` layouts, the straight line from the pick area to the dock hits the closed side of a U of racks. The dock is reachable only the long way round, which first moves away from the dock.
  - Add a disclosed shaping bonus `β × (d_prev − d)` for reducing the **Manhattan (straight-line) distance** to the dock while carrying items. Do *not* use the shortest-path (BFS) distance: it would reward the detour and remove the trap.
- **Target:** near-zero deliveries on `T0` and on trap layouts, with normal performance on `open` and `probe`.
- Prefer shaping into the pocket on layouts the policy otherwise solves, rather than "train on 95% open layouts and hope". The latter tends to produce a policy that never learned detours, and that fails §6.3.
- **Document the exact procedure that produced the checkpoint.** Engineering the failure is allowed if it is disclosed.
- The second failure, symmetric-aisle oscillation, waits until the first one has a result.

### 6.3 Competence check (hard gate)
The blog's limit is exact: "Editing weights could remove an obstacle that hid an ability, but it could not add an ability the model lacked" [S1]. Before any paid call, the chosen checkpoint must:
- deliver normally on `open` layouts
- deliver on `probe` layouts: the pocket mirrored, rotated, or with its opening facing the robot
- route around walls of the same shape elsewhere

If it can't detour anywhere, the ability is missing. **Change the checkpoint or the trap. Do not call a model.** Log the competence metrics on the failure card; they are legitimate diagnostic context.

### 6.4 Capacity check (hard gate; replaces Review #4's "bottom quartile" rule, which is ill-formed)
A bypass needs units it can take over. Astra took 2–6 units per layer in a 128-wide net [S4]. For each hidden layer of our 32-wide net:
1. Rank units by the L2 norm of their outgoing weights.
2. Zero the 4 lowest-ranked units in each layer, including their outgoing weights.
3. Check that `open`-layout mean score drops by **< 2% relative** over 1,024 episodes.

If it fails, **widen to 64** and retrain. Note that a 64-wide net is ~43k parameters and pushes the prompt past Haiku's 100k-token price tier.

**This list is never shown to the model** in the primary condition. Review #2 was right to want the report and right to hide it.

### 6.5 The failure card (what the model sees as diagnosis)
Exp 4 showed an aggregate over 256 episodes and one episode with positions every 16 steps [S3][S4]. Ours contains:
- **Aggregate table** over 256 diagnosis-seed episodes on `T0`:
  - deliveries, orders, score
  - loop index (revisits of the same cell and carried state within k steps)
  - idle/bump fraction
  - full-but-not-delivering fraction
- **The same aggregate on `open`**, which shows competence.
- **One full 200-step ASCII trace** on `T0`. It's small, so include all of it.
- **Excluded from the primary condition:** action-probability snapshots, heatmaps, probes and quiet-unit lists. Those are extra ablations.

---

## 7. Serialization and the edit language
### 7.1 Weight dump (copy Exp 4's format) [S2][S3]
- One header per tensor, then one line per **input row**. The examples below show the header style; values are illustrative:
  ```
  L1.W (600 x 32), scale 1e-2: one line per row (row i)
  0: 1 -3 -4 -3 -6 3 1 -6 -2 7 ...
  L1.b (32), scale 1e-4: ...
  ```
- **Scale rule:** `scale = 10^(floor(log10(max|w|)) − 2)`, so the largest entry has 3 significant digits. Values are `round(w / scale)` as integers. Choose the scale per tensor. Exp 4 used `1e-2` for embed.W, `1e-4` for biases and `1e-3` for the other weights, so never use one global scale [S3].
- **Transpose:** `nn.Linear.weight` is `(out, in)`. Dump `weight.T`. Dumping `layer.weight` directly silently transposes the policy. Test for it.
- The **rounded** network, dequantized as `int × scale`, is the network the model edits and the network baseline (a) evaluates.

### 7.2 Edit language (Exp 4 verbatim in substance) [S3][S4]
- A code block whose opening line is three backticks followed by `edits`. One statement per line, `#` comments, and long lists may continue over lines.
- **Only the last `edits` block is read.** Statements apply **in order** to the network as shown, and a later statement overwrites an earlier one.
- **Values are plain decimal numbers in real weight units** (|v| ≤ 1e6). They are *not* integers in the display scale. Astra's `actor0.W[6,122] = 80` means a real weight of 80, and real weights that large are what produce hard switches through saturated tanh units.
- Forms to support: `T[i, j] = v`, `T[a:b, j] = v`, `T[:, j] = v`, `T[a:b, :] = v`, `T[a:b] = [v, …]` (one value per entry), `T[...] += v`, `-=`, `*=`, `ZERO T`. Slices are half-open, and `:` alone means the whole axis.
- **Tensor names** in our prompt: `L1.W L1.b L2.W L2.b L3.W L3.b`.

### 7.3 Parser and applier
- A strict hand-written grammar (regex or `lark`). **Never `eval`.**
- **Store the raw reply before parsing.**
- **Unreadable statements are skipped and logged**, and the rest still apply, as in Exp 4.
  - Pre-registered rule: an attempt with any skipped line is still evaluated and reported, but is **not counted as a clean success** in the headline.
  - The attempt is never deleted.
- **A parser bug is not a model failure.** Fix the parser and re-read the *stored* reply, as Exp 5 did [S6]. Never spend another call.
- After the edits, the network stays in float. It isn't re-quantized.
- **Golden test (Day 1).** Exp 4's networks are public: `networks/ppo_final_seed0.npz` and `edited_actor.npz` [S2].
  1. Dump `ppo_final_seed0` with our serializer and check that it matches the published `prompt.txt` weight lines.
  2. Apply the 51 statements from `answer.md` with our applier and check that the result equals `edited_actor.npz`, with **6,073** changed weights.

  This one test validates the scale rule, the transpose convention, the grammar and the apply order against ground truth. **[UNVERIFIED]** the npz key names. Inspect them first.
- Round-trip test on our own net: float → quantize → text → parse → network. Action distributions must equal the quantized network's, and seeded episodes must be identical.

---

## 8. The prompt and API protocol
### 8.1 Prompt contract (order follows Exp 4) [S3]
1. **Framing:** what the network controls, how it was trained (PPO, N steps), and that it fails on the shown layout.
2. **Game rules:** the grid, the `T0` layout drawn in ASCII, picking, dropping, orders and episode length.
3. **Reward**, including the shaping term used in training, stated honestly.
4. **Observation index map:** the interleaved formula, the channel list and the gauge encoding.
5. **Network formula and shapes:** `W` is `(inputs, outputs)`, row i = input i, plus the tanh layers and the softmax over 7 actions.
6. **The failure card** (§6.5).
7. **Task, in the blog's words in substance:**
   - first read the weights and explain why the network fails, before the edits block
   - keep what already works and fix only the cause
   - **success matters more than changing fewer weights**
   - the result is measured on 1,024 new episodes with sampled actions
8. **Edit format** (§7.2), then "after the edits block, at most about 200 words on what the edits do."
9. **All actor weights, last.**

- **System prompt:** a single neutral line, as in the logs harness [S2a].
- The prompt is **byte-identical across the k attempts** of a condition. That makes the whole prompt a cacheable prefix.
- Freeze the template in `prereg.md`.

### 8.2 Provider settings
**OpenAI `gpt-6-astra` (replication arm)**
- Mirror the logs harness: `codex exec`, reasoning effort `high`, and every built-in tool disabled (shell, edit, web, image, sub-agents, plugins). Copy `harness/codex_config.toml` and `instructions.md` [S2a].
- Alternatively, use the Responses API with reasoning effort `high` and no tools.
- Log a `tool_free` audit per turn.

**Anthropic `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`** [S9][S10][S11]
- Use the official `anthropic` Python SDK, `messages.stream(...)` + `get_final_message()`, `max_tokens` 64k–128k (it covers thinking plus text), **no tools**.
- `output_config={"effort": "high"}`, **set explicitly**. Opus 5.5 and Haiku 5.5 default to `medium`; Sonnet 5.5 defaults to `high`. Opus 5.5 thinking can't be disabled.
- **Don't send `temperature`, `top_p` or `top_k`.** On API version `2026-10-21` they return a 400 on every model, so v1's "fix the temperature" is impossible. Fix and log effort instead.
- **Don't enable server-side refusal `fallbacks`.** A fallback would silently answer with a different model and contaminate the condition. Check `stop_reason`. A `refusal` is logged as a failed attempt with its `stop_details`.
- Measure prompt size with `messages.count_tokens` (free) before any paid call.
- Log `usage` per attempt: input, cache read and write, and output, which includes thinking.

**Log for every attempt, all providers:** model ID, effort, API/SDK version, timestamp, prompt SHA-256, raw reply, parsed edits, skipped lines, usage and cost.

---

## 9. Conditions and controls
| ID | Condition | Purpose |
|---|---|---|
| **M** | Main: the full prompt, k = 5 independent first attempts per model | Headline |
| (a) | Rounded-unedited policy, next to the float policy | True baseline. If rounding alone moves the target metric beyond noise, add a significant digit and re-dump before any call. |
| (b′) | Matched random edits: same tensors, same number of changed entries, same number of units, magnitudes drawn from the model's edit, 20 draws per successful edit | Rules out "any big perturbation helps" |
| (c) | Our own hand-written bypass, or an activation-direction ablation | Shows how hard the fix is without an LLM |
| (d) | More PPO for about the wall-clock of one reply (~7 minutes in Exp 4), reported as a reference | Context only. Beating (d) is not the claim. |
| **(e1)** | The same prompt **with the weights omitted**. No free units named, nothing pre-zeroed. The model writes edits against a stated architecture. | **Reading vs writing.** If e1 ≈ M, the post says "the model wrote a patch from the rules", not "it read the weights". |
| (e2) | Weights omitted, and we name 4 units per layer that we have zeroed, after verifying behaviour is unchanged | Hinted write-from-scratch upper bound |
| (f) | Apply M's edit, unchanged, to another seed's actor that has the same failure | Specific reading vs a generic circuit |
| mech | Zero the units the reply says it used and check that the fix disappears | Does the model's stated mechanism hold? |
| G+E | Main protocol on the entity-aided observation | Did raw entity scalars make it fixable? |

- e1 is run at the gate (Day 4) on Astra and Opus 5.5, k = 5 each. The rest run on Days 5–6.
- **Run after a GO only, if the budget allows:** "no trace" (the failure card without the episode trace), action-probability snapshots added, and the oscillation failure.

---

## 10. Evaluation and statistics
**Seeds:** training, diagnosis (the model sees these episodes) and evaluation seeds are disjoint. Eval seeds are never shown.

**Headline (replication claim):**
- **1,024 paired episodes** on **`T0`**, the layout described in the prompt.
- Actions **sampled**, using the same seeds before and after the edit.
- Success = an episode delivers at least one needed item. Pre-register this definition. It's the analogue of Kinetix's binary success.

**Also reported:**
- **Generalization:** the same metrics on held-out `trap` layouts. This is not required for the replication claim, but always reported.
- **Non-regression (our addition, not in the blog):** mean unshaped score on 1,024 paired **`open`**-layout episodes must not fall by more than **5% relative**. Measure it on `open` because the baseline score on `T0` is about zero.
- **Greedy (argmax)** versions of every metric.
- **Statistics:**
  - 95% bootstrap CIs
  - paired tests: McNemar or a paired bootstrap for binary success, Wilcoxon signed-rank for score
  - **Holm** correction across metrics
  - for the headline first-attempt success rate over k: an exact binomial (Clopper–Pearson) CI
- **Edit analysis:**
  - fraction of weights changed (on the rounded net)
  - L2 of the change
  - units taken over per layer
  - number of statements and skipped lines
  - the mechanism-ablation outcome

**No hard reject on the fraction changed.** Taking over one L1 unit means rewriting its 600 incoming weights. Clearing 3 units per layer is therefore ~2,000 weights, or ~10% of the net, before any new wiring. Exp 4's 3.54% came from a much wider net. If a cap is wanted, pre-register a **cap on units taken over**, e.g. ≤ 6 per layer, which covers Astra's largest slice [S4].

**`prereg.md`** is written and SHA-256-hashed into the log before the first paid call. It contains:
- the hypothesis
- the success definition
- k
- the non-regression rule
- the skipped-line rule
- the unit cap, if any
- the control list
- the decision rule
- the frozen prompt template

---

## 11. Budget
Exp 4's real usage was 355,201 input tokens, 0 cached, and 10,134 output tokens, of which 9,146 were reasoning [S5].
- **Tokens per weight:** the integer format costs ~2.07 tokens per weight on 128-wide rows. Our 32-wide rows amortize the `i:` prefix less, so 2.07 is a floor.
- **Expected prompt size:** **45–80k input tokens** for the primary prompt. Measure with `count_tokens` on Day 3.
- **Exp 4's own bill (correction):** that prompt was over OpenAI's 272k short-context tier, so it cost about 355,201 × $20/M + 10,134 × $75/M ≈ **$7.86**. Review #4's $4.06 used short-tier rates. Our prompt stays in the short tier.

Per attempt, at ~60k input and 10–40k output (thinking included), using official list prices checked 2026-11-04 [S8][S9]:

| Model | In / Out $/MTok (cache read) | ≈ $/attempt |
|---|---|---|
| `gpt-6-astra` (≤272k) | 10 / 50 (1.00) | $1.10–2.60 |
| `claude-opus-5-5` | 4 / 20 (0.20) | $0.44–1.04 |
| `claude-sonnet-5-5` | 2 / 10 (**0.10**) | $0.22–0.52 |
| `gpt-6.1-sol` (≤272k) | 2 / 10 (0.10) | $0.22–0.52 |
| `claude-haiku-5-5` (prompt ≤100k) | 0.10 / 0.50 (0.01) | $0.01–0.03 |
| `gpt-6-luna` (≤272k) | 0.10 / 0.50 (0.01) | $0.01–0.03 |

- **Haiku 5.5's whole request** bills at $0.50 / $2.50 once the prompt exceeds 100k tokens. It's a cheap control only while the prompt stays under that.
- Batch APIs are 50% off on both providers.
- **Gate estimate (Day 4):**
  - M: k = 5 each on Astra, Opus 5.5 and Haiku
  - e1: k = 5 each on Astra and Opus 5.5, with much smaller input
  - Total ≈ **$12–35**
- **Before the batch:** make one paid probe call with the real prompt and trust its cost. Total project budget: stay under v1's **$100** envelope.

---

## 12. Experiment log
- An append-only `runs/log.jsonl`, plus content-addressed weight files. One row per attempt or evaluation:
  `run_id, utc, condition, model_id, effort, api_version, sdk_version, prompt_sha256, prereg_sha256, checkpoint_sha256, raw_reply_path, n_statements, n_skipped, refusal, usage{in,cache_read,cache_write,out}, cost_usd, frac_changed, l2_change, units_per_layer, metrics{T0,trap_heldout,open; sampled,greedy}, CIs, p_values`.
- Every prompt and reply is stored verbatim.
- Publish the log with the repo, with keys removed.

**Suggested repo layout** (the builder may change it):
`env/` · `train/` · `diagnose/` · `serialize/` (dump, parse, apply, plus the golden test) · `llm/` (one adapter per provider) · `eval/` · `export_usd/` · `render_ovrtx/` (its own venv) · `runs/` · `prereg.md` · `ATTRIBUTION.md`

---

## 13. 3D replay and video
**Assets** [S19]:
- `nvidia/simready-assets` now carries a **cc-by-4.0** licence tag and lists **2,677 assets, about 490 GB**. **Never clone it.** Use `huggingface_hub.snapshot_download(repo_type="dataset", allow_patterns=[...])` per package.
- Start from the warehouse-specific `nvidia/PhysicalAI-SimReady-Warehouse-01` CSV (CC-BY-4.0, ~15 GB) to pick racks, totes and pallets. `rack_large_empty_a02` is a confirmed rack.
- The robot is **[UNVERIFIED]**: search for an AMR or forklift, otherwise use a box-on-wheels proxy.
- Read each package's `LICENSE.txt` into `ATTRIBUTION.md`.
- Avoid the older Omniverse content packs, which carry restrictive Isaac Sim asset terms.

**Stage authoring (main venv, `usd-core`):**
- 1 grid cell = 1.2 m. Assets are referenced as USD references or payloads.
- Per grid step: **rotate in place, then translate** (e.g. frames 0–3 rotate, 4–12 translate), as time-sampled `xformOp:rotateZ` and `xformOp:translate`.
- Carried totes are child prims with time-sampled visibility.
- Before and after episodes run on the **same seed**, side by side with an X offset or in two stages.
- The `.usda` is the portable canonical artifact.

**Rendering, primary path: ovrtx + ovstage, in its own venv** [S14][S15][S16]
- `pip install ovrtx ovstage`. Python 3.10–3.13, **≥ 3.11 on Windows**. Windows 11 or Ubuntu 22.04/24.04, an RTX GPU, VC runtime ≥ 14.38. Both packages are pre-release under the NVIDIA SLA, which is fine for rendering our own frames. Expect 1–2 minutes of shader compilation on first run.
- Loop:
  1. `ovrtx.Renderer()` first. It loads OpenUSD.
  2. `ovstage.Stage(...)`, then `renderer.attach_ovstage(stage)`.
  3. `ovstage.population.open_usd(stage, root, ordinal=1)`, then `stage.advance_write_floor(1, ovstage.Scope.ALL).wait()`.
  4. Per frame:
     - `population.update_from_usd_time_async(stage, ordinal=n, time=seconds).wait()`. Time is in **seconds**.
     - `advance_write_floor(n, ...)`.
     - `renderer.step(render_products={"/Render/Camera"}, delta_time=1/30, ordinal=n)`.
     - Map `LdrColor` to numpy, then save a PNG.
  5. Run ffmpeg over the frames.
- **There are two independent clocks.** `step()` doesn't advance USD animation. Drive both.
- **Fallback within ovrtx:** write batched `omni:xform` `(N,4,4)` float64 row-major matrices per frame with `stage.write_attribute(...)`, then `compute_hierarchy` and `advance_write_floor`. This drives the replay straight from the trajectory JSON.
- Add the Camera, RenderProduct and RenderVar through an **inline root layer** that sublayers the replay `.usda`. The assets are never edited.
- **Don't import `usd-core` and `ovrtx` in the same process**, because of the OpenUSD load order.
- Copy the code from the docs on the day, not from this plan.

**Day-7 spike (1–2 h) before committing:** render one SimReady rack, a box robot and a camera for 30 animated frames. Confirm three things:
1. **[UNVERIFIED]** time-sampled `xformOp`s animate through `update_from_usd_time`. If not, use the `omni:xform` path.
2. **[UNVERIFIED]** SimReady MDL/OmniPBR materials look right.
3. Local references and payloads resolve.

**Fallback order:**
1. ovrtx
2. **current-stable Isaac Sim** (6.2.0 per its 2026-10-14 compatibility page, per Review #4, **[UNVERIFIED]** by me), Movie Capture, its own Python
3. three.js with glTF plus the trajectory JSON, which doubles as an interactive demo
4. Blender, last: its USD importer doesn't apply MDL, so materials must be re-assigned

**Video (60–90 s, 1080×1350 or 1920×1080):**
1. **0–5 s:** hook, "I asked an AI to perform surgery on a robot's brain"
2. **5–15 s:** the game in 2D
3. **15–30 s:** the trained robot stuck in the dock pocket (3D)
4. **30–45 s:** a weight heatmap with a quote from the diagnosis, the edited units highlighted, and forward-hook activations lighting up as it escapes
5. **45–70 s:** before and after side by side with live score counters
6. **70–85 s:** stats card: first-attempt success k/5, the CI, % weights changed, $ per edit, **and the e1 (no-weights) result**
7. **85–90 s:** credit to MIT CSAIL, links

Compose in DaVinci Resolve or ffmpeg. Score overlays are rendered from the trajectory JSON.

---

## 14. Hardware and software stack (RTX 5090, Windows 11)
- **Driver: 595.97** (R595).
  - ovrtx validates 581.42 (R580) and 595.97 for GeForce Blackwell on Windows [S16].
  - Windows 11 KB5074109 (2026-01-13) requires ≥ 582.41 on R580 for Vulkan, so v1's "install 581.42" is wrong on an up-to-date machine.
  - 595.97 also satisfies PyTorch's Blackwell requirement (≥ 580.88) [S17].
  - The old 595.x crash reports were against Isaac Sim 5.1. Don't downgrade the driver to chase them.
  - In WSL/Linux, use 580.95.05 or 595.58.03.
- **PyTorch:** use the current stable **cu130** wheel (`--index-url https://download.pytorch.org/whl/cu130`). cu128 is deprecated as of 2.12, and cu132 is experimental. Check that `torch.cuda.get_arch_list()` includes `sm_120`. Avoid cu126 wheels and packages that silently downgrade torch. Training is CPU-bound anyway.
- **venvs:**
  1. main: env, training, LLM harness, eval and `usd-core`; Python 3.11/3.12
  2. ovrtx
  3. Isaac Sim's own Python, only if used
- **SDKs:** `anthropic` (official Python SDK) for Claude. Codex CLI or the `openai` SDK for GPT-6, configured with tools off.

---

## 15. Milestones and gates (~1.5–2 weeks part-time)
| Day | Work | Exit criterion |
|---|---|---|
| 1–2 | Env, ASCII renderer, metrics, scripted-greedy and random baselines, determinism tests. Serializer, parser and applier. **Exp 4 golden test.** Round-trip and transpose tests. | All tests green, golden test reproduces 6,073 changed weights |
| 3 | PPO with 3–5 seeds and trap shaping. Pick the checkpoint. **Competence check** and **capacity check**. Failure card. Measure tokens with `count_tokens`. Write and hash `prereg.md`. One paid probe call. | **Gate A:** competence and capacity pass, prompt ≤ ~80k tokens, probe cost known |
| 4 | **Gate B (GO/NO-GO):** M with k = 5 on Astra, Opus 5.5 and Haiku. e1 with k = 5 on Astra and Opus 5.5. Evaluate (a) and M. | **GO** = at least one significant, non-regressing fix on `T0`, *or* a clean documented negative worth publishing. No 3D work before GO. |
| 5–6 | (b′), (c), (d), e2, (f), mechanism ablation. Sonnet 5.5 and GPT-6.1 Sol on M. G+E condition. Held-out trap layouts. | Full results table |
| 7 | **ovrtx spike**, then the full render, falling back per §13 | Before/after clips rendered |
| 8–10 | Video, post, publish the repo with the log, prereg and `ATTRIBUTION.md` | Published |
| Stretch | 3 robots with deadlocks. Write a policy from scratch (the Exp 5 analogue). The oscillation failure. | — |

---

## 16. Fair play and licensing
- **Grocery Bot** [S13]: "any method and any language" is welcome, including AI assistants, and there is **no scheduled end date**. The page doesn't forbid publishing strategies.
- **Our rule (a courtesy, stated as such):** publish nothing tuned to Grocery Bot stores. Don't reuse its code, protocol, maps, assets or branding, and don't connect to its servers.
- Our game is structurally distinct: one robot, one dock, no trucks or away times, no three-cell load zones, no ID-ordered move resolution.
- **Licences:**
  - MIT logs repo: CC BY 4.0 (logs and networks), MIT (scripts). Using the Exp 4 networks for the golden test requires attribution.
  - SimReady assets: CC BY 4.0 per package.
  - ovrtx and ovstage: NVIDIA SLA, not redistributed.

---

## 17. Risks
- **PPO won't produce a clean failure that also passes the competence check.** Iterate on the shaping and the trap geometry. Disclose the process.
- **A 32-wide tanh net may have no spare capacity.** The capacity check catches this, and the fix is widening to 64.
- **The model "fixes" `T0` by over-steering and hurts `open` layouts.** Non-regression catches this.
- **e1 succeeds.** Then the post's claim changes to "writes a patch". That is still interesting, and it's the honest headline.
- **Results don't carry over from Astra to Claude, or the reverse.** Report it. That's a finding.
- **Parser or format mismatch.** The golden test catches it on Day 1. Re-read stored replies; don't re-call.
- **Refusals.** Logged as failed attempts. No silent fallback models.
- **ovrtx is pre-release.** The spike plus the fallback chain covers it.
- **Prices change.** Re-check both pricing pages on the morning of Day 4.

---

## 18. Corrections ledger (why this plan differs from the earlier documents)
| Topic | Earlier claim | Final | Source |
|---|---|---|---|
| Weight text | v1: 3–4 decimals, row = out-unit. R2: 3 decimals | Integer × per-tensor scale, row = input | [S3] |
| Edit values | R3: integers in the tensor's scale | **Real decimals**: "plain decimal numbers (absolute value at most 1e6)" | [S3] |
| Edit grammar | R2: JSON diffs. R3: a small subset | Full Exp 4 grammar, last block wins, skip bad lines | [S3][S4] |
| Grid index | R3: planar `offset_k + 10r + c` | **Interleaved** `(r*10+c)*6+ch` | [S6][S7] |
| Observation | R2: 12 derived scalars at `obs[0:12]` | None in the primary condition. Raw scalars only in G+E. | — |
| Actor/critic | R2: SB3 shares a backbone | Separate by default. Assert anyway. | [S12] |
| Units Astra used | R2: "3–4 lightly-used per layer" | embed 3, actor0 6, actor1 4, actor2–4 2 each, head rows 126–127 | [S4] |
| Prompt tokens | v1: 90–130k. R3: 25–40k | **45–80k**, to be measured. Exp 4: 355,201 in / 10,134 out (9,146 reasoning) | [S5] |
| Exp 4 Astra cost | R4: ≈ $4.06 | ≈ **$7.86**, because 355k > 272k puts it in the long-context tier | [S5][S8] |
| Param count | R2: 20,903 (612 in). R4: 20,711 for 600 in | **20,519** for 600→32→32→7 (20,711 is the G+E net) | arithmetic |
| Sonnet 5.5 cache read | R3: $0.20 | **$0.10** | [S9] |
| Temperature | v1: fix it | Not sendable on API 2026-10-21. Fix and log effort. | [S11] |
| Weight cap | v1: ≤ 10% hard | Report it. Optional pre-registered unit cap. | — |
| Hoarder trap | R2: recommended | Rejected (γ-discounted reward still pays). Dock pocket with shaping instead. | [S3] |
| Quiet-unit gate | R4: "< 4 units in bottom quartile" | Ill-formed. Use the ablation-based capacity check (§6.4). | — |
| Driver | v1: 581.42 | **595.97** | [S16][S17] |
| Isaac Sim | v1: 6.0.0 | Current stable as fallback 2 only | [S18] |
| SimReady licence | v1: no tag, 2,575 assets | cc-by-4.0 tag, 2,677 assets, 490 GB | [S19] |
| GB publishing rule | v1: implied competition rule | Self-imposed courtesy | [S13] |

---

## 19. Sources (accessed 2026-11-04)
- [S1] Takehi, Teoh, Zhang, Khattab, "Sparks of Weight-Space Programming", MIT CSAIL, 2026-11-02. https://rikiyat.github.io/blog/weight-space-programming/
- [S2] Exp 4 README (format, edit language, networks). https://github.com/RikiyaT/lm_understand-logs/tree/main/exp4_fix_car · [S2a] logs README and harness (model, effort, tools off, licences). https://github.com/RikiyaT/lm_understand-logs
- [S3] Exp 4 prompt (577,774 chars; edit format, reward with `0.2×(d_prev−d)`, `W` is `(inputs, outputs)`). https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp4_fix_car/runs/oneshot_repair/reply/prompt.txt
- [S4] Exp 4 reply (51 statements). https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp4_fix_car/runs/oneshot_repair/reply/answer.md
- [S5] Exp 4 `run.json` (355,201 in, 10,134 out, 9,146 reasoning, effort high, `tool_free: true`, 412.8 s). https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp4_fix_car/runs/oneshot_repair/reply/run.json
- [S6] Exp 5 README (`(row*10+column)*6+channel`, 20,486 parameters, `xhigh`, parser re-read). https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp5_write_from_scratch/README.md
- [S7] Exp 1 README (3 decimals, interleaved index). https://github.com/RikiyaT/lm_understand-logs/tree/main/exp1_which_game
- [S8] OpenAI pricing, official page (Astra $10/$1/$50 at ≤272k and $20/$2/$75 above; Sol $2/$0.10/$10; Luna $0.10/$0.01/$0.50; Batch 50%). https://developers.openai.com/api/docs/pricing
- [S9] Anthropic pricing, official page (Opus 5.5 $4/$20, cache read $0.20; Sonnet 5.5 $2/$10, cache read $0.10; Haiku 5.5 $0.10/$0.50 ≤100k and $0.50/$2.50 above; Batch 50%). https://platform.claude.com/docs/en/about-claude/pricing
- [S10] Claude effort docs (effort defaults; Opus 5.5 thinking always on), as cited in Review #4 and the bundled Claude API reference. https://platform.claude.com/docs/en/build-with-claude/effort
- [S11] Anthropic API versions (`2026-10-21` removes `temperature`, `top_p`, `top_k`: "returns a 400 error on every model"). https://platform.claude.com/docs/en/api/versioning
- [S12] Stable-Baselines3 2.9.2 and policy networks. https://github.com/DLR-RM/stable-baselines3/releases/tag/v2.9.2 · https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
- [S13] Grocery Bot, How to play. https://grocerybot.no/how-to-play
- [S14] ovrtx Python getting started. https://nvidia-omniverse.github.io/ovrtx/python_api/getting_started.html
- [S15] ovrtx Loading USD (two clocks, time in seconds, inline root layers). https://nvidia-omniverse.github.io/ovrtx/scene/loading_usd.html · ovstage transforms: https://nvidia-omniverse.github.io/ovstage/scene/transforms.html
- [S16] ovrtx hardware and driver requirements (581.42 / 595.97; KB5074109). https://nvidia-omniverse.github.io/ovrtx/driver_requirements.html
- [S17] PyTorch 2.12 release (cu130 default, cu128 deprecated, Blackwell driver ≥ 580.88), as cited in Review #4. https://pytorch.org/blog/pytorch-2-12-release-blog/
- [S18] Isaac Sim compatibility and releases, as cited in Review #4. https://docs.isaacsim.omniverse.nvidia.com/latest/installation/compatibility.html
- [S19] `nvidia/simready-assets` (cc-by-4.0, 2,677 assets, ~490 GB). https://huggingface.co/datasets/nvidia/simready-assets · Warehouse set: https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01

**Still [UNVERIFIED]:**
- whether ovrtx animates time-sampled `xformOp`s, and how faithfully it renders SimReady MDL (Day-7 spike)
- the key names inside the Exp 4 `.npz` files (check before the golden test)
- the current stable Isaac Sim version and its listed driver (only if we fall back)
- an AMR or forklift asset name
- our real prompt token count on each tokenizer (Day 3)
