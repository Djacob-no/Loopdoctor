# WeightShift (Loop Doctor): Review #4, last pass before the merged plan
*Date: 2026-11-04 · Reviews `ORIGINAL_PLAN.md` (v1), `weightshift_plan_review_recommendations.md` (Review #2), and `REVIEW_3_recommendations.md` (Review #3) · Audience: whoever concatenates these into the build plan*

> Checked against the blog, the Exp 4 prompt file (577,774 characters, downloaded), `answer.md`, `run.json`, the Exp 1 and Exp 5 READMEs, the ovrtx docs, the live Grocery Bot rules, the SimReady dataset card, and the current Claude effort docs. Claims below that I did not open a primary page for are marked **[UNVERIFIED]**. This file does not replace the plan. It is the last set of corrections to fold in.

---

## 1. Verdict
The experiment is real, and the right thing to copy is Experiment 4's **protocol**, not a paraphrase of it. Review #3 is the best of the three documents and should be the base of the merge. It is right about the integer-times-scale weight dump, the row-is-input layout, the quantized baseline, the competence check, the no-weights control, ovrtx, and the Claude effort defaults.

It is wrong, in ways that would make a faithful harness reject a real Exp 4 reply, on four points:

1. **Edit values are plain decimals in real weight units, not integers in the tensor's scale.** The prompt's own examples are `1.25`, `0.5`, and `-= 0.1`. Astra's reply uses `80`, `-200`, and `+= 100` as real weights. Read as scaled integers, those numbers are not the "large logit margins" the reply describes.
2. **The edit grammar is larger than Review #3's grammar.** It includes slice fills, list assignment, `+=` `-=` `*=`, `ZERO tensor`, comments, continued lines, and "skip a line you cannot read; apply the rest." Only the last `edits` block counts.
3. **The grid index formula in the blog is interleaved, not planar.** Exp 1 and Exp 5 both use `i = (row * 10 + column) * n_channels + channel`, with row 0 at the top.
4. **"Output dominates the cost" and "25–40k input tokens" do not match the measured run.** Exp 4 was **355,201 input tokens** and **10,134 output tokens, of which 9,146 were reasoning**. Input was the bill. Our smaller net should land nearer **45–80k input tokens** until we measure it. The 3× token saving versus 3-decimal text is about **2×** on the published runs.

Two design choices in Review #3 should be relabelled rather than copied as "exactly Exp 4":

- The Day-4 gate should include **`gpt-6-astra`**, the only model the blog showed doing this. Claude is the generalization arm. Astra at this size is a few dollars an attempt, not a reason to drop it.
- A hard 10% weight budget, and a parser that rejects the whole reply, both diverge from the prompt that produced the 1,024/1,024 result.

---

## 2. What the published Exp 4 run actually was
Sources: blog [S1], Exp 4 README [S2], `prompt.txt` [S3], `answer.md` [S4], `run.json` [S5].

| Fact | Value |
|---|---|
| Model | `gpt-6-astra` via `codex exec`, effort `high`, tools off. `audit.tool_free: true` |
| When | Run ended `2026-10-16T01:57:39-0400`. Wall clock 412.8 s |
| Context the harness set | `model_context_window=872000` (do not cite a 1.05M window for this run) |
| Network | Official Kinetix actor: 679 → 128 ReLU → five tanh layers of 128 → 16 logits. **171,664 actor weights.** Value head exists and is **not shown** |
| Training | Official PPO, **1,001,390,080** steps. 0/256 successes at the end, and 0 at every earlier eval |
| Prompt | 577,774 characters, 1,567 lines. Task, scene, observation index ranges, network formula, 256-episode summary, one episode (positions every 16 steps), edit-format spec, then all actor weights |
| Tokens | input 355,201, cached 0, output 10,134, reasoning 9,146 |
| Edit | **51 statements**, all applied, 0 errors. **6,073 / 171,664 weights changed (3.54%)**, counted on the rounded network. Most of those are zeros on the units the path takes over |
| Score | 1,024/1,024 test episodes, actions sampled. Also 256/256 on the search stream. The scene is deterministic; episodes differ only by action samples |
| Units actually taken | embed **6, 7, 8** (3). actor0 **122–127** (6). actor1 **124–127** (4). actor2, actor3, actor4 **126–127** (2 each). Head rows **126 and 127** write action-entry-0 logits. Not "3–4 units per layer", and the reply never says those units were lightly used |

The logs README is explicit that **trial runs of earlier prompt versions were left out**. "One try" means one reply to the final prompt, with no feedback and no second try. It does not mean the prompt was never drafted. Our rule is stricter and better: pre-register the prompt, and if we change it after seeing a result, that is a new condition and the old attempts stay in the log.

Per-tensor scales in the published prompt, so the merge does not invent one global scale:

| Tensor | Scale |
|---|---|
| `embed.W` | `1e-2` |
| `embed.b` | `1e-4` |
| `actor0.W` | `1e-3` |
| `actor0.b` | `1e-4` |
| `actor1` through `actor4` weights and biases, and `actor_head.W` / `actor_head.b` | `1e-3` |

The README's rule for choosing the scale: integers times one scale, **3 significant digits of that tensor's largest entry**. Implement that, then check a round-trip against a dumped tensor before any model call.

---

## 3. The edit language to implement
This replaces Review #3 §3 items 2 and 3. The weight **display** is integer × scale. The edit **values** are not.

From the prompt, verbatim in substance [S3]:

- A code block whose opening line is three backticks plus the word `edits`.
- One statement per line. `#` starts a comment. A long list may continue on the next lines.
- Only the **last** `edits` block is read.
- Statements apply **in order** to the network **as shown** (the rounded weights). A later statement overwrites an earlier one.
- A statement that cannot be read is **skipped**. The others still apply.
- Values are plain decimal numbers, absolute value at most `1e6`.

Forms the prompt itself demonstrates:

```
actor_head.b[0:3] = [0.5, 2.0, -1.0]   # list, one value per entry; a:b is half-open
actor_head.W[7, 0:3] = 0               # fill a slice; ':' alone is the whole axis
embed.W[26, 5] = 1.25                  # one entry, real value, not an integer in the scale
actor0.W[:, 3] *= 0.5                  # also += and -=
embed.b[5] -= 0.1
ZERO actor4.W                          # whole tensor to 0
```

Astra's scored reply also uses `W[a:b, :] = 0`, `W[:, a:b] = 0`, `b[a:b] = [..]`, and `b[i] += n` [S4]. Exp 5's reader initially rejected a space in `[ [ ... ], ... ]`, then the **same replies** were re-read with a fixed parser and those are the networks that were scored [S6]. So:

- Store the raw reply before parsing.
- Accept the full grammar above, including `ZERO`, `*=`, `+=`, `-=`, list assignment, and a continued list.
- A parser bug is not a model failure. Fix the parser and re-read the stored reply. Do not spend another call.
- Skipping a bad line matches Exp 4. Log every skipped line. For the headline, pre-register that an attempt with any skipped line is reported and is not counted as a clean success. Do not delete the attempt.
- Apply numbers as **real weights on the dequantized rounded tensors**. `actor0.W` scale `1e-3` does not turn `= 80` into `0.080`.

`nn.Linear.weight` is `(out, in)`. The blog's `W` is `(inputs, outputs)`, and `h = act(x @ W + b)`. Export and import must transpose. Dumping `layer.weight` straight into the text file silently transposes the policy.

---

## 4. Token and cost math, from the run rather than from a guess

| Published run | Weights | Input tokens | Tokens per weight | Chars per token |
|---|---|---|---|---|
| Exp 4, integer × scale [S5] | 171,664 | 355,201 | 2.07 | 1.63 (577,774 / 355,201) |
| Exp 1, 3 decimal places [S7] | 17,253 | about 73,700 | about 4.27 | — |

The integer format is about **2×** cheaper than 3-decimal text on these two runs, not 3×. Exp 4's rows are 128 wide, so the `i:` prefix is amortized. Our rows are 32 wide, so 2.07 tokens/weight is a floor, not a forecast.

For a 600→32→32→7 tanh actor (20,711 parameters: `600·32+32 + 32·32+32 + 32·7+7`) a planning range is **45–80k input tokens** until `count_tokens` says otherwise. A 612-input net is 20,903, which is Review #2's layout, not Exp 5's. Exp 5's 600→32→32→6 net is **20,486** parameters [S6].

Exp 4's Astra bill at the published $10 / $50 rates was about `355201·10 + 10134·50` dollars per million, **about $4.06**, and input was 87% of it. Review #3's "output now dominates" is true only if thinking runs long on a small prompt. Astra at effort `high` used 9,146 reasoning tokens on the big prompt. Exp 5 at effort `xhigh` used 16,828–25,694 reasoning tokens on the from-scratch task [S6]. Budget both.

Recomputed gate, short-context list prices, before cache. Confirm the OpenAI page the morning of the first call. The aggregators agree with v1's Astra $10/$50, GPT-6.1 Sol $2/$10, Luna $0.10/$0.50, and with the 272k short-context tier [S8]. I did not load the official OpenAI pricing HTML in this pass, so those three stay **confirmed by several independent writeups, not by me opening the official page**.

| Model | Rate in/out $/MTok | One attempt at 60k in, 10–40k out |
|---|---|---|
| GPT-6 Astra | 10 / 50 | about $1.10–$2.60 |
| Claude Opus 5.5 | 4 / 20 | about $0.44–$1.04 |
| Claude Sonnet 5.5 | 2 / 10 | about $0.22–$0.52 |
| Claude Haiku 5.5, prompt ≤100k | 0.10 / 0.50 | about $0.01–$0.03 |

Haiku 5.5 bills the **whole** request at $0.50 / $2.50 once the prompt exceeds 100k tokens [S9]. Keep the replication prompt under that line or stop calling Haiku a cheap control.

Cache reads: Opus 5.5 is $0.20/MTok. Sonnet 5.5 is **$0.10/MTok**, not $0.20. Review #3 grouped them. Astra cached input is reported at $1/MTok [S8]. The weight dump has to be a stable prefix, identical across the k attempts, or the cache does nothing.

k = 5 on Astra, Sonnet 5.5, and Haiku, plus the no-weights control on Astra and Sonnet, is about **$20–50** if thinking stays in the range above. That still sits inside v1's $100 envelope. One paid probe call before the batch, with the real prompt, is the number to trust.

Claude API facts I re-checked on the current effort doc [S10], which matches Review #3:

- Opus 5.5 and Haiku 5.5 default to effort `medium`. Sonnet 5.5 defaults to `high`. Set `output_config.effort` explicitly.
- On Opus 5.5, adaptive thinking is always on. `thinking: {"type": "disabled"}` returns 400. `max_tokens` counts thinking plus text. Stream, and set `max_tokens` with room (128k is the documented max on these models).
- `temperature`, `top_p`, and `top_k` return 400 on API version `2026-10-21`, even at the default value [S11]. On Claude 4.7 and later they already failed for non-default values on the old version. Do not send them. Log effort, model id, API version, and `usage`.

v1's "Claude 4.7+ tokenizer emits ~30% more tokens" appears in secondary pricing writeups. It is unused if we call `count_tokens`. Leave it out of the plan.

---

## 5. Observation layout
Use the blog's grid convention for the primary condition. Exp 1 [S7] and Exp 5 [S6]:

```
i = (row * 10 + column) * n_channels + channel
row 0 is the top, column 0 is the left
```

Review #3's `offset_k + 10*r + c` is planar and is not what Astra read or wrote. An index map of a few lines belongs in the prompt. Per-column `(channel, row, col)` tags do not.

Primary net, matching Exp 5's shape as closely as this game allows:

- 6 binary channels × 10×10 = **600** inputs, tanh, **600 → 32 → 32 → 7**.
- Channel order, in the interleaved slots: robot, obstacles/racks, needed-SKU racks, dock, trail (previous robot), gauge.
- The gauge is a bar or a block of cells that encodes carried count and needed counts, the way Exp 5's sixth channel is a gauge. Document the encoding in the prompt in one short paragraph.
- Actions: one 7-way softmax, sampled. Kinetix's six independently sampled action entries are a different action space. Do not copy that structure.
- No sticky-action noise. That was Exp 5's MinAtar scoring, not Exp 4.

Do not put Review #2's derived features in this condition. `dx_to_dock`, `steps_since_last_pick_or_drop`, `adjacent_to_needed_sku`, and `is_inventory_full` already contain the diagnosis. A win would not show that the model read the weights.

A second, labelled condition is worth running if Day 4 is a go: the same 600 channels plus a handful of **raw** scalars (robot x, y, dock x, y, items carried, time fraction). No relative vectors, no stall timer, no "can pick" bit. Report it as its own row.

Activation choice, say it in the preregistration: all tanh, because that is the grid-policy shape in Exp 1 and Exp 5. The Kinetix actor's first layer is ReLU, which makes a hard off-switch easier. We are not reproducing that. If the edit can only work as a hard switch, that is a finding.

Width is a Day-3 measurement, not a token decision. Kinetix had 128 units and used the high indices as a side path. A trained 32-unit net may have no spare units. For our own go/no-go, before any paid call: compute, per layer, the L2 of each unit's outgoing weights. If fewer than four units sit in the bottom quartile, widen to 64 and retrain. Do not put that list in the prompt. Review #2 was right to want the report and right to hide it from the model. Review #2 was wrong that the blog established those units were lightly used.

Actor and critic stay separate. SB3 2.9.2 (released 2026-10-26) is real, and its default `MlpPolicy` is already separate `pi` and `vf` MLPs; the vector-obs feature extractor is a parameter-free flatten [S12]. Review #2's "shared backbone by default" is not the current default. Use a single-file CleanRL-style PPO anyway, because the export is six tensors and the transpose is visible. Assert at save time that no actor parameter is a critic parameter. Serialize the actor only.

---

## 6. The failure, and the check that has to pass first
The Gemma sentence is exact [S1]: "Editing weights could remove an obstacle that hid an ability, but it could not add an ability the model lacked." The competence check in Review #3 is a prerequisite, not a nice-to-have.

Review #2's hoarder is a weak trap for the reason Review #3 gives. At γ = 0.94 a delivery 18 steps away is still worth about a third of its reward, inventory is capped at 3, and PPO with an entropy bonus usually finds the dock on a 10×10. Also, `+1` per pick is not this game's score. The live Grocery Bot page awards **+1 when an item is loaded into a truck** and **+5 when that truck's order completes** [S13]. v1's `+1` per delivered item and `+5` per completed order is the right shape. Keep it.

Review #3's dock pocket is the right picture for the video. The way Exp 4 got the local minimum is more specific than "train mostly on easy maps." The Kinetix reward adds `0.2 * (d_prev - d)` toward the blue box [S3]. That shaping pays the car for driving at the ramp. The analogue here:

- On trap layouts, the straight line from the pick area to the dock runs into the closed side of a U of racks. The dock is reached by going the long way around.
- Add a small potential-based bonus for reducing distance to the dock, disclosed as shaping.
- Target: near-zero deliveries on trap layouts, and normal deliveries when the dock is open or the U is mirrored so the opening faces the robot.
- The competence check, logged on the failure card: the same checkpoint delivers when the pocket is mirrored, rotated, or absent. If it cannot route around a wall anywhere, the ability is missing. Change the checkpoint. Do not call the model.

"Train on 95% open layouts and hope" can fail that check, because the policy never acquired the detour. Prefer shaping-into-the-pocket on layouts the policy otherwise solves. Document whichever procedure actually produced the checkpoint. v1 is right that an engineered failure is allowed if it is disclosed.

The second failure, symmetric-aisle oscillation, can wait until the first one has a result.

---

## 7. What "the Exp 4 condition" means, and what we add on purpose
Match these, and say so:

- One tool-free reply. No sandbox, no forward pass, no second try, no "here is the error."
- Task text in the blog's order: explain the failure **before** the edits block; keep what already works; fix only the cause; **success outranks a smaller edit**; edits are scored on new episodes with **sampled** actions.
- After the block, at most about 200 words on what the edits do.
- Show the rounded actor, the index map, the aggregate of the diagnosis episodes, and one episode trace. Exp 4 subsampled positions every 16 steps. A full 200-step ASCII trace of our game is small; include it. Do not add action-probability snapshots to this condition. Those are an extra ablation.
- Headline eval: **1,024 episodes, paired seeds, actions sampled**, on the **same layouts** the prompt described. Exp 4 had one deterministic scene. Unseen layouts are a separate metric. A model can replicate Exp 4 by fixing the layouts it was shown and still fail to generalize. Report both. Do not require the unseen-layout number for the replication claim.
- Also report greedy (argmax). The blog's headline is the sampled one.
- Baseline (a) is the **rounded, unedited** policy, next to the float policy. If rounding itself moves the target metric by more than the eval noise, use one more significant digit and re-dump before any call.
- k ≥ 5 **independent first attempts**, same prompt, same checkpoint. The headline is the first-attempt success rate with a CI. Best-of-k is labelled best-of-k. Exp 4 was k = 1. We are improving on it, not copying the sample size.

Add these, labelled as ours:

- Non-regression: mean score on the paired episodes does not fall by more than 5% relative. Exp 4 did not require this. It asked for success first. Keep the rule, and do not pretend it is in the blog.
- No hard reject at 10% of weights. Zeroing three columns of a 600-wide first layer is 1,800 weights, about 9% of a 20k-parameter net, before the rest of the path. Exp 4's 3.54% fell out of a 128-wide net. Report the fraction, the L2, and the units. If a cap is wanted, cap **units taken over** (for example 6 per layer, which covers the actor0 slice Astra used) and pre-register that.
- (b′) Matched random edits: same tensors, same number of changed entries, same number of units, magnitudes drawn from the model's own edit. 20 draws. Keep.
- (c) A hand-written bypass or an activation ablation, computed by us. Keep.
- (d) More PPO for about the wall-clock of one reply (about 7 minutes in Exp 4). Report it as a reference. A policy that survived a billion steps will not move in seven minutes. Beating (d) is not the result.
- (e) split in two, because Review #3's version gives the model a hint Exp 4 never gave:
  - **(e1)** Same prompt, weights omitted, no free units named, nothing pre-zeroed.
  - **(e2)** Weights omitted, and we name a few units we have zeroed, after checking the zeroing did not change behaviour.
  - If (e1) matches the main condition, the post is "the model wrote a patch from the rules and the episode," not "the model read the weights." Exp 5 already shows a frontier model can write a 600→32→32→6 grid policy from the rules alone, so (e1) can succeed. That is the result, if it happens.
- (f) Transplant the edit onto another seed's actor with the same failure. Keep.
- Mechanism check: zero the units the reply names and see whether the fix disappears. Keep. Forward hooks for the video are fine and are not part of the prompt.
- `prereg.md` hashed before the first paid call: hypothesis, target metric, k, non-regression rule, unit cap if any, the control list, the decision rule, and the prompt template.

---

## 8. Rendering and the software stack
These checks came out in Review #3's favour. Small updates only.

**ovrtx.** The getting-started page really does `ovrtx.Renderer()`, `ovstage.Stage(...)`, `attach_ovstage`, `population.open_usd`, `advance_write_floor` [S14]. The loading page really does say the two clocks are independent, and it shows `update_from_usd_time_async(..., time=time_seconds)` with time in seconds [S15]. Python 3.10–3.13, and 3.11+ on Windows. Pre-release. The snippet in Review #3 is close enough to start from, and it should still be copied from the docs on the day it is run, not from this review.

Still **[UNVERIFIED]**, and still the Day-7 spike: whether time-sampled `xformOp:translate` / `rotateZ` move under `update_from_usd_time`, whether SimReady MDL/OmniPBR looks right, and whether local asset references resolve. Rotate in place, then translate. That part of Review #2 is worth keeping.

**Drivers.** ovrtx's validated Windows GeForce Blackwell drivers are 581.42 (R580) and 595.97 (R595) [S16]. Windows 11 KB5074109 (2026-01-13) needs ≥582.41 on R580, or 595.97 on R595, for Vulkan. v1's "install 581.42" is the wrong instruction on a current Windows 11 box. PyTorch 2.12's release note says the default wheel is CUDA 13.0, cu128 is deprecated, cu132 is experimental, and Blackwell wants a Windows driver ≥580.88 [S17]. **595.97 satisfies both of those pages.** Current PyTorch is past 2.12 (2.12.1 and 2.13 are on the previous-versions list). Install the current stable cu130 wheel, and re-read both requirement pages on Day 7 rather than pinning 2.12.

**Isaac Sim.** 6.0.0 GA was 2026-06-04/08, and 6.0.1 exists. The compatibility page and the repo history show **6.1.0 and 6.2.0 GA** as well, with a 7.0.0a1 pre-release commit [S18]. The fallback is the current stable release, which was 6.2.0 on the 2026-10-14 compatibility page, not 6.0.0. The 595.x crashes were reported against Isaac Sim 5.1. A later report says 6.0 plus a recent Game Ready driver ran. Do not downgrade the driver to chase 5.1. Keep Isaac Sim in its own Python (3.12 for the 6.x Isaac Lab line). It stays fallback 2.

Fallback order stays: ovrtx, current Isaac Sim, three.js plus the trajectory JSON, Blender last. Blender's USD importer still does not apply MDL.

**Assets.** `nvidia/simready-assets` card data is `license: cc-by-4.0`, tagged `license:cc-by-4.0`, **2,677 assets, about 490 GB** [S19]. `rack_large_empty_a02` is on that dataset. Download packages with `allow_patterns`. Do not clone. Read each package's `LICENSE.txt` into `ATTRIBUTION.md`. The robot asset name is still **[UNVERIFIED]**; a box proxy is the planned fallback. `PhysicalAI-SimReady-Warehouse-01` remains the smaller warehouse-specific set from v1. I did not re-open its README in this pass.

**Grocery Bot, from the live rules page [S13].** Any method and any language, including AI coding assistants. No scheduled end date. Score is +1 per item loaded and +5 per completed truck. The stores run from 12×10 with 1 robot and 2 trucks up to 30×18 with 20 robots and 6 trucks. Easy is 4 products and 3–4 items per order. 300 rounds, 500 on Nightmare. The page does not forbid publishing strategies. v1's "do not publish a competition policy" is our courtesy. Keep it, and call it that. Our game stays one robot, one dock, no trucks, no three-cell load zone, no ID-ordered move resolution, and the name Grocery Bot stays out of the title.

---

## 9. Decisions to lock when the four documents are merged

1. **Primary condition = Exp 4's prompt contract**, with our 10×10 one-robot game. Tool-free, one reply, integer-times-scale dump, real-valued edit language from §3, rounded network as the baseline, sampled actions, 1,024 paired episodes on the shown layouts.
2. **Day-4 models: Astra and Sonnet 5.5, k = 5 each, plus Haiku as a cheap control if the prompt is under 100k tokens.** Opus 5.5 joins on Day 5 if the gate is interesting. Effort `high` for the fix (that is what Exp 4 used). `xhigh` was Exp 5, the from-scratch task, and is a different condition.
3. **Observation:** 600 interleaved channels, all tanh, 32-wide, unless the quiet-unit check fails, in which case widen to 64 and retrain before paying for edits. No derived features in the primary prompt.
4. **Failure:** dock pocket plus a disclosed distance-to-dock shaping term, and the competence check must pass first.
5. **Parser:** full grammar, raw reply saved, bad lines skipped and logged, last block wins, transpose documented. No JSON schema. No `eval`.
6. **No hard 10% reject.** Report the fraction. Optional pre-registered cap on units taken over.
7. **Controls (e1), (e2), (b′), (c), (d), (f)** as in §7, plus the float-versus-rounded comparison.
8. **Preregister** before the first paid call. Prompt edits after that are a new condition.
9. **Renderer:** ovrtx spike on Day 7, then render. Isaac Sim means current stable, separate venv. Driver target 595.97 unless that day's requirements page says otherwise. Rotate, then translate.
10. **Negative results are part of the post.** Every attempt, including parser skips and the no-weights control.

---

## 10. Sources
Accessed 2026-11-04 unless noted.

- [S1] Takehi, Teoh, Zhang, Khattab, "Sparks of Weight-Space Programming", MIT CSAIL, 2026-11-02. https://rikiyat.github.io/blog/weight-space-programming/
- [S2] Exp 4 README. https://github.com/RikiyaT/lm_understand-logs/blob/main/exp4_fix_car/README.md
- [S3] Exp 4 prompt, 577,774 bytes. https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp4_fix_car/runs/oneshot_repair/reply/prompt.txt
- [S4] Exp 4 reply. https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp4_fix_car/runs/oneshot_repair/reply/answer.md
- [S5] Exp 4 `run.json` (tokens, effort, context window, wall clock). https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp4_fix_car/runs/oneshot_repair/reply/run.json
- [S6] Exp 5 README (20,486 parameters, interleaved index, `xhigh`, parser re-read). https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp5_write_from_scratch/README.md
- [S7] Exp 1 README (3 decimals, ~73,700 input tokens, interleaved 5-channel index). https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp1_which_game/README.md
- [S8] GPT-6 list prices as repeated by independent pricing writeups on 2026-11-03/04 (Astra $10/$50, Sol $2/$10, Luna $0.10/$0.50, 272k short-context tier, 1.05M context claimed). Official page not opened in this pass: https://developers.openai.com/api/docs/pricing
- [S9] Claude list prices (Opus 5.5 $4/$20 cache read $0.20, Sonnet 5.5 $2/$10 cache read $0.10, Haiku 5.5 $0.10/$0.50 up to 100k and 5× above). Cross-checked across several pricing writeups dated late October and early November 2026. Re-open https://docs.anthropic.com/en/docs/about-claude/pricing before publishing a dollar figure.
- [S10] Claude effort. https://platform.claude.com/docs/en/build-with-claude/effort (Opus 5.5 and Haiku 5.5 default `medium`; Sonnet 5.5 defaults `high`; Opus 5.5 thinking cannot be disabled).
- [S11] API version `2026-10-21` removes `temperature`, `top_p`, and `top_k` (400 even at the default). Reported from Anthropic's migration notes by several independent writeups, 2026-10-21 to 2026-10-27. I did not open the official version page itself.
- [S12] SB3 2.9.2, 2026-10-26. https://github.com/DLR-RM/stable-baselines3/releases/tag/v2.9.2 · policy nets: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
- [S13] Grocery Bot, How to play. https://grocerybot.no/how-to-play
- [S14] ovrtx Python getting started. https://nvidia-omniverse.github.io/ovrtx/python_api/getting_started.html
- [S15] ovrtx Loading USD (two clocks, `update_from_usd_time_async`). https://nvidia-omniverse.github.io/ovrtx/scene/loading_usd.html
- [S16] ovrtx driver requirements, including KB5074109. https://nvidia-omniverse.github.io/ovrtx/driver_requirements.html
- [S17] PyTorch 2.12 release, 2026-05-13. cu128 deprecated, cu130 default, cu132 experimental, Windows driver ≥580.88 for Blackwell. https://pytorch.org/blog/pytorch-2-12-release-blog/
- [S18] Isaac Sim releases and compatibility. 6.0.0 GA June 2026, 6.0.1, then 6.1.0 and 6.2.0. https://docs.isaacsim.omniverse.nvidia.com/latest/installation/compatibility.html · https://github.com/isaac-sim/IsaacSim/releases
- [S19] `nvidia/simready-assets` dataset card via the Hub API: `cardData.license = cc-by-4.0`, tag `license:cc-by-4.0`. Page shows 2,677 assets and 490 GB, and the asset `rack_large_empty_a02`. https://huggingface.co/datasets/nvidia/simready-assets

**Still [UNVERIFIED] after this pass:**

- Official OpenAI pricing HTML (prices above are from agreeing secondary writeups).
- Official Anthropic pricing HTML and the official `2026-10-21` version page (effort behaviour was checked on the official effort doc; the sampling-parameter 400 was checked through secondary writeups).
- Whether ovrtx animates `xformOp` time samples and renders SimReady MDL.
- The exact AMR / forklift package name, if one exists.
- Real token count of our prompt on each tokenizer.
- Whether driver 595.97 is the one Isaac Sim 6.2's own requirements page lists on the day we install it.
