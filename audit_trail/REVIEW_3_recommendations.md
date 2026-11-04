# WeightShift (Loop Doctor): Review #3, fact-check and further improvements
*Date: 2026-11-04 · Reviews: `ORIGINAL_PLAN.md` (v1) and `weightshift_plan_review_recommendations.md` (Review #2) · Audience: the builder agent in `C:\Projects\QuickWins\LoopDoctor`*

> This review checks both earlier documents against the primary sources. The main one is the MIT post's **companion logs repo**, which contains the actual prompts used. The review then adds changes neither document proposed. Each claim below cites a source in §10. Anything marked **[UNVERIFIED]** still needs checking.

---

## 1. Summary verdict
- **v1 is a sound research plan.** Its strengths are the Day-4 go/no-go gate, the required baselines, publishing every attempt including negatives, and files as the only interface between sim and render. Keep all of that.
- **Review #2 found one real issue but drew the wrong conclusion.** It is correct that the 10×10 grid networks in Exp 1 were only read, never edited. But several of its "corrections" are themselves wrong when checked against the logs. The most important is the serialization format, which it describes incorrectly.
- **Three things neither document caught:**
  1. **The real Exp 4 format is quantized integers × a per-tensor scale.** Edits go out in a small *edit language* with slice assignment. This cuts input tokens about 3× compared with v1's estimate, and it shows a JSON diff is the wrong choice.
  2. **ovrtx + ovstage** are pip-installable Omniverse RTX rendering SDKs, and they should be the primary renderer instead of a full Isaac Sim install.
  3. **The induced failure should be a "move away from the goal first" trap,** the exact structural analogue of the Kinetix car having to back up first. It should come with a **competence check**, because the blog states that editing can't add an ability the policy lacks.

---

## 2. Fact-check of Review #2

| Review #2 claim | Verdict | Evidence |
|---|---|---|
| The Exp 1 grid networks (500→32→32→5) were only read, never edited | ✅ Correct | Blog, Exp 1 [S1] |
| Raw 10×10 grid inputs make surgery hard, so factored scalars are needed | ⚠️ Overstated | In Exp 5, Astra wrote a working **600→32→32→6 grid network from scratch** with no tools. It scored 517 on Breakout vs PPO-500M's 25.3, and its layer 1 computes where the ball lands. Spatial grid inputs are clearly not a blocker [S1]. |
| Kinetix inputs were dedicated "car x-position / x-velocity" columns | ⚠️ Partly | The 679 inputs are generic **per-body slots**: "Polygon slots … 26 numbers each; circle slots … 18 numbers each", covering position, velocity and shape, plus joints and thrusters [S1][S2c]. The car's state is in there as raw entity state. It is *not* task-derived features like `dx_to_dock` or a stall timer. |
| Astra hijacked "3–4 lightly-used hidden units" | ❓ Unverified | The blog says only "a small separate path through a few units per layer" [S1]. Check `exp4_fix_car/runs/oneshot_repair/reply/answer.md` before repeating the number. |
| SB3 shares the actor/critic backbone by default | ❌ Mostly wrong | In current SB3, `MlpPolicy` builds **separate** `pi`/`vf` MLPs, and the default features extractor for flat observations is `Flatten`, which has no parameters [S7]. CleanRL's `ppo.py` also has separate `actor`/`critic` modules. It's still worth adding an assertion that no actor parameter is shared with the critic, but this is not a design flaw. |
| Mirror the logs' format: "3 decimal places, `W1 row 3: -0.080 0.255 …`" | ❌ Wrong for Exp 4 | Exp 1 did use 3 decimals [S2d]. **Exp 4, the fix experiment we are replicating, did not.** See §3 [S2b][S2c]. |
| `reasoning_effort="high"`, all tools disabled | ✅ Correct | Logs README: effort `high` for Exps 1, 2 and 4. All built-in Codex tools were disabled, and each turn's `run.json` audits `tool_free` (all 1,125 turns passed) [S2a]. |
| Use Pydantic/JSON structured outputs for weight diffs | ⚠️ Counterproductive | Exp 4's output was an edit block with statements such as `actor0.W[6, 122] = 80` and `actor0.W[:, 122:128] = 0` [S2b]. JSON is about 3× more tokens per edit, can't express slices, and moves the model off the in-distribution format. |
| Stock Blender can't render SimReady MDL/OmniPBR materials | ✅ Broadly right | Blender's USD importer maps `UsdPreviewSurface`, not MDL. Expect to re-assign materials. Blender drops to last fallback (§7). |
| Rotate-in-place, then translate, per grid step | ✅ Good | Keep it. It avoids diagonal "skating" during 90° turns. |
| A batched torch env gives >500k steps/s | ❓ Plausible, unmeasured | Fine as an option. Measure before relying on it. A Gymnasium vector env on CPU is enough for 5–50M steps. |

**Also corrected in v1:**
- v1 writes tensors as `row=out-unit`. The Exp 4 logs write **row i = input entry i** (`embed.W (679 x 128)`) [S2c]. Use the logs' orientation: each row then answers "what does input i feed?", which is how you read a channel's footprint.
- v1 says simready-assets has "no licence tag". It has one now (§8).
- v1's driver advice is outdated for Windows 11 (§7).

---

## 3. Serialization and the edit protocol: copy Exp 4 exactly
*(Replaces v1 §8 "Weight text format" and Review #2 Flaw 3.)*

**What the Exp 4 prompt actually contained** [S2b][S2c]:
- A framing paragraph ("Below is a policy network that controls a small car … trained with … PPO for 1,000,000,000 environment steps, and it still never succeeds").
- The scene rules.
- An observation-layout section that names index ranges.
- Weights, with each tensor stored as **integers times one scale**. The scale comes from the tensor's largest entry and keeps 3 significant digits. For example:
  ```
  embed.W (679 x 128), scale 1e-2: one line per row (row i)
  0: 1 -3 -4 -3 -6 3 1 -6 -2 7 2 -1 9 13 -12 1 ...
  1: 0 -2 5 -1 7 3 2 0 -3 -8 -5 0 1 -4 1 6 ...
  ```
- Output instructions: *"Write a code block whose opening line is three backticks followed by the word edits, one statement per line"*, with `#` comments allowed. After the block, *"explain briefly (at most about 200 words) what your edits do."*
- The harness applied the edits **in order to the rounded weights**.
- There was one attempt. Any "second try" or feedback was explicitly not allowed.

**Do the same for WeightShift:**
1. **Weights in.** One header per tensor, e.g. `L1.W (612 x 32), scale 1e-2`, then `i: <ints>` per input row, then `L1.b (32), scale …`, and so on for L2 and L3. Next to the header, add the input-index map: which index ranges are which channel, and that cell (r,c) of channel k is index `offset_k + 10*r + c`. That map replaces v1's per-column `(channel,row,col)` annotations, which cost many tokens.
2. **Edits out.** A fenced `edits` block. The minimal grammar is:
   `T.W[i, j] = v` · `T.W[a:b, j] = v` · `T.W[i, a:b] = v` · `T.W[:, j] = v` · `T.b[j] = v`, plus `#` comments, where `v` is an integer in that tensor's scale. Then ≤200 words of explanation.
3. **Parser.** A strict hand-written grammar (regex or `lark`), **never `eval`**. Reject unknown tensors, out-of-range indices, non-integers, and any edit set over the weight-change budget. Store the raw reply text alongside the parsed edits.
4. **Use the rounded policy as baseline.** Because edits apply to the quantized weights, baseline (a) must be the **quantized, unedited policy**, evaluated next to the float policy, so rounding effects are never credited to the LLM. If quantization alone changes the target metric by more than noise, lower the scale (more digits) before going further.
5. **Day-1 round-trip test.** Run float → quantize → text → parse → network. The action distributions must match the quantized network exactly, and seeded episodes must be identical.

**Token and cost math, revised:**
- The Exp 4 prompt is **577,774 characters for 171,664 weights**, ≈3.4 characters per weight including all the prose [S2c].
- For our **20,903 weights** (612·32+32 + 32·32+32 + 32·7+7), that comes to about **70k characters ≈ 25–40k input tokens**. v1 estimated 90–130k. Measure the real number with `messages.count_tokens`.
- **Output now dominates the cost.** The edits block plus explanation is about 2–8k tokens. But on Opus 5.5 and Sonnet 5.5 at `high` effort, **thinking tokens are billed as output** and may add roughly 10–30k.

| Model | In/Out $/MTok | ≈ $/attempt (35k in, 10–40k out) |
|---|---|---|
| Claude Opus 5.5 | 4 / 20 | ~$0.35–0.95 |
| Claude Sonnet 5.5 | 2 / 10 | ~$0.17–0.47 |
| Claude Haiku 5.5 (cheap control) | 0.10 / 0.50 (≤100K-token prompts) | ~$0.01–0.02 |
| GPT-6 Astra / Sol / Luna | carried from v1, **[UNVERIFIED]** | (recompute with ~35k in) |

- Claude prices come from Anthropic's current model table [S9]. A 35k-token prompt also keeps Haiku 5.5 in its ≤100K price tier.
- Cache reads are $0.20/MTok on Opus 5.5 and Sonnet 5.5. The Batches API gives a further 50% off.
- Kinetix shows a **171k-weight prompt was workable**, so our net is far from any context limit. Don't shrink the net preemptively (v1 §13); keep 32 units.

---

## 4. Observation design: test it as a factor
*(Replaces Review #2 Recommendation 1.)*

Review #2 proposed putting 12 scalars at `obs[0:12]`. Several of them are `steps_since_last_pick_or_drop`, `is_inventory_full`, `adjacent_to_needed_sku`, `dx/dy_to_dock` and `dx/dy_to_nearest_needed_sku`. These are not the kind of raw state Kinetix gave its policy. They are **derived features that already contain the diagnosis and most of the fix.** With them, the "surgery" reduces to "if stall_timer > θ, follow dx/dy_to_dock", a two-unit if-statement. Two consequences follow:
- The headline claim weakens. "The LLM read the policy" becomes "the LLM wired up the feature we gave it".
- PPO is **less likely to fail at all**, which makes inducing a clean failure harder.

**Recommendation: fix two variants up front and label them clearly.**
- **G (primary, MinAtar-like).** 6 grid channels × 10×10 = 600 inputs: robot, racks/obstacles, needed-SKU racks, dock, trail (previous position), and a **gauge** channel encoding carried/needed counts. The gauge channel mirrors Exp 5, whose inputs were player, danger, reward, projectile, trail and gauge [S1]. This is closest to the input style of the blog's grid networks, and Exp 5 shows frontier models can write working circuits over it.
- **G+E ("entity-aided").** G plus about 6 **raw entity scalars**: robot x, y; dock x, y; items carried; fraction of time left. These are Kinetix-like state slots. **No** relative vectors, no stall timers, no "can pick" flags.
- If the budget allows, train both and report which one the LLM could fix. "It needed entity scalars" is a publishable finding.

---

## 5. Failure-mode design: a Kinetix trap with a competence check
*(Replaces Review #2 §4 Option A. Option B stays as the second failure.)*

**Why the "hoarder" (Option A) probably won't trap PPO:** a 10×10 grid gives trips of at most about 18 steps, and 0.94¹⁸ ≈ 0.33, so the +2 dock reward still has a clear positive value. PPO with entropy bonus will usually find it. The +1-per-pick shaping also moves away from the Grocery-Bot-style scoring v1 committed to (+1 per delivered item, +5 per order).

**Proposed main trap: the "dock pocket".** It is the direct analogue of Kinetix, where the car has to drive *away* from the goal first and the greedy policy stalls against the ramp.
- On *trap layouts*, the dock opens onto a corridor that is reached only by going around a U-shaped rack row. From the pick area, the straight-line direction to the dock leads into the closed side of the U.
- A greedy policy pushes into the dead end and oscillates or idles. Getting out means moving away from the dock for a few steps.
- **Target baseline:** about 0% delivery on trap layouts, comparable to Exp 4's 0/256, with normal performance on non-trap layouts. It is an instantly readable shot for the video: the robot pushing into a dead end.
- How to get it: train mostly on open layouts with a small fraction of trap layouts. Alternatively, stop at a checkpoint before PPO has solved the trap. Either way, document it as v1 §6 requires.

**Competence check (a hard prerequisite before Day 4).** The blog's failed Gemma experiment ends: *"Editing weights could remove an obstacle that hid an ability, but it could not add an ability the model lacked"* [S1]. So, before calling any LLM:
- Show that the policy **can** route around walls of the same shape when the pocket is mirrored, rotated, or opens toward the robot.
- Show that it can deliver when the dock is reachable directly.
- If it can't route around obstacles anywhere, the problem is a missing ability, not a hidden one. Change the trap or the checkpoint; don't spend LLM budget.
- Log the competence metrics in the failure card too, because they are legitimate diagnostic context.

**Option B (symmetric aisle oscillation)** from Review #2 is a good second failure. It suits the "trail" channel well: a hysteresis path from the trail channel is a natural fix.

---

## 6. Protocol additions

**Primary condition = Exp 4, exactly.** One tool-free reply, no feedback, no second try [S2b].
- Run **k ≥ 5 independent first attempts** per model, and report the **first-attempt success rate** (with a CI) as the headline. Disclose any best-of-N separately.
- Use **1,024 evaluation episodes** per condition, the number the blog used, with paired seeds as v1 specifies. Optionally add sticky actions with p = 0.1, as Exp 5 did.

**Claude API facts that change v1's guardrails** [S9]:
- `temperature`, `top_p` and `top_k` **return a 400** on API version 2026-10-21, so v1's "fix the temperature" can't be done on Claude. Fix and log **effort** instead, along with the API version.
- On Opus 5.5, thinking **can't be disabled**, and its effort **defaults to `medium`**. Set `output_config.effort = "high"` explicitly to match Exp 4. Sonnet 5.5's default is already `high`.
- Opus 5.5 and Sonnet 5.5 both have **1M context and 128K max output**. Large requests must stream.
- Log for each attempt: model ID, effort, API version, `usage`, including cache reads and output tokens, which include thinking.

**New controls.** These go beyond v1's (a)–(d).
- **(b′) A matched random edit.** v1's (b) perturbs "the same weights at the same magnitude". Make it stricter: the same *number* of changed entries, in the *same tensors*, on the *same count of units*, with per-tensor magnitudes drawn from the LLM edit's own distribution. Use 20 draws.
- **(e) A no-weights control.** This is the most informative new control.
  - Give the model the same env spec and failure card, but **hide the weights**.
  - Tell it that units {28–31} in L1 and L2 are free (we zero them beforehand and verify the policy is unchanged), and ask it to write a bypass circuit in the same edit language.
  - Exp 5 shows frontier models can write circuits from scratch. So if (e) works as well as the main condition, the result is "LLM writes a patch", **not** "LLM read the weights". That distinction is the scientific core of the post.
- **(f) A transplant control.** Apply the LLM's edit unchanged to a PPO policy from a **different training seed** with the same failure. If the fix carries over, the edit is a generic circuit; if it breaks, it depended on reading *these* weights. Either outcome is interesting.

**Pre-registration.** Before the first LLM call on Day 4, write the following to `prereg.md` and log its SHA-256:
- the hypotheses and the target failure metric
- k
- the non-regression threshold
- the edit budget
- the controls
- the decision rule

---

## 7. Rendering: use ovrtx + ovstage first
*(Replaces v1 §10 "Renderer options" order and Review #2 Flaw 4 fix 2.)*

**What these are** [S3][S4]:
- **ovrtx** is NVIDIA's lightweight **C and Python SDK for Omniverse RTX**. Its render modes include real-time path tracing, full path tracing and spectral rendering, and scenes are described in OpenUSD.
- **ovstage** is a vectorized, GPU-native **runtime stage** for USD data, which holds transforms, attributes and the hierarchy. Since ovrtx 0.4, ovrtx attaches to an ovstage stage, and the renderer-owned scene APIs are deprecated.
- Both are **pre-release** (ovstage documentation is at v0.3.0) and under the NVIDIA Software License Agreement. That is fine for rendering our own frames; we don't redistribute the SDK.

**Why ovrtx is the better first choice than Isaac Sim 6.0:**
- It's a `pip install` into a normal venv, with no Kit app and no multi-GB Isaac download.
- It's fully scripted and headless: frames come back as numpy arrays.
- It uses the same RTX renderer, so it should handle SimReady's MDL/OmniPBR materials, which Blender can't. **[UNVERIFIED]**, checked in the spike below.

**Install and requirements** [S3a][S3b][S3d]:
- `pip install ovrtx ovstage` (or `uv add ovrtx ovstage`) in a **dedicated venv**.
- Python 3.10–3.13. **On Windows, use Python ≥ 3.11.**
- Windows 11 or Ubuntu 22.04/24.04 (x86_64), with VC runtime ≥ 14.38.
- An RTX GPU. The minimum listed is an RTX 3070; Blackwell is supported.
- The first run blocks for 1–2 minutes while shaders compile, and the minimal example downloads its scene over the internet.

**Drivers. This corrects v1 §10/§12** [S3b]:
- Validated GeForce Blackwell drivers on Windows are **581.42** (R570 line) and **595.97** (R595).
- **Windows 11 update KB5074109 (2026-01-13) requires ≥ 582.41 on R580 (or 595.97 on R595) for Vulkan.** On an up-to-date Windows 11, v1's "install 581.42" advice can break Vulkan. **Use 595.97** (in WSL/Linux: 580.95.05 or 595.58.03).
- This may conflict with v1's note that 595.x crashed *Isaac Sim 5.1* on 50-series cards [S8 in v1]. That report predates the validated 595.97, so test it, and keep 591.74 as a known-good fallback for Isaac Sim only.

**The render loop (Python, from the official minimal example and the Loading-USD and Transforms pages)** [S3a][S3c][S4b]:
```python
renderer = ovrtx.Renderer()                      # create first: it loads OpenUSD
stage = ovstage.Stage("weightshift.replay")
renderer.attach_ovstage(stage)
ovstage.population.open_usd(stage, "replay_root.usda", ordinal=1)
stage.advance_write_floor(1, ovstage.Scope.ALL).wait()

for n, t in enumerate(frame_times_seconds, start=2):
    # Option A: let USD time samples drive the animation (time is in SECONDS, not timecodes)
    ovstage.population.update_from_usd_time_async(stage, ordinal=n, time=t).wait()
    stage.advance_write_floor(n, ovstage.Scope.ALL).wait()
    products = renderer.step(render_products={"/Render/Camera"}, delta_time=1/30, ordinal=n)
    # map frame.render_vars["/Render/Camera/LdrColor"] to CPU -> np.from_dlpack -> PIL save PNG
```
- **There are two clocks.** `renderer.step(delta_time)` advances simulation and sensor time but **does not move USD animation**. `update_from_usd_time*` moves the animation but not the sim clock. Drive both [S3c].
- **Option B:** skip the time samples on the render path. Write every robot and tote transform per frame as one batched `(N, 4, 4)` float64 row-major tensor, using `stage.write_attribute(query, "omni:xform", …, semantic=AttributeSemantic.MATRIX)`, then call `compute_hierarchy` and `advance_write_floor`. This drives the replay straight from the trajectory JSON [S4b].
- **The scene file is never edited.** Use an **inline root layer** that sublayers the replay `.usda` and adds the Camera, RenderProduct and RenderVar prims [S3c].
- Then run `ffmpeg -framerate 30 -i frame_%05d.png …` over the frames.

**Recommended split:**
- The **main venv** (`usd-core`) authors the portable `.usda`: asset references, time-sampled `xformOp`s with rotate-then-translate keys, and tote visibility. That file stays the canonical artifact for anyone using Isaac Sim, usdview or Blender.
- The **ovrtx venv** only renders.
- Don't import `usd-core` and `ovrtx` in the same process. ovrtx loads its own OpenUSD, and the docs describe strict load-order rules [S3a].

**Day-7 spike, 1–2 hours, before committing:** render one SimReady rack, one box-proxy robot and a camera for 30 animated frames. Confirm three things:
1. **[UNVERIFIED]** time-sampled `xformOp:translate`/`rotateZ` animate through `update_from_usd_time`. If not, use Option B.
2. SimReady MDL/OmniPBR materials look correct.
3. References and payloads resolve from local paths.

If any of these fail and can't be fixed quickly, fall back.

**Renderer fallback order:**
1. **ovrtx**
2. Isaac Sim 6.0 Movie Capture
3. three.js with glTF plus the trajectory JSON. This doubles as an interactive demo.
4. Blender, last (MDL materials need re-assigning).

---

## 8. Assets and fair-play corrections

**SimReady assets** [S5]:
- The `nvidia/simready-assets` page now shows a **cc-by-4.0** licence tag and lists **2,677 assets totalling ~490 GB**. v1 said there was no tag and about 2,575 assets.
- **Never clone the repo.** Download single packages with `huggingface_hub.snapshot_download(repo_id=..., repo_type="dataset", allow_patterns=["<package>/**"])`.
- Start from `PhysicalAI-SimReady-Warehouse-01` (~15 GB, CSV catalog) as v1 suggests. `rack_large_empty_a02` is a confirmed rack asset in simready-assets.
- Keep `ATTRIBUTION.md` per package. The robot is still **[UNVERIFIED]**: search the CSV for AMR or forklift, otherwise use a box proxy.

**Grocery Bot, from the live rules page** [S6]:
- "Any method and any language are welcome", including AI coding assistants.
- The competition has **no scheduled end date**.
- The page links an open-source game repository, and says **nothing** about publishing strategies.
- So v1's rule against publishing GB-tuned policies is a **self-imposed courtesy**, not a competition rule. Keep it, but say so plainly in the post ("inspired by", no GB name in the title).

Our environment is already clearly different, and v1 §14 should note it:
- 10×10 with one robot and one dock.
- No trucks, truck away times, or three-cell load zones.
- No ID-ordered move resolution.

GB's smallest store, Easy, is 12×10 with 1 robot, **2 trucks**, 4 products and 3–4 items per order.

---

## 9. Revised milestones (changes only)
| Day | Add or change |
|---|---|
| 1–2 | Quantize/serialize/parse **round-trip test**. **Edit-grammar parser** with rejection tests. An assertion that no parameters are shared between actor and critic. |
| 3 | **Trap layouts** and the **competence check** (§5). G and G+E observation variants, if trained. Write and hash **`prereg.md`**. |
| 4 (gate) | k ≥ 5 first attempts each on Opus 5.5 and Sonnet 5.5, plus the Haiku 5.5 control and **control (e)**. Estimated ≲ $15 total (§3). Evaluate on the quantized baseline with 1,024 episodes. |
| 5–6 | Controls (b′), (c), (d), (f). Mechanism ablation (zero the edited units). GPT models, if their prices are verified. |
| 7 | **ovrtx spike** in the morning, then the full render. Fall back per §7 if the spike fails. |
| 8–10 | Unchanged from v1. Add the no-weights and transplant results to the stats card. |

---

## 10. Sources (all accessed 2026-11-04)
- [S1] Takehi, Teoh, Zhang, Khattab, "Sparks of Weight-Space Programming", MIT CSAIL, 2026-11-02. https://rikiyat.github.io/blog/weight-space-programming/
- [S2a] Logs repo README (model `gpt-6-astra`, `codex exec`, effort per experiment, tools disabled, `audit.tool_free`, licences CC BY 4.0 / MIT). https://github.com/RikiyaT/lm_understand-logs
- [S2b] Exp 4 README (integer × scale weights, edit language, edits applied to rounded weights, one attempt). https://github.com/RikiyaT/lm_understand-logs/tree/main/exp4_fix_car
- [S2c] Exp 4 prompt, verbatim (577,774 characters). https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp4_fix_car/runs/oneshot_repair/reply/prompt.txt
- [S2d] Exp 1 README ("exactly as shown to Astra (3 decimals)"). https://github.com/RikiyaT/lm_understand-logs/tree/main/exp1_which_game
- [S3] ovrtx documentation. https://nvidia-omniverse.github.io/ovrtx/
  - [S3a] Python getting started (install, minimal example, load order). https://nvidia-omniverse.github.io/ovrtx/python_api/getting_started.html
  - [S3b] Hardware and driver requirements. https://nvidia-omniverse.github.io/ovrtx/driver_requirements.html
  - [S3c] Loading USD (time-sampled USD, references, inline composition). https://nvidia-omniverse.github.io/ovrtx/scene/loading_usd.html
  - [S3d] Repository (platforms, Windows notes, licence). https://github.com/NVIDIA-Omniverse/ovrtx
- [S4] ovstage documentation (v0.3.0, pre-release). https://nvidia-omniverse.github.io/ovstage/
  - [S4b] Transforms (`omni:xform` batched writes, `compute_hierarchy`). https://nvidia-omniverse.github.io/ovstage/scene/transforms.html
- [S5] `nvidia/simready-assets` dataset page (cc-by-4.0, 2,677 assets, 490 GB). https://huggingface.co/datasets/nvidia/simready-assets
- [S6] Grocery Bot, How to play. https://grocerybot.no/how-to-play
- [S7] Stable-Baselines3 policy networks documentation. https://stable-baselines3.readthedocs.io/en/v2.2.1/guide/custom_policy.html
- [S8] (v1's sources S2–S11 are still valid except where corrected above.)
- [S9] Anthropic model and pricing table (Opus 5.5 $4/$20, Sonnet 5.5 $2/$10, Haiku 5.5 $0.10/$0.50 up to 100K-token prompts, 1M context, 128K output, effort defaults, sampling parameters rejected on API 2026-10-21), from Claude Code's bundled `claude-api` reference, cached 2026-10-06. Re-check https://docs.anthropic.com/en/docs/about-claude/pricing before publishing figures.

**Still [UNVERIFIED] after this review:**
- whether ovrtx animates `xformOp` time samples through `update_from_usd_time`
- how faithfully ovrtx renders SimReady MDL materials
- GPT-6 Astra, Sol and Luna prices and context limits (carried over from v1)
- the exact number of units per layer in Astra's Exp 4 edit (read `answer.md`)
- the real token count of our serialized prompt per tokenizer (measure on Day 1)
- whether driver 595.97 is stable for Isaac Sim 6.0 on a 5090 (only relevant if we fall back)
