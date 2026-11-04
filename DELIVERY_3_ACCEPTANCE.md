# Delivery 3: PPO training, the induced dock-pocket failure, the gates, and the failure card
*Defines scope, locked rules, and acceptance criteria for the third implementation delivery. Parent plan: `FINAL_PLAN.md` (§6 training, failure induction and pre-call checks; §15 Day 3, Gate A). Previous deliveries: `DELIVERY_1_ACCEPTANCE.md`, `DELIVERY_2_ACCEPTANCE.md`. Date: 2026-11-04. Status: **draft for review**.*

**One-line goal:** a trained 600→32→32→7 tanh policy (on 2 or more seeds) that delivers normally on `open`, `probe` and wall-detour layouts but almost never on `T0`. It passes the capacity and rounding checks and comes with the failure card the model will read. Still no LLM calls.

**Decisions taken in planning (2026-11-04):**
- **Run on Windows natively.** The BIOS is updated (ASUS 1840, microcode 0x133). Afterwards, the hot tests still crashed in **20 of 40** WSL runs but **0 of 60** native Windows runs (P-cores, E-cores and unpinned, 20 each). So training, gates and every reported number run natively on Windows (`.venv-win`). WSL stays usable for editing and quick tests, but its results are not trusted. Acceptance item A0 checks stability on Windows.
- **Strict failure bar,** like Exp 4: `T0` success ≤ 2%.
- **The failure card** is built in this delivery.
- **Tuning budget:** up to about 10 attempts or 8 hours of compute, then stop and report.
- **PyTorch is CPU-only.** The network is tiny; the machine has 32 threads.

---

## 1. Scope
In: CPU PyTorch; training env wrapper (family mix + disclosed shaping); single-file CleanRL-style PPO; torch→npz export; checkpointing + resume; a new **`detour`** layout family; a new **`select`** seed role; gates module + CLI; stats helpers (paired bootstrap, Clopper–Pearson); failure card; demo CLI plays a policy; `TRAINING_LOG.md`; committed artifacts for the chosen checkpoints.
Out: prompt assembly, `count_tokens`, `prereg.md`, any paid call, control (d) "more PPO" (all Delivery 4+).

## 2. Locked rules
**Torch:** CPU wheel via a uv index (`download.pytorch.org/whl/cpu`, `explicit = true`, `[tool.uv.sources]`), works on WSL and Windows. D1/D2 suites unchanged.

**New seed role** `select` = `3,000,000 … 3,009,999`, for checkpoint selection and gates. So:
- train 0–999,999 (training episodes)
- diagnosis 1,000,000–1,000,999 (what the model sees: failure card)
- eval 2,000,000–2,009,999 (headline, Delivery 4 only; **never used in D3**)
- select 3,000,000–3,009,999 (gates)

**New `detour` family:** the straight (Manhattan-descent) path from the spawn to the dock is blocked by **one straight rack wall** (all its racks in one row or column, length 4–6), and the BFS path must step away from the dock. Unlike `trap`, there is no U around the dock. It answers FINAL_PLAN §6.3's "route around walls of the same shape elsewhere": a policy that detours here but not on `T0` has the ability, hidden by the pocket.

**Shaping (FINAL_PLAN §6.2), training only:** `r_train = r + β·(d_prev − d)`, with `d` = Manhattan distance from robot to dock, applied on steps where the robot carries ≥ 1 item. Never BFS distance. The unshaped reward stays in `info`; every reported score is unshaped.

**Training env:** each episode draws a family from a configured mix (e.g. open / detour / trap / T0 weights) and a layout + episode seed from the train range, from a per-run RNG. Gymnasium `SyncVectorEnv`, 32–64 envs.

**PPO:** single file, CleanRL defaults unless a config says otherwise (lr 2.5e-4 annealed, 128 steps/env, γ 0.99, λ 0.95, 4 minibatches, 4 epochs, clip 0.2, ent 0.01, vf 0.5, grad-norm 0.5, orthogonal init). Actor 600→32→32→7 tanh; **separate** critic (600→64→64→1 tanh). Assert no shared parameters at save. `torch.set_num_threads(1)` per run; 3 seeds run as parallel processes.

**Config:** one TOML per attempt (`configs/attemptNN.toml`): β, family mix, total steps (5–50M), envs, hyperparameters, seeds. The config hash goes into every artifact.

**Checkpoints:** every 250k steps: `actor.npz` (D2 format, `W = weight.T`), `full.pt` (actor + critic + optimizer + RNG states, for resume), and one `metrics.jsonl` row with 128 select-seed episodes per family (sampled): success, mean score, loop index, idle fraction. Runs live in `runs/train/<config>-s<seed>/` (gitignored).

**Checkpoint choice (pre-registered):** the final checkpoint of the run; if it fails a gate, the latest checkpoint that passes all gates. All selection uses `select` seeds only.

**Gates are measured on the rounded network** (`serialize.rounded`): that is the network the model edits and baseline (a) evaluates. 1,024 paired select-seed episodes, sampled actions unless noted. Success = at least one needed item delivered.

**Failure card (FINAL_PLAN §6.5)**, on 256 diagnosis seeds, rounded network, sampled:
- T0 aggregate: deliveries, orders, score, loop index, idle/bump fraction, full-but-not-delivering
- the same on `open`
- one full 200-step trace on `T0`, diagnosis seed 1,000,000, **one line per step**: step, action, robot cell, carried SKUs, score (the layout itself is drawn once in the prompt's rules section, in D4)
- **excluded:** action probabilities, heatmaps, probes, the capacity ranking or any unit list

## 3. Deliverables
```
pyproject.toml / uv.lock      torch (CPU index), train dependency
src/weightshift/env/          generate.py + seeds.py: detour family, select role
src/weightshift/train/        envs.py (mix + ShapingWrapper), ppo.py, export.py, config.py
src/weightshift/gates.py      failure, competence, capacity, rounding, second-seed gates -> report
src/weightshift/stats.py      paired bootstrap CI, Clopper–Pearson
src/weightshift/diagnose/     failure_card.py
src/weightshift/cli/          train.py, gates.py, card.py; demo.py gains --actor/--mode
configs/attemptNN.toml        one per attempt
TRAINING_LOG.md               every attempt, including failures
artifacts/d3/<sha12>/         committed: actor.npz, config, gate_report.{json,md}, failure_card.md (primary + second seed)
tests/                        test_train_*.py, test_gates.py, test_failure_card.py, detour layout tests
```
Reuse: `evaluate` + policy spec (`evaluation.py`), `PolicyPlayer`, `rounded`, `zero_units`, `rank_units_by_outgoing_l2`, `actor_sha256`, `compute_metrics`, `seeds_for`, `generate_layout`, D1's layout checks (`monotone_path_exists`, `shortest_path` in `env/layout.py`).

## 4. Acceptance criteria
- **A. Stability and setup.** A0: on **native Windows**, the full suite passes **5 runs in a row**; record BIOS 1840 / microcode 0x133 and the WSL-vs-Windows crash counts in the README. Training, gate and card commands are documented and tested for Windows (PowerShell, `UV_PROJECT_ENVIRONMENT=.venv-win`; multiprocessing uses spawn). A1: `uv sync` installs CPU torch; `uv run pytest` green; D1/D2 tests unchanged. A2: README gives train / gates / card commands.
- **B. Env additions.** B1 `detour`: over 10,000 layouts: playable (D1 E1 checks), straight path blocked, BFS steps away, the wall is one straight segment, no U around the dock. B2 the oracle completes ≥ 3 orders in ≥ 95% of `detour` episodes; `greedy_straight` delivers in ≤ 10% (so the family really needs a detour). B3 `select` role: disjoint from the other three, enforced like C3 of D1.
- **C. Training code.** C1 actor/critic share no parameters (asserted at save; a test that sharing raises). C2 **transpose test:** torch actor logits equal numpy `forward` on the exported npz for 1,000 random observations (≤ 1e-5). C3 shaping: hand-built steps give exactly β·Δd while carrying, 0 otherwise; Manhattan, not BFS (a test layout where they differ); unshaped reward unchanged. C4 determinism: two 20k-step runs with the same config and seed give the same actor SHA-256. C5 resume: stopping at a checkpoint and resuming gives the same final actor as an uninterrupted run. C6 throughput ≥ 15,000 env steps/s per training process, in the README. C7 smoke: 300k steps on `open` beats `random`'s success rate (12.4%) by a clear margin, in < 60 s.
- **D. Procedure.** D1 every attempt is a row in `TRAINING_LOG.md` (config hash, seeds, wall time, gate results, what changed and why), failures included. D2 3 seeds per attempt. D3 knobs that may change between attempts: β, family mix, total steps, entropy coefficient, trap geometry (a geometry change is a D1 rule change: update the rules there first). D4 stop and report after ~10 attempts or ~8 h of compute without a pass.
- **E. Gates** on the chosen checkpoint (rounded network, 1,024 select-seed episodes):
  - E1 failure: `T0` success ≤ 2% sampled, and ≤ 2% greedy.
  - E2 it is the pocket: ≥ 80% of `T0` episodes pick ≥ 1 needed item; `T0` loop index and full-but-not-delivering each ≥ 3× the policy's own `open` values.
  - E3 competence: `open` success ≥ 95% and mean score ≥ 40% of the oracle's (72.25 → ≥ 28.9); `probe` success ≥ 90%; `detour` success ≥ 80%.
  - E4 capacity (§6.4): zero the 4 lowest outgoing-L2 units in each hidden layer → `open` mean score drops < 2% relative (paired seeds). If it fails: widen to 64 (a rule change, logged).
  - E5 rounding (control a): the float network also meets E1, and the paired bootstrap 95% CI of (rounded − float) `open` score contains 0. If not: stop and raise it with you (it would change D2's 3-digit rule).
  - E6 second seed: ≥ 2 of the 3 seeds of the chosen config pass E1–E5 (control (f) needs one).
  - Also reported, not gated: generated `trap` success, greedy versions of all metrics.
  - E7 `uv run pytest -m gates` recomputes every committed gate report bit-identically (also a stability check).
- **F. Failure card.** F1 contents exactly as §2, from 256 diagnosis seeds. F2 regenerates byte-identically. F3 a test that the card contains no probabilities, unit indices or capacity ranking. F4 total size reported in characters (token count is D4).
- **G. Artifacts.** G1 `artifacts/d3/<sha12>/` committed for the primary and the second-seed checkpoint (~170 KB each), with config and reports. G2 README: training time, throughput, the gate table, the chosen attempt.
- **H. Manual (you).** H1 watch the chosen policy with `demo --actor … --layout T0` and on `open`/`detour`; confirm it fetches items and then sits stuck by the pocket, and plays normally elsewhere. H2 read `TRAINING_LOG.md` and the failure card.

## 5. DoD
A–G pass, including `pytest -m gates`; you sign off H1–H2. Then Delivery 4 (prompt, `count_tokens`, `prereg.md`, probe call, Gate B) starts.

## 6. Dependencies
- Needs from D2: actor format, `rounded`, `zero_units`, ranking, policy specs in `evaluate`, `actor_sha256`.
- Gives D4: the primary and second-seed actors, the failure card, the gate report (competence numbers for the card), `stats.py`, and the shaping β / training steps the prompt must disclose.

## 7. Risks
PPO learns the detour anyway (raise β, more T0 weight) or learns nothing (lower β, more `open`) — the tuning budget covers this. Competence fails because the policy never learned detours (FINAL_PLAN §6.2's warning) — the `detour` family in the mix is the counter. Capacity fails at width 32 — widen to 64 (prompt grows to ~43k params). The capacity gate's 2% may be within sampling noise — paired seeds reduce this; if it fails by noise, report the CI and raise it rather than re-running until it passes. Hardware: A0.

## 8. Implementation notes (filled in during implementation)
