# Delivery 1: a playable, tested WeightShift environment
*Defines scope, locked rules, and acceptance criteria for the first implementation delivery. Parent plan: `FINAL_PLAN.md` (§4 game, §5 observation, §6.5 failure card, §15 Days 1–2). Date: 2026-11-04.*

**One-line goal:** by the end of this delivery we can play the game by hand, watch scripted players play it, and trust the rules, layouts, observations and metrics, **with no neural network involved**.

---

## 1. Scope

### In
1. Project skeleton: Python 3.12 venv managed by `uv`, `pyproject.toml`, a lockfile, `pytest`.
2. The environment: rules, orders, scoring, 200-step episodes, seeded layouts in three families (`open`, `trap`, `probe`), plus the fixed headline layout `T0`.
3. The 600-number observation (6 channels, interleaved index, gauge channel).
4. A Gymnasium-compatible wrapper.
5. ASCII renderer and a keyboard play CLI.
6. Trajectory recorder (JSONL) and replayer.
7. Three scripted players: `random`, `greedy_straight`, `oracle` (BFS planner).
8. Metrics: deliveries, orders, score, loop index, idle fraction, full-but-not-delivering fraction.
9. The test suite and a speed benchmark.

### Out (later deliveries)
- PPO and any PyTorch code (Delivery 3). `torch` is **not** installed in this delivery.
- Weight serializer, edit parser/applier and the Exp 4 golden test (Delivery 2).
- Any LLM call, `prereg.md`, USD export, rendering.

---

## 2. Locked rules (tests depend on these)
Where the plan left a detail open, it is decided here. Change a rule here first, then in the code.

**Coordinates and actions**
- Grid 10×10, `(row, col)`, **row 0 = top, col 0 = left**.
- Actions: `0` no-op, `1` up, `2` down, `3` left, `4` right, `5` pick, `6` drop.
- Moving into a wall or rack leaves the robot in place and counts as a **bump**.

**Layout elements**
- Outer ring of cells is wall. Racks are obstacle cells, each holding one of **4 SKUs** (A–D).
- **One dock cell**: a walkable floor cell directly adjacent to the outer wall.
- Racks never run out.

**Orders**
- An order is **3–4 items**, each SKU drawn uniformly from A–D, so repeats are possible. Orders are generated from the episode seed.
- When an order completes, the next order appears **in the same step**.
- `remaining[s] = required[s] − delivered[s]` for the current order.

**Pick (action 5)**
- Succeeds if the robot is 4-adjacent (not diagonal) to **any** rack **and** carries fewer than 3 items. The robot takes one item of that rack's SKU.
- If several racks are adjacent, take from the one first in the order up, down, left, right.
- A pick of a SKU the order does not need still succeeds. It occupies a slot and cannot be discarded. (This is the Grocery Bot "wrong pick" pressure.)
- Otherwise the action does nothing.

*Correction to my earlier explanation: I said pick works "only next to a needed rack". The actual rule is above. "Needed" affects scoring and the observation, not whether the pick succeeds.*

**Drop (action 6)**
- Works only when the robot **stands on the dock cell**.
- Delivers every carried item the current order still needs, up to `remaining[s]` per SKU. Items the order doesn't need stay in the inventory.
- +1 per delivered item. If the order completes: **+5**, and the new order appears.

**Episode**
- Exactly **200 steps**, then `truncated = True`. There is no early termination.

**Observation** (float32, shape `(600,)`, values 0 or 1)
- Index `i = (row * 10 + col) * 6 + channel`.

| Channel | Meaning |
|---|---|
| 0 | robot cell |
| 1 | wall or rack cells |
| 2 | rack cells whose SKU has `remaining[s] − carried[s] > 0` ("still needed") |
| 3 | dock cell |
| 4 | the robot's previous cell (all zero on step 0) |
| 5 | gauge, below |

- **Gauge (channel 5):**
  - Row 0, cols 0–2: the first `carried_total` cells are 1 (0–3).
  - Rows 1–4, cols 0–3: row `1+s` shows SKU `s`, with the first `remaining[s]` cells set to 1 (0–4).
  - The gauge sits on top of grid cells in channel 5 only, and the other channels are unaffected.

**Seeds**
- Two seeds per episode: a **layout seed** and an **episode seed** (orders, spawn cell, action-sampling RNG).
- Disjoint ranges, enforced in code:

| Use | Seed range |
|---|---|
| train | `0 … 999,999` |
| diagnosis | `1,000,000 … 1,000,999` |
| eval | `2,000,000 … 2,009,999` |

**Layout families**
- `open`: 2–3 rack segments of length 2–3, separated so that every rack has a free neighbour. All 4 SKUs appear.
- `trap`: the dock sits at the bottom of a U-shaped rack block whose closed side faces the spawn area, so the straight line from spawn to dock hits the closed side. The only route first moves away from the dock.
- `probe`: the `trap` geometry mirrored, rotated, or with its opening facing the spawn, so a direct walk works.
- `T0`: one hand-authored `trap` layout, committed as a fixture file. Spawn and orders still vary with the episode seed.

---

## 3. Deliverables (files)
Package name `weightshift` under `src/`.

```
pyproject.toml, uv.lock, README.md
src/weightshift/env/        rules, layouts, observation, gym wrapper, renderer
src/weightshift/players/    random, greedy_straight, oracle
src/weightshift/metrics.py
src/weightshift/trajectory.py   record + replay (JSONL)
src/weightshift/cli/        play.py (human), demo.py (scripted run)
fixtures/T0.txt
tests/                      unit, property, determinism, snapshot tests
scripts/bench_steps.py
```

---

## 4. Acceptance criteria
Each item is checkable. "Pass" means the stated command or inspection gives the stated result.

### A. Setup
- **A1.** From a clean clone, `uv sync` then `uv run pytest` succeeds on Python 3.12. No `torch` is installed.
- **A2.** The lockfile is committed and `README.md` states the three commands: install, test, play.

### B. Rules
- **B1.** One hand-built test per rule in §2: walls and racks block movement, pick adjacency (including the diagonal case), the 3-item cap, a wrong pick taking a slot, drop only on the dock, partial delivery leaving unneeded items, +1 per item, +5 on completion, the next order appearing in the same step, truncation at exactly step 200.
- **B2.** Line coverage of `src/weightshift/env/` is at least **90%** (`pytest --cov`).
- **B3.** An invariants test runs **1,000 random-action episodes**, asserting every step:
  - the robot is never inside a wall or rack
  - the carried count is between 0 and 3
  - score never decreases
  - observations are shape `(600,)`, float32, and only 0 or 1

### C. Determinism and seeds
- **C1.** For **1,000 seeds**, the same layout seed, episode seed and action list produce a **byte-identical** trajectory when run twice.
- **C2.** Different episode seeds on the same layout give different orders or spawns for at least 99% of 1,000 seed pairs.
- **C3.** Requesting a seed outside its declared range for its role raises an error, and a test enumerates that the three ranges are disjoint.

### D. Observation
- **D1.** A decoder turns a 600-vector back into a drawable grid. For **1,000 random states**, `decode(encode(state))` equals the original grid, robot, dock, trail, carried count and `remaining`.
- **D2.** Hand-built states check each channel cell by cell, including the interleaved index formula, and the gauge at 0, 1, 2, 3 carried and at 0 and 4 remaining.
- **D3.** Channel 2 turns off for a rack whose SKU has become fully covered by `delivered + carried`.

### E. Layouts (run over **10,000 generated layouts per family**, 0 failures allowed)
- **E1.** All families: the dock and all 4 SKUs are reachable from every spawn candidate. Every rack has at least one free 4-neighbour. Exactly one dock.
- **E2.** `trap` and `T0`: the straight-line (Manhattan-descent) path from the spawn to the dock is blocked, **and** a BFS path exists, **and** its first steps increase distance to the dock.
- **E3.** `probe`: the Manhattan-descent path from the spawn to the dock is **not** blocked.
- **E4.** `T0` is loaded from `fixtures/T0.txt`, and a snapshot test pins its ASCII rendering.

### F. Scripted players (the calibration of the game itself)
Measured over **1,000 episodes** per condition using diagnosis or eval seeds.
- **F1. `oracle`** completes **at least 3 orders in at least 95% of `open` episodes**, and also on `T0` and `probe`.
- **F2. `random`** has mean score below **10%** of the oracle's mean on `open`, never crashes, and never violates B3.
- **F3. `greedy_straight`** delivers **zero items in at least 99% of `T0` episodes**, while the oracle succeeds on those same episodes.
- **F4. `greedy_straight`** delivers at least one item in at least **90%** of `probe` episodes. If it doesn't, the `probe` geometry is wrong, and F3 would be measuring generic blockage instead of the pocket.
- **F5.** If F1 fails, adjust **the game** (step limit, layout density, order size), record the change in this document, and re-run. Do not weaken the criterion silently.

### G. Metrics (§6.5 of the plan)
- **G1.** Functions for deliveries, orders completed and unshaped score, tested on hand-built trajectories with known answers.
- **G2.** **Loop index** is the fraction of steps whose (cell, carried multiset) state already occurred in the previous **20 steps**. **Idle fraction** is the fraction of no-op or bump steps. **Full-but-not-delivering** is the fraction of steps with 3 items carried, at least one currently deliverable, and no delivery in the last 20 steps. Each is tested on a hand-built trajectory with a known answer.
- **G3.** Over 1,000 episodes, the loop index and idle fraction of `greedy_straight` on `T0` are each at least **3× higher** than the oracle's on `open`. This shows the metrics can see the failure we plan to induce.

### H. Gymnasium wrapper and speed
- **H1.** `gymnasium.utils.env_checker.check_env` passes.
- **H2.** `scripts/bench_steps.py` reports at least **10,000 environment steps per second on one core**, including observation building, for random actions. The measured figure is recorded in the README. If it is slower, vectorize the observation build before moving on.

### I. Human play, rendering and replay
- **I1.** `uv run python -m weightshift.cli.play --layout T0 --seed N` runs an interactive episode: the ASCII grid, a legend, the current order, the carried items and the score. Keys: arrows or WASD, `p` pick, `d` drop, `.` no-op, `q` quit.
- **I2.** A recorded episode (JSONL: header with layout seed, episode seed, layout ASCII and code version, then per-step action, robot cell, carried, score delta) **replays to the identical final score and trajectory** (test).
- **I3.** `uv run python -m weightshift.cli.demo --player oracle --layout T0` prints an ASCII movie of a whole episode and the final metrics.

### J. Manual acceptance (you)
- **J1.** You play **one full `T0` episode by hand** and confirm the rules match your mental model, including the trap.
- **J2.** You watch the `oracle` and `greedy_straight` demos on `T0` and confirm the greedy player visibly gets stuck.
- **J3.** You confirm the "Locked rules" section reads correctly, or send changes.

---

## 5. Definition of done
All of A–I pass in one `uv run pytest` run, the benchmark and the calibration numbers from F and G are written into `README.md`, and you have signed off J1–J3. Delivery 2 (the serializer, parser and golden test) starts only after that.

## 6. Risks and open items
- **F1 may fail on the first try.** Order size, 200 steps and a 10×10 grid might leave too little room for orders. F5 says to adjust the game, not the test.
- **The trap may be too easy or too hard for the oracle.** E2 and F3 check both ends, and the Day-3 competence check later uses the `probe` family.
- **The gauge encoding** is my proposal. It only needs to be readable in the prompt and stable. Flag any change now, because it feeds the index map in the Day-3 prompt.
- **Order generation** with repeats can make an order needing 4 of the same SKU. That is allowed, and it is why the gauge shows up to 4 per SKU.

---

## 7. Implementation notes and clarifications (added during implementation)
Recorded here so reviewers can see where the code had to interpret this document.

1. **`greedy_straight` is defined as "competent at fetching, straight at the dock".** It plans its way to racks with BFS, like the oracle, but every step towards the dock must reduce Manhattan distance, and it no-ops when all such steps are blocked.
   - *Why:* the first version also walked straight at racks. On `probe` it delivered in **89.9%** of episodes, just under F4's 90%. Every failure was in the picking phase, stuck behind a rack, never on the way to the dock (E3 already proves the straight line to the dock is open on every probe layout). That measured generic blockage, which is exactly what F4 is meant to rule out.
   - This definition also matches the failure we want PPO to show: a policy that can fetch items but drives straight at the dock (FINAL_PLAN §6.2).
   - With it: `T0` 0% deliveries (F3), `probe` 100% (F4).
   - The thresholds are unchanged.
2. **E2 "its first steps increase distance to the dock"** is implemented as: no strictly-distance-reducing path exists from any spawn to the dock, **and** the BFS shortest path contains at least one step that increases Manhattan distance to the dock.
