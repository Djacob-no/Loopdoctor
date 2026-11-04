# WeightShift

A small warehouse grid game built to test one question: **can a frontier LLM repair a trained policy by editing its weights directly?** The experiment reproduces, at hobby scale, Experiment 4 of MIT CSAIL's *Sparks of Weight-Space Programming*. See [`FINAL_PLAN.md`](FINAL_PLAN.md) for the experiment, and [`DELIVERY_1_ACCEPTANCE.md`](DELIVERY_1_ACCEPTANCE.md) and [`DELIVERY_2_ACCEPTANCE.md`](DELIVERY_2_ACCEPTANCE.md) for each delivery's rules and acceptance criteria. `audit_trail/` holds the earlier plan and reviews.

**Status: Deliveries 1 and 2 are implemented.**
- **Delivery 1:** a playable, tested environment.
- **Delivery 2:** a policy network stored as numpy arrays, playable in the game. It can be written out as text in MIT's Exp 4 format, edited with Exp 4's edit language, and analysed. The pipeline reproduces Exp 4 exactly (see [Delivery 2 results](#delivery-2-results-2026-11-04)).

PPO training (Delivery 3) and LLM calls (Delivery 4) come next. There is still no PyTorch.

## Commands

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

WSL and Windows share this folder, so they do not share a virtual environment. A `uv sync` on one OS replaces the other's `.venv`. WSL uses `.venv`. On Windows, set `UV_PROJECT_ENVIRONMENT=.venv-win` (user environment variable, or `$env:UV_PROJECT_ENVIRONMENT = ".venv-win"` in the current shell) and then run the same commands. `.venv-win/` is gitignored. Leave the variable unset in WSL.

```bash
uv sync                                                    # install (exact versions from uv.lock)
uv run pytest                                              # all acceptance tests (~30 s)
uv run python -m weightshift.cli.play --layout T0          # play by hand
uv run python scripts/fetch_exp4.py                        # download MIT's Exp 4 files for the golden test
```

Without the Exp 4 files, the 8 golden tests are skipped and say so. Delivery 2 counts as done only when they have run.

More:

```bash
uv run python -m weightshift.cli.demo --player oracle --layout T0                  # ASCII movie + metrics
uv run python -m weightshift.cli.demo --player greedy_straight --layout T0 --clear --delay 0.05
uv run python -m weightshift.cli.play --layout trap --seed 42 --record runs/me.jsonl
uv run python scripts/bench_steps.py --family open                                  # steps/s on one core
uv run python scripts/bench_steps.py --policy --min 3000                            # same, with a numpy actor playing
uv run pytest -m golden                                                             # only the Exp 4 golden tests
uv run python scripts/fetch_exp4.py --check                                         # verify the downloaded files' hashes
WEIGHTSHIFT_WORKERS=1 uv run pytest --cov=weightshift --cov-report=term             # coverage (serial, so workers are counted)
```

**Play keys:** arrows or `w a s d` move, `p` pick, `x` drop (X marks the dock), `.` or space no-op, `q` quit.

## The game in brief

- A 10×10 room with racks holding 4 SKUs (A–D) and one dock.
- One robot carries up to 3 items. An order is 3–4 items.
- **Pick** works next to any rack: up, down, left or right, with that priority when several racks are adjacent. A wrong pick takes up a slot and can't be discarded.
- **Drop** works only on the dock, and delivers what the current order still needs.
- Scoring: +1 per delivered item, +5 per completed order, and the next order appears immediately.
- Episodes are exactly 200 steps.

The policy sees **600 numbers**: 6 channels × 10×10, indexed `i = (row*10 + col)*6 + channel`.

| Channel | Meaning |
|---|---|
| 0 | robot |
| 1 | wall or rack |
| 2 | rack whose SKU is still needed |
| 3 | dock |
| 4 | previous robot cell |
| 5 | gauge: carried count bar on row 0, remaining-per-SKU bars on rows 1–4 |

Layout families:

| Family | What it is |
|---|---|
| `open` | normal rooms, where the policy should be competent |
| `trap` | the dock sits below a U of racks whose closed side faces the robot, so you must first move *away* from the dock |
| `probe` | the same U with a door in line with the dock |
| `T0` | the fixed headline trap in `fixtures/T0.txt` |

```
##########
#..ssss..#    s  spawn candidates
#........#
#.ABCDAB.#    closed side of the U
#.A....C.#
#.D....B.#
#.C....A.#
#........#
#...X....#    X  dock
##########
```

## Repository layout

```
src/weightshift/env/       constants, seeds, layout, generate, game, observation, render, gym_env
src/weightshift/players/   base (Player protocol), scripted (random, greedy_straight, oracle)
src/weightshift/metrics.py        loop index, idle fraction, full-but-not-delivering, wrong picks
src/weightshift/trajectory.py     JSONL recording and replay
src/weightshift/evaluation.py     many-seed runner (parallel; WEIGHTSHIFT_WORKERS=1 for serial)
src/weightshift/cli/       play (human), demo (scripted)
src/weightshift/policy/    actor file format and forward pass (actor), PolicyPlayer (player)
src/weightshift/serialize/ quantize, dump (weights as text + reader), edits (parser), apply, analysis (+ ablation)
fixtures/T0.txt            the headline layout
fixtures/exp4/             MIT's Exp 4 files, downloaded by scripts/fetch_exp4.py (gitignored)
scripts/bench_steps.py     throughput benchmark
scripts/fetch_exp4.py      golden-test download, pinned commit + SHA-256
tests/                     one file per acceptance area
```

## Measured results (2026-11-04)

### Calibration (acceptance F, G3)
1,000 episodes per row on eval-range seeds, from `tests/test_calibration.py` and the same runner.

| Player | Layouts | Mean score | Mean orders | Episodes with 3+ orders | Episodes with any delivery | Loop index | Idle fraction |
|---|---|---|---|---|---|---|---|
| oracle | open | 72.25 | 8.42 | 100% | 100% | 0.156 | 0.000 |
| oracle | T0 | 81.07 | 9.47 | 100% | 100% | 0.250 | 0.000 |
| oracle | probe | 75.19 | 8.76 | 100% | 100% | 0.157 | 0.000 |
| oracle | trap | 74.25 | 8.66 | 100% | 100% | 0.155 | 0.000 |
| random | open | 0.19 | 0.00 | 0% | 12.4% | 0.749 | 0.257 |
| greedy_straight | **T0** | **0.00** | 0.00 | 0% | **0.0%** | **0.965** | **0.965** |
| greedy_straight | **probe** | 75.19 | 8.76 | 100% | **100%** | 0.157 | 0.000 |
| greedy_straight | trap | 2.47 | 0.29 | — | 4.4% | 0.934 | 0.929 |
| greedy_straight | open | 56.17 | 6.55 | — | 83.7% | 0.324 | 0.200 |

**Reading the table:**
- **The trap works.** A player that fetches competently but drives straight at the dock never delivers on `T0`. It delivers on every `probe` layout, which is the same geometry with a door. So the pocket is the only thing stopping it.
- **Some generated traps leak.** On 4.4% of generated `trap` episodes, an item picked on the outside of a leg near the pocket's opening leaves a straight line to the dock. `T0` has no such case.
- **The oracle's loop index isn't zero (0.156)** because it legitimately walks back along corridors it used. The stuck greedy player's 0.965 is still 6× higher, comfortably above G3's 3× threshold.

### Speed (acceptance H2)
`scripts/bench_steps.py`, one pinned core, random actions, including observation building:

| Layouts | steps/s |
|---|---|
| open | 138,800 |
| trap | 154,300 |
| T0 | 204,200 |

The target was 10,000.

### Coverage (acceptance B2)
`src/weightshift/env/` has 97% line coverage (708 statements, 18 missed). The whole package is at 96%.

## Delivery 2 results (2026-11-04)

### The Exp 4 golden test (acceptance F)
We used MIT's published network and Astra's published reply:

| Check | Result |
|---|---|
| Our text dump of their network vs the weight section of their `prompt.txt` (567,049 characters) | **identical, byte for byte** |
| Astra's `answer.md` parsed with our parser | **51 statements, 0 skipped** |
| Their 51 edits applied by our applier vs their `edited_actor.npz` | **identical** (every float64 value equal) |
| Weights changed | **6,073 of 171,664 (3.54%)**, every per-tensor count matches `oneshot_result.json` |
| Units taken over per layer | embed 3, actor0 6, actor1 4, actor2–4 2 each, outputs 0–2, as described in the post |

### Speed (acceptance B6)
With a random numpy actor playing (sampled actions, forward pass included), `scripts/bench_steps.py --policy`, one pinned core: **29,000 steps/s on `open`** and **33,000 on `T0`**. The target was 3,000.

### Coverage (Delivery 2 DoD)
`src/weightshift/serialize/` is 92–100% per file and `src/weightshift/policy/` is 99–100%. The whole package is at 97%. Coverage is measured serially, as before.

## Acceptance status

Delivery 2 (`DELIVERY_2_ACCEPTANCE.md`):

| Area | Status |
|---|---|
| A setup | pass. Still no torch. The fetch script is pinned and hash-checked; a second run is a no-op; a tampered file is rejected |
| B actor + player | pass. Bit-exact save/load, hand-computed forward, row = input, deterministic episodes, serial = parallel, speed |
| C quantize + dump | pass. Scale rule, ≤3 digits, dump → reader exact for 100 actors |
| D parser | pass. Every form and operator, every skip reason, 10,000-string fuzz with `eval`/`exec` trapped |
| E applier | pass. In order, orientation on non-square tensors, input untouched |
| F golden | pass. See the results above |
| G round trip | pass. Identical action probabilities and identical sampled `T0` episodes |
| H analysis + ablation | pass |
| I manual sign-off | **pending you** |

Delivery 1:


| Area | Status |
|---|---|
| A setup | pass. `uv sync` + `uv run pytest` from a clean clone, no torch installed |
| B rules | pass. Rule tests, 97% env coverage, invariants over 1,000 random episodes |
| C determinism | pass. Byte-identical trajectories, distinct seeds, enforced disjoint seed ranges |
| D observation | pass. Decode/encode round trip over 1,000 states, per-channel checks |
| E layouts | pass. 10,000 layouts per family, `T0` snapshot |
| F calibration | pass. See the table above |
| G metrics | pass. Hand-built known answers, plus G3 |
| H gym + speed | pass. `check_env` on all families, ≥138k steps/s |
| I CLI + replay | pass. Play, demo, JSONL replay |
| J manual sign-off | **pending you**: play one `T0` episode, watch the oracle and greedy demos, confirm the rules |

Where the code had to interpret an acceptance doc, the notes are in its last section (§7 for Delivery 1, §8 for Delivery 2).

## Known issue: crashes under WSL on this machine

On the development machine (Intel Core i9-13900KF), the test suite segfaults intermittently **under WSL**: about 1 in 3 full runs at first, and 7 of 9 while Delivery 2 was built. The crashes are in random hot code. The earlier investigation (commit message of `6812886`) found the processor executing one byte into a valid instruction in numpy, which pointed at the hardware.

**After the BIOS update (2026-11-04):** ASUS ROG STRIX B760-F BIOS 1840 with Intel microcode 0x133.
- The two hottest test files still crashed in **20 of 40 runs under WSL**, and the full suite in 5 of 10.
- The same tests run **natively on Windows** crashed in **0 of 60 runs**: 20 each pinned to the P-cores, pinned to the E-cores, and unpinned.

So the problem is tied to WSL / Hyper-V on this machine, not to a core that fails under Windows. **Decision:** training, gates and every reported number run natively on Windows (`.venv-win`, see Commands). WSL is fine for editing and quick checks, but don't trust results produced there. Intel's diagnostic tool and a CPU stress test have not been run yet; they're still worth doing before long training runs.
