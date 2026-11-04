# WeightShift

A small warehouse grid game built to test one question: **can a frontier LLM repair a trained policy by editing its weights directly?** The experiment reproduces, at hobby scale, Experiment 4 of MIT CSAIL's *Sparks of Weight-Space Programming*. See [`FINAL_PLAN.md`](FINAL_PLAN.md) for the experiment and [`DELIVERY_1_ACCEPTANCE.md`](DELIVERY_1_ACCEPTANCE.md) for this delivery's rules and acceptance criteria. `audit_trail/` holds the earlier plan and reviews.

**Status: Delivery 1 is implemented. It is a playable, tested environment with no neural network yet.** PPO, weight serialization and LLM calls come in later deliveries.

## Commands

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                                    # install (exact versions from uv.lock)
uv run pytest                                              # all acceptance tests (~25 s)
uv run python -m weightshift.cli.play --layout T0          # play by hand
```

More:

```bash
uv run python -m weightshift.cli.demo --player oracle --layout T0                  # ASCII movie + metrics
uv run python -m weightshift.cli.demo --player greedy_straight --layout T0 --clear --delay 0.05
uv run python -m weightshift.cli.play --layout trap --seed 42 --record runs/me.jsonl
uv run python scripts/bench_steps.py --family open                                  # steps/s on one core
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
fixtures/T0.txt            the headline layout
scripts/bench_steps.py     throughput benchmark
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

## Acceptance status

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

Where the code had to interpret the acceptance doc, the notes are in its §7.

## Known issue: intermittent crashes on this machine (hardware, not code)

On the development machine (Intel Core i9-13900KF), about 1 in 3 full test runs crashes with a segfault, or once with `SystemError: error return without exception set`, in random hot code. Re-running passes.

The investigation is in the commit message of `6812886`. In short, the crash executes one byte into the middle of a valid instruction in numpy's `array_assign_subscript`, and no code path jumps there. That is consistent with Intel's confirmed 13th/14th-gen Raptor Lake instability, not with a software bug. **This matters for the experiment.** An unstable CPU can also produce wrong numbers silently, not only crashes. It should be fixed before trusting long PPO runs:
- Update the BIOS to a version with Intel microcode 0x12B or later.
- Use Intel default power settings.
- Run the Intel Processor Diagnostic Tool.
- If it persists, use Intel's extended warranty.
