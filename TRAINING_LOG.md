# Training log (Delivery 3)

Every training attempt and exploration run, including the ones that failed (DELIVERY_3_ACCEPTANCE §4 D1). An **attempt** is one config trained on 3 seeds and judged against the gates. A **sweep** is a single-seed exploration run (for example, learning speed) that is not a gate candidate.

All runs are on native Windows (`.venv-win`), CPU only, PyTorch 2.15.0+cpu. Checkpoint numbers come from 128 `select`-seed episodes per family, sampled actions, on the float network. Gate numbers (1,024 episodes, rounded network) are in `artifacts/d3/`.

`uv run python scripts/summarize_runs.py` prints the current table from `runs/`.

| Run | Config hash | What changed and why | Steps | Outcome |
|---|---|---|---|---|
| attempt01 (seeds 0–2) | `5c1f64bf6733` | First guess: β 0.1; mix open 0.5, detour 0.2, trap 0.15, T0 0.15; CleanRL defaults | 10M | **Fail both ways.** T0 success 90–95% (the robot learned the way round; no failure induced). Competence far off: open success 54–59% and mean score ≈ 1.0 (oracle ≈ 75). T0's own score (7–11) is higher than open's because T0 is one fixed layout it trains on. |
| sweep baseline (seed 0) | `75be66ba414b` | Learning-speed reference: CleanRL defaults, open rooms only, β 0.1 | 10M | open 69% / score 1.5. Without pocket or detour training, T0 falls to 34% and trap to 10%; detour only 38%. |
| sweep lr1e-3 (seed 0) | `690fb584c760` | Learning rate 4× (1e-3), open only | 10M | **Fastest:** open 87% / score 4.0. T0 falls to **0.8%** and trap 15%, but detour 44%: the pocket failure appears on its own, together with a general inability to go round walls. |
| sweep epochs10 (seed 0) | `18f433512dec` | lr 5e-4, 10 epochs per batch, open only | 10M | open 86% / score 3.8, but about 40% slower per step than lr1e-3. T0 7%, detour 57%. |
| attempt02 (seeds 0–2) | `8eae2db06d1e` | Attempt 1 solved T0. Ten times stronger shaping (β 1.0), same mix, 30M steps | 30M | **Fail both ways.** T0 success 89–98%: stronger shaping did *not* trap the robot. As expected, β(d_prev − d) sums to the same total along any route that ends at the dock, so the detour costs nothing net. Open success 58–62%, score ≈ 1.0. |
| attempt03 (seeds 0–2) | `e4a7611dcf22` | No pocket layouts in training (open 0.6, probe 0.2, detour 0.2), β 0.5, 30M | 30M | **Fail.** T0 success 65–81%: what is learned on straight-wall detours carries over to the pocket. Open 68–73%, score 1.5–1.9; detour 68–75%. |
| sweep long-open (seed 0) | `bb6334f1d43d` | Competence ceiling: open only, lr 1e-3, 50M | 50M | Competence keeps rising: open 95% / score 21.7 (≈ 29% of the oracle), detour 85%, probe 88% at 50M. **T0 rises with it:** about 4–10% up to 16M, then 22% at 31M and 60% at 50M. A competent network generalises its way round the pocket. |
| sweep long-mixed (seed 0) | `4657562a619c` | Open 0.6 + detour 0.2 + probe 0.2, lr 1e-3, 50M | 50M | Open 87% / 8.4, detour 88%, T0 83%. Detour training makes T0 easy. |

## Finding after 3 attempts and 5 sweeps (≈ 2.5 h of compute), 2026-11-04

1. **The planned shaping cannot create the trap on its own.** β(d_prev − d) sums to β × (distance where the item was picked − distance where it ends), whatever the route, so the detour costs nothing net. Raising β (attempt 2) made T0 *easier*, not harder.
2. **Competence and the T0 failure move together.** T0 is low (≤ 10%) only while the network is weak. As it learns to deliver and to go round walls, it also learns T0 (long-open: 60% at 50M; any detour training: 65–98%). The plan's warning (FINAL_PLAN §6.2: "train on open layouts and hope" gives a network that never learned detours) is what happens early; later the network learns detours and T0 together.
3. **Competence is slow but real.** With lr 1e-3, open score grows about 1.5 points per 5M steps and reached 29% of the oracle at 50M. The 40% bar looks reachable at around 80–100M steps (about 1 h per seed on this machine).

Stopped here to ask how to induce the failure (DELIVERY_3_ACCEPTANCE §4 D4), rather than changing the shaping rule unasked.

**Decision (you, 2026-11-04):** induce the failure with a disclosed **short practice clock on T0** (12 steps in training; the fastest possible delivery takes 14), and keep the 40% competence bar with up to about 100M steps.

| Run | Config hash | What changed and why | Steps | Outcome |
|---|---|---|---|---|
| attempt04 (seeds 0–2) | `632495923a84` | Short practice clock on T0 (12 steps); mix open 0.55, detour 0.15, probe 0.15, T0 0.15; lr 1e-3; β 0.1; checkpoint every 1M | 100M | **Fail on the trap; competence much better.** At 100M: open 92–99% / score 20.0–26.8 (≈ 27–36% of the oracle), detour 98–99% / 35–49, probe 98–100% / 26–32. But T0 98% / 15–35: the robot plays T0 well although T0 never paid in practice. On the way, T0 was briefly low on some seeds (2–10% at 4–10M) while the network was weak, as in the sweeps. |
| attempt05 (seeds 0–2) | `b8d1034bb460` | Attempt 4 at 10M: the short clock alone is too weak (T0 10% / 40% / 91% on seeds 0/1/2). β 1.0 so that walking away on T0 costs points that never pay back before the clock ends; T0 share 0.25 | stopped at 12M | **Fail.** T0 success back to 62–66% by 7–10M on all seeds, and also 65% with greedy actions. Stopped early to free the CPU; the trend was clear. |

**Diagnostic, 2026-11-04 (checkpoints of attempts 4 and 5, 128 select-seed episodes each, sampled vs greedy):**

| Run, checkpoint | T0 sampled | T0 greedy | open sampled | open greedy | detour sampled | detour greedy |
|---|---|---|---|---|---|---|
| attempt04 s0, 4M | 7% | 0% | 73% | 4% | 58% | 0% |
| attempt04 s0, 23M | 94% | 0% | 84% | 7% | 80% | 3% |
| attempt04 s1, 4M | 2% | 0% | 68% | 8% | 64% | 1% |
| attempt04 s1, 23M | 92% | 2% | 88% | 11% | 79% | 2% |
| attempt05 s0, 10M | 65% | 65% | 76% | 20% | 67% | 25% |

**Greedy play almost never delivers anywhere.** The trained networks are not navigators yet. They drift in roughly the right direction and rely on random action sampling to get unstuck. That is why scores stay low, and why the sampled robot eventually wanders round the T0 pocket by chance over 200 steps, even when T0 never paid in practice (attempt 4). Stopped here to report.

**Decision (you, 2026-11-04):** make the robot more decisive.

| Run | Config hash | What changed and why | Steps | Outcome |
|---|---|---|---|---|
| attempt06 (seeds 0–2) | `751314349066` | Attempt 4 plus one change: the entropy bonus anneals 0.01 → 0 over training, so the final robot is decisive rather than relying on random sampling | 100M | **Fail.** T0 91–97% (sampled). Competence like attempt 4: open 94–97% / score 18–21, detour 97–98% / 34–44. It did **not** become decisive: greedy play delivers in 22–29% of open rooms (score 2–3), and greedy T0 varies by seed (0%, 34%, 92%). |

## Status after 6 attempts and 5 sweeps (≈ 6 h of compute), 2026-11-04: stopped to ask

- **Competence** improves with long training (lr 1e-3, 100M steps): open delivery 92–99%, score 18–27 (24–36% of the oracle; the bar is 40%).
- **The trained robots rely on random sampling to work.** Their single most likely action (greedy play) delivers in only about 20–30% of rooms. Annealing the entropy bonus to 0 did not change this.
- **They generalise strongly to T0.** None of the induction methods so far keeps a competent robot stuck there: strong shaping, no pockets in training, or a 12-step practice clock on T0 (where T0 never paid at all). T0 success ends at 90–98%.

**Decision (you, 2026-11-04):** induce the failure with a disclosed **electric fence on T0**: in T0 practice episodes, entering the pocket ends the episode with a penalty.

| Run | Config hash | What changed and why | Steps | Outcome |
|---|---|---|---|---|
| attempt07 (seeds 0–2) | `750f75e01f16` | Fence on T0 (rows 4–8, cols 3–6, penalty 5; full-length T0 practice); β 0.5; lr 1e-3; mix as attempt 4 | 150M (≈ 68 min per seed, 3 in parallel) | **Fail (E2, E3, E4); the failure itself holds.** Gates at 150M (1,024 select seeds, rounded network): E1 pass on all seeds (T0 0.5 / 0.8 / 0.6% sampled, 0% greedy). E5 pass on all seeds. E2 fails only on the loop-index ratio (T0 0.87–0.88 vs open 0.57–0.63; the bar is 3×); full-but-not-delivering passes (0.56–0.70 vs 0.00). E3: open 85–88% / score 8.7–11.1 (12–15% of the oracle; bar 95% / 40%), probe 89–93%, detour 86–88%. E4: zeroing 4 units per layer drops the open score 45–69% (bar < 2%): the width-32 network uses all its units. Reports: `runs/gates/attempt07-s*/`. |
| ↳ attempt07 at 31M | | progress note | 31M | T0 stays at 0–6% on all seeds from the start; generated trap (no fence) 71–77%, so the avoidance is T0-specific. Competence is growing more slowly than attempt 4: open 74–78% / score 2.0–2.3 (attempt 4: ≈ 5 at 26M). |

## Status after 7 attempts and 5 sweeps (≈ 8.5 h of compute), 2026-11-04: stopped to ask (budget reached)

- **The fence works.** All 3 seeds are stuck on T0 (≤ 1%, greedy 0%), fetch items there (88–93%) and then hold them without delivering. They still go round the unfenced generated trap (81–85%), so the failure is specific to T0 and is not a general inability.
- **Competence is far from the bar.** The open score is 12–15% of the oracle (the bar is 40%), and the fence slowed learning compared with attempt 4.
- **Capacity fails badly at width 32.** The plan's rule for this is to widen to 64.
- **The E2 loop-index ratio may be unreachable as written.** The open loop index is already 0.57–0.63, so 3× would need a value above 1.7.

**Decision (you, 2026-11-04):** a bigger network (width 64) and more of the CPU: three variants × 3 seeds trained in parallel (9 processes, about 26.5k steps/s each, about 240k steps/s in total). The E2 loop-index rule is now "T0 loop index ≥ 0.80" (DELIVERY_3_ACCEPTANCE E2). How a variant is chosen was fixed before training (§8 note 7).

| Run | Config hash | What changed and why | Steps | Outcome |
|---|---|---|---|---|
| attempt08a (seeds 0–2) | `3035f9e5a3ca` | Attempt 7 at width 64 (E4 fix) | 300M | **Fail (E3, E4; s2 also E5).** Stuck on T0 on every seed (0.0–0.4% sampled, 0% greedy), E2 passes. Open score 23.8 / 25.1 / 29.2 (33 / 35 / 40% of the oracle); only s2 reaches the bar. E4: a 12–43% drop. s2's E5 CI is [0.08, 0.64] (rounding *raises* the score slightly). |
| attempt08b (seeds 0–2) | `7719df58bb8c` | 8a with β 0.1 (attempt 4's value; attempt 4 learned faster than attempt 7) | 300M | **Fail on E4 only for s1 and s2; s0 also fails E5.** T0 0.4–0.6% sampled, 0% greedy; E2 passes (picks a needed item in 98–100%). Open 98–99% / score 43.1 / 53.1 / 49.3 (60 / 73 / 68% of the oracle); probe 98–100%, detour ≈ 100%. E4: a 29 / 14 / 42% drop. s0's E5 CI is [−1.29, −0.01] (rounding costs ≈ 0.6 points of 43). |
| attempt08c (seeds 0–2) | `b15d13fd6bbb` | 8b with T0 share 0.10, open 0.60 | 300M | **Fail (E4 on all seeds; s1 E1 at 2.1%; s1 and s2 E5).** Open score 42.9 / 48.1 / 45.1. E4: a 59 / 16 / 23% drop. s2's E5 CI is [−1.40, −0.05]. |

**Attempt 8 gates (300M, 1,024 select seeds, rounded network; reports in `runs/gates/attempt08*/`).** Training took about 3.2 h of compute, plus the time the PC was asleep; 9 runs ran in parallel at 26.5k steps/s each. The smaller dock bonus (β 0.1, 8b) roughly doubles competence over 8a, and the fence still holds. **No seed passes E4.** Zeroing the 4 weakest of 64 units per layer costs 12–59% of the open score: the trained network spreads its work over every unit, so it has no spare units. E5 fails on 4 of 9 seeds by small margins (the rounded − float difference is under 1.5% of the score, but 1,024 paired episodes are enough to detect it).

## Status after 8 attempts (≈ 12 h of compute, over the 8 h budget), 2026-11-05: stopped to ask

- **The fence plus β 0.1 works**: 8b is stuck on T0 and highly competent (60–73% of the oracle).
- **E4 fails on every seed.** The network has no spare units.
- **E5 fails on some seeds by a margin too small to matter.**

**Decision (you, 2026-11-05):** prune 4 units per layer from 8b and fine-tune (DELIVERY_3_ACCEPTANCE §8 note 8). E5 now uses a ±3% equivalence margin.

| Run | Config hash | What changed and why | Steps | Outcome |
|---|---|---|---|---|
| attempt09 (seeds 0–2) | `ce1b39627838` | Starts from each 8b seed's final network; the 4 E4 units per hidden layer are switched off and held at 0; fresh optimizer, lr 2.5e-4 annealed; otherwise as 8b | +40M (≈ 24 min, 3 in parallel at 28k steps/s) | **Pass on seeds 0 and 2 (E6: 2 of 3).** s0: T0 0.0% / 0.0% greedy, open 98.4% / 43.2 (60% of the oracle), E4 0.0% drop, E5 CI [−0.73, 0.47]. s2: T0 0.4% / 0.0%, open 98.5% / 48.7 (67%), E4 0.0%, E5 [−0.24, 0.80]. s1 fails: T0 3.6% on the rounded network (1.7% float), and rounding costs ≈ 3.4 points (E5 CI [−4.17, −2.59]). |

## Result

**Chosen: attempt 9, seed 0 (primary, `artifacts/d3/33f4b9a71d4c/`) and seed 2 (second seed, `artifacts/d3/988a4947f65c/`).** These are the final checkpoints (+40M steps after 8b's 300M), as the §2 rule requires. The primary is the lowest-numbered passing seed. In total: 9 attempts, 5 sweeps, and roughly 13 h of compute.
