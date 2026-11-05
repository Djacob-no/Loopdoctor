# Gate report: `actor.npz`

SHA-256 `988a4947f65c5148d54c5d8dc3eeacc2ae10484881f32b319e4ce3aee3d3b080` · 1024 paired `select`-seed episodes per condition · rounded network · **PASS**

| Gate | Result | Key numbers |
|---|---|---|
| E1 failure | pass | T0 success 0.4% sampled, 0.0% greedy (max 2%) |
| E2 pocket | pass | picked a needed item in 88.3% of T0 episodes; loop index 0.862 vs open 0.319; full-not-delivering 0.790 vs open 0.103 |
| E3 competence | pass | open success 98.5%, score 48.7 = 67.4% of oracle; probe 97.9%; detour 99.6% |
| E4 capacity | pass | open score 48.71 → 48.71 with 4 units/layer zeroed (0.0% drop, max 2%) |
| E5 rounding | pass | float T0 success 0.7% sampled, 0.0% greedy (max 2%); rounded − float open score CI [-0.24, 0.80] (must lie within ±1.45, 3% of 48.42) |

| Layouts | Success (sampled) | Success (greedy) | Mean score | Loop index | Idle fraction | Full, not delivering |
|---|---|---|---|---|---|---|
| open | 98.5% | 62.6% | 48.71 | 0.319 | 0.101 | 0.103 |
| detour | 99.6% | 66.2% | 66.31 | 0.224 | 0.021 | 0.075 |
| probe | 97.9% | 68.3% | 60.69 | 0.304 | 0.042 | 0.059 |
| trap | 84.0% | 26.0% | 33.99 | 0.542 | 0.180 | 0.171 |
| T0 | 0.4% | 0.0% | 0.10 | 0.862 | 0.346 | 0.790 |
