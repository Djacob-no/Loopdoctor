# Gate report: `actor.npz`

SHA-256 `33f4b9a71d4cc4060040fada717776f8e680a0139eae7f4976abc137af96db11` · 1024 paired `select`-seed episodes per condition · rounded network · **PASS**

| Gate | Result | Key numbers |
|---|---|---|
| E1 failure | pass | T0 success 0.0% sampled, 0.0% greedy (max 2%) |
| E2 pocket | pass | picked a needed item in 97.9% of T0 episodes; loop index 0.857 vs open 0.370; full-not-delivering 0.873 vs open 0.103 |
| E3 competence | pass | open success 98.4%, score 43.2 = 59.8% of oracle; probe 99.0%; detour 100.0% |
| E4 capacity | pass | open score 43.20 → 43.20 with 4 units/layer zeroed (0.0% drop, max 2%) |
| E5 rounding | pass | float T0 success 0.2% sampled, 0.0% greedy (max 2%); rounded − float open score CI [-0.73, 0.47] (must lie within ±1.30, 3% of 43.33) |

| Layouts | Success (sampled) | Success (greedy) | Mean score | Loop index | Idle fraction | Full, not delivering |
|---|---|---|---|---|---|---|
| open | 98.4% | 63.2% | 43.20 | 0.370 | 0.116 | 0.103 |
| detour | 100.0% | 71.4% | 63.54 | 0.239 | 0.023 | 0.076 |
| probe | 99.0% | 77.0% | 61.33 | 0.255 | 0.037 | 0.083 |
| trap | 80.6% | 27.7% | 28.72 | 0.579 | 0.173 | 0.218 |
| T0 | 0.0% | 0.0% | 0.00 | 0.857 | 0.343 | 0.873 |
