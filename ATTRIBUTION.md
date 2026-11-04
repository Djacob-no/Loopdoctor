# Attribution

## MIT CSAIL, *Sparks of Weight-Space Programming*: Experiment 4 logs
- Authors: Takehi, Teoh, Zhang, Khattab (MIT CSAIL), 2026-11-02.
  Post: https://rikiyat.github.io/blog/weight-space-programming/
- Logs repository: https://github.com/RikiyaT/lm_understand-logs, commit
  `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`.
- Licence: **CC BY 4.0** for the logs and networks, MIT for the scripts.
- What we use: `exp4_fix_car/networks/ppo_final_seed0.npz`, `networks/edited_actor.npz`,
  `runs/oneshot_repair/reply/prompt.txt`, `runs/oneshot_repair/reply/answer.md` and
  `eval/oneshot_result.json`. They serve as ground truth for our weight serializer, edit
  parser and applier (the golden test, `tests/test_golden_exp4.py`).
- How: the files are **not redistributed** in this repository. `scripts/fetch_exp4.py`
  downloads them from the pinned commit into `fixtures/exp4/` (gitignored) and checks
  their SHA-256 hashes. No changes are made to them.
- Our weight-dump text format and edit language follow Exp 4's published prompt.

## Inspiration
- The game is loosely inspired by Grocery Bot (grocerybot.no). None of its code, assets,
  maps, protocol or branding is used (FINAL_PLAN §16).
