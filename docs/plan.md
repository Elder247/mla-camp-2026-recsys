# Implementation plan

Source of truth:
`/Users/astrofimuk/REPS/mla_camp_recsys/docs/next_iterations_plan.md`.
This file tracks implementation status only; it does not redefine priorities.

| block | status | gate |
|---|---|---|
| Audit VM/git/data/artifacts/GPU/baseline | completed | clean `main`, H100 visible through torch, schemas and legacy metrics captured |
| Docs/config/run contract | completed | commit `786f69e`; 10 pytest + 10 legacy unittest pass |
| Iteration 0 baseline pipeline | in progress | temporal/natural pool, cache parity, reproducible honest baseline |
| Iteration 1 candidate generators | pending | complementarity and SC ceiling gates |
| Iteration 1 feature v2 | pending | chunk parity, leakage-safe schema, memory/timing report |
| Iteration 1 SC-aware CatBoost | pending | honest SC@50 win, importance, validated batch prediction |
| Iteration 2/3 | blocked by design | do not start before Iterations 0/1 reproduce |

Implementation proceeds in small commits matching these blocks. Architecture
changes update `docs/architecture.md`; command changes update
`docs/commands.md`; scope/priorities change only in the source-of-truth plan.

Iteration 0 smoke evidence:

- `20260807_2300_i0_smoke` and `20260807_2330_i0_smoke_repeat` completed;
- both produced 36,459/36,271 merged train/holdout rows and 10,000 feature rows per split;
- natural positives: train 16/20, holdout 11/20; misses stayed out of fit;
- candidate and ranker metrics were identical across runs;
- direct and cached top-50 matched for all 40 requests in the repeat run.
