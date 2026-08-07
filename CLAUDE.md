# ML Camp RecSys repository map

Read first:

- `docs/requirements.md` — acceptance criteria and metrics;
- `docs/arch-rules.md` — non-negotiable leakage/cache/process invariants;
- `docs/architecture.md` — data flow, split and run contract;
- `docs/commands.md` — supported commands;
- `docs/plan.md` — implementation status.

Code ownership:

- `configs/` controls all paths, scopes, quotas, features and model parameters;
- `src/mla_recsys/` owns reusable pipeline infrastructure;
- `scripts/` contains one subprocess entry point per heavy stage;
- `tests/` contains unit, smoke and regression contracts;
- `solution/`, `generators/` and `code_maxim/` are legacy-compatible adapters;
- `common/` is external and must not be refactored.

Never inject targets into candidate pools, change the temporal split between
experiments, mix `offline` and `full` counters, read token values, or commit
data/models/runs.

