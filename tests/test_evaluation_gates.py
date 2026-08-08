from __future__ import annotations

from mla_recsys.evaluation import source_gate_rows


def test_incremental_oracle_and_high_value_unique_hits() -> None:
    a = ("r1", 1)
    b = ("r2", 2)
    c = ("r3", 3)
    truth = {a: 10.0, b: 20.0, c: 100.0}
    sources = ["baseline", "new"]
    found = {"baseline": {a: 1}, "new": {b: 2, c: 3}}
    metric = {
        "50": {"recall": 0.0, "sourcecost_recall": 0.0},
        "500": {"recall": 0.0, "sourcecost_recall": 0.0},
    }
    rows = source_gate_rows(
        truth=truth,
        found=found,
        sources=sources,
        metrics={"baseline": metric, "new": metric},
        baseline_sources=["baseline"],
        high_value_quantile=0.9,
    )
    new = next(row for row in rows if row["source"] == "new")
    assert new["incremental_oracle_hits_vs_baseline"] == 2
    assert new["incremental_oracle_sourcecost"] == 120.0
    assert new["unique_high_value_hits"] == 1
    assert new["high_value_hits"] == 1
    assert new["high_value_sourcecost_recall"] == 1.0
