from __future__ import annotations

import argparse
import json
import sys

from scripts.wait_and_materialize_top50 import materialize_command, read_status


def test_read_status_handles_missing_and_completed(tmp_path) -> None:
    result = tmp_path / "result.json"
    assert read_status(result) == "missing"
    result.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    assert read_status(result) == "completed"


def test_materialize_command_forwards_configured_sources(tmp_path) -> None:
    args = argparse.Namespace(
        input=[tmp_path / "a.parquet", tmp_path / "b"],
        weight=[0.25, 0.75],
        requests=tmp_path / "requests.parquet",
        banner_index=tmp_path / "index.parquet",
        candidate_top_k=100,
        rrf_constant=5.0,
        exponent=0.1,
        rerank_top_n=75,
        output=tmp_path / "submission.parquet",
        report=tmp_path / "report.json",
    )

    command = materialize_command(args)

    assert command[0] == sys.executable
    assert command.count("--input") == 2
    assert command.count("--weight") == 2
    assert command[command.index("--rrf-constant") + 1] == "5.0"
    assert command[command.index("--output") + 1] == str(args.output)
