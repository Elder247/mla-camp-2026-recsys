from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from mla_recsys.candidate_cache import (
    SourceSpec,
    generate_source_candidates,
    source_part_path,
)
from mla_recsys.pipeline import Generator


class EchoModule:
    @staticmethod
    def rank(*, model, example, features, top_k):
        del model, features, top_k
        value = int(str(example["request_id"])[1:])
        return [
            {"banner_id": value * 10 + offset, "score": float(3 - offset)}
            for offset in range(3)
        ]


def _spec() -> SourceSpec:
    return SourceSpec(
        name="echo",
        feature_name="echo",
        generator=Generator(
            name="echo",
            module=EchoModule,
            model={},
            top_k=3,
            quota=2,
            weight=1.0,
            features={},
            batch_size=1,
        ),
        code_path=Path("echo.py"),
        dependency_paths=(),
        artifact_dir=Path("echo"),
    )


def _rows(root: Path) -> list[dict]:
    rows = []
    for partition in range(2):
        rows.extend(
            pq.read_table(source_part_path(root, "train", "echo", partition)).to_pylist()
        )
    return sorted(rows, key=lambda row: (row["request_id"], row["source_rank"]))


def test_parallel_generation_matches_single_worker(tmp_path: Path) -> None:
    requests = [
        {"request_id": f"r{index}", "hit_log_id": index, "query": "query"}
        for index in range(1, 7)
    ]
    single = tmp_path / "single"
    parallel = tmp_path / "parallel"

    generate_source_candidates(
        spec=_spec(),
        requests=requests,
        run_path=single,
        split="train",
        partitions=2,
        buffer_rows=3,
    )
    report = generate_source_candidates(
        spec=_spec(),
        requests=requests,
        run_path=parallel,
        split="train",
        partitions=2,
        buffer_rows=3,
        request_workers=2,
        parallel_batch_size=2,
    )

    assert _rows(parallel) == _rows(single)
    assert report["requests"] == len(requests)
    assert report["rows"] == len(requests) * 2
