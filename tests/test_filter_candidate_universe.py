from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from filter_candidate_universe import filter_and_rerank  # noqa: E402


def test_filter_and_rerank_restores_contiguous_request_ranks() -> None:
    schema = pa.schema(
        [
            pa.field("hit_log_id", pa.uint64(), nullable=False),
            pa.field("banner_id", pa.uint64(), nullable=False),
            pa.field("source_rank", pa.int32(), nullable=False),
        ]
    )
    table = pa.Table.from_pylist(
        [
            {"hit_log_id": 1, "banner_id": 10, "source_rank": 1},
            {"hit_log_id": 1, "banner_id": 20, "source_rank": 2},
            {"hit_log_id": 1, "banner_id": 30, "source_rank": 3},
            {"hit_log_id": 2, "banner_id": 20, "source_rank": 1},
            {"hit_log_id": 2, "banner_id": 40, "source_rank": 2},
        ],
        schema=schema,
    )

    filtered = filter_and_rerank(table, {10, 20}).to_pylist()

    assert filtered == [
        {"hit_log_id": 1, "banner_id": 30, "source_rank": 1},
        {"hit_log_id": 2, "banner_id": 40, "source_rank": 1},
    ]
