from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def validate_submission(
    path: Path,
    *,
    expected_hitlog_ids: set[int],
    valid_banner_ids: set[int],
    top_k: int,
    allow_short: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    table = pq.read_table(path)
    expected_schema = pa.schema(
        [
            pa.field("HitLogID", pa.uint64(), nullable=False),
            pa.field("BannerID", pa.list_(pa.uint64()), nullable=False),
        ]
    )
    if table.schema != expected_schema:
        errors.append(f"schema mismatch: {table.schema} != {expected_schema}")
    rows = table.to_pylist()
    hitlogs = [int(row["HitLogID"]) for row in rows]
    if len(hitlogs) != len(set(hitlogs)):
        errors.append("duplicate HitLogID rows")
    missing = expected_hitlog_ids - set(hitlogs)
    extra = set(hitlogs) - expected_hitlog_ids
    if missing:
        errors.append(f"missing HitLogID count: {len(missing)}")
    if extra:
        errors.append(f"unknown HitLogID count: {len(extra)}")
    short = 0
    unknown = 0
    duplicate_rows = 0
    for row in rows:
        banners = [int(value) for value in row["BannerID"]]
        if len(banners) > top_k:
            errors.append(f"HitLogID {row['HitLogID']} has more than {top_k} banners")
        if len(banners) < top_k:
            short += 1
        if len(banners) != len(set(banners)):
            duplicate_rows += 1
        unknown += sum(value not in valid_banner_ids for value in banners)
    if short and not allow_short:
        errors.append(f"short rows: {short}")
    if duplicate_rows:
        errors.append(f"rows with duplicate BannerID: {duplicate_rows}")
    if unknown:
        errors.append(f"unknown BannerID values: {unknown}")
    return {
        "ok": not errors,
        "path": str(path),
        "rows": table.num_rows,
        "expected_rows": len(expected_hitlog_ids),
        "short_rows": short,
        "errors": errors,
    }

