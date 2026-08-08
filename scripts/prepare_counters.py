#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pyarrow as pa
import pyarrow.parquet as pq

from mla_recsys.artifacts import (
    atomic_output_path,
    atomic_write_json,
    fingerprint_file,
    write_output_manifest,
)
from mla_recsys.command import load_stage_context
from mla_recsys.config import config_fingerprint
from mla_recsys.counters import COUNTER_EVENT_SCHEMA, scalar_key, stable_text_key, url_domain
from mla_recsys.data import read_request_parquet


EMPTY_WEEKLY_SCHEMA = pa.schema(
    [
        pa.field("entity_type", pa.string(), nullable=False),
        pa.field("entity_id", pa.string(), nullable=False),
        pa.field("week_start", pa.uint64(), nullable=False),
        pa.field("prior_shows", pa.uint64(), nullable=False),
        pa.field("prior_clicks", pa.uint64(), nullable=False),
        pa.field("prior_source_cost", pa.float64(), nullable=False),
    ]
)


def _banner_metadata(path: Path, required: set[int]) -> dict[int, tuple[int | None, str]]:
    table = pq.read_table(path, columns=["BannerID", "GroupExportID", "BannerURL"])
    result: dict[int, tuple[int | None, str]] = {}
    for banner_id, group_id, url in zip(
        table["BannerID"].to_pylist(),
        table["GroupExportID"].to_pylist(),
        table["BannerURL"].to_pylist(),
    ):
        value = int(banner_id)
        if value in required:
            result[value] = (
                int(group_id) if group_id is not None else None,
                url_domain(url),
            )
    return result


def _event_rows(requests: list[dict], banner_index_path: Path) -> list[dict]:
    required = {
        int(banner_id)
        for request in requests
        for banner_id in request["clicked_banner_ids"]
    }
    metadata = _banner_metadata(banner_index_path, required)
    output = []
    for request in requests:
        timestamp = int(request["show_time"])
        query_key = stable_text_key(request.get("query"))
        region_key = scalar_key(request.get("region_id"))
        user_key = scalar_key(request.get("crypta_id_v2"))
        for banner_id, source_cost in zip(
            request["clicked_banner_ids"], request["clicked_source_costs"]
        ):
            banner_id = int(banner_id)
            group_id, domain = metadata.get(banner_id, (None, ""))
            output.append(
                {
                    "show_time": timestamp,
                    "banner_id": banner_id,
                    "group_id": group_id,
                    "domain": domain,
                    "query_key": query_key,
                    "region_key": region_key,
                    "user_key": user_key,
                    "source_cost": float(source_cost),
                }
            )
    output.sort(key=lambda row: (row["show_time"], row["banner_id"]))
    return output


def main() -> int:
    context = load_stage_context("Prepare scope-separated leakage-safe counter artifact")
    cfg = context.cfg
    scope = str(cfg.runtime.scope)
    spec = cfg.split.counters[scope]
    output_dir = context.store.path / "counters" / scope
    output_dir.mkdir(parents=True, exist_ok=True)
    config_sha = config_fingerprint(cfg)

    if str(cfg.features.version) == "feature_v1":
        output = output_dir / "weekly_snapshots.parquet"
        table = pa.Table.from_pylist([], schema=EMPTY_WEEKLY_SCHEMA)
        inputs = [
            fingerprint_file(
                Path(str(cfg.paths.history_artifact)) / "history_candidates.parquet"
            )
        ]
        version = "weekly_past_only_v1"
        semantics = "week(row) sees only weeks strictly before week(row)"
        source_split = None
    else:
        source_split = str(spec.fit_split)
        request_path = context.store.path / "data" / f"{source_split}_requests.parquet"
        banner_index_path = Path(str(cfg.paths.banner_index))
        requests = read_request_parquet(request_path)
        rows = _event_rows(requests, banner_index_path)
        table = pa.Table.from_pylist(rows, schema=COUNTER_EVENT_SCHEMA)
        output = output_dir / "click_events.parquet"
        inputs = [fingerprint_file(request_path), fingerprint_file(banner_index_path)]
        version = str(cfg.features.counter_version)
        semantics = (
            "click-only events; train/full_train lookup is strict show_time < row; "
            "holdout/test state is frozen at scope cutoff"
        )

    with atomic_output_path(output) as temporary:
        pq.write_table(table, temporary, compression="zstd")
    write_output_manifest(
        output,
        stage="prepare_counters",
        artifact_version=version,
        config_sha256=config_sha,
        inputs=inputs,
        rows=table.num_rows,
        schema=str(table.schema),
        scope=scope,
    )
    scope_manifest = {
        "version": 2,
        "scope": scope,
        "fit_split": source_split,
        "frozen_cutoff_ts": int(spec.frozen_cutoff_ts),
        "source_tables": list(spec.sources),
        "semantics": semantics,
        "counter_rows": table.num_rows,
        "counter_artifact": output.name,
        "contains_impressions": False,
        "used_by_features": str(cfg.features.version) != "feature_v1",
        "inputs": inputs,
    }
    atomic_write_json(output_dir / "scope.json", scope_manifest)
    print(json.dumps(scope_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
