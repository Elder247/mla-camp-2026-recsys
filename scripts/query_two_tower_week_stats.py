#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def load_config(path: Path):
    cfg = OmegaConf.load(path)
    parent = cfg.get("extends")
    if parent:
        base = load_config((path.parent / str(parent)).resolve())
        cfg = OmegaConf.merge(base, cfg)
        del cfg["extends"]
    return cfg


def issue_messages(issues: object) -> list[str]:
    output = []
    for issue in issues or []:
        output.append(str(getattr(issue, "message", None) or issue))
        output.extend(issue_messages(getattr(issue, "issues", None)))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Query weekly raw-train statistics")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--token-path", type=Path, default=Path.home() / ".yql" / "token"
    )
    args = parser.parse_args()
    if not args.token_path.is_file():
        raise FileNotFoundError("YQL token file is unavailable")
    os.environ["YQL_TOKEN_PATH"] = str(args.token_path)
    cfg = load_config(args.config.resolve())
    sys.path.insert(0, str(cfg.paths.step2_root))
    from common.yt_data import make_client
    from mla_recsys.artifacts import atomic_write_json
    from yql.api.v1.client import YqlClient
    from yql.client.explain import YqlSqlValidateRequest

    output = Path(str(cfg.paths.week_stats_file))
    report_path = output.with_suffix(".json")
    if output.is_file():
        table = pq.read_table(output)
        if table.schema.names == ["week_start", "impressions", "clicks"]:
            report = {
                "version": 1,
                "status": "cache_hit",
                "output": str(output),
                "weeks": table.num_rows,
            }
            atomic_write_json(report_path, report)
            print(json.dumps(report, indent=2))
            return 0
        raise RuntimeError("Existing week stats parquet has an incompatible schema")

    client = make_client()
    source = str(cfg.paths.raw_train_table)
    if not client.exists(source):
        raise FileNotFoundError(f"YT raw table is unavailable: {source}")
    source_rows = int(client.get(f"{source}/@row_count"))
    query = (ROOT / "yql" / "two_tower_week_stats.yql").read_text(
        encoding="utf-8"
    ).replace("__INPUT_TABLE__", source)
    query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
    validation = YqlSqlValidateRequest(query, sql=True, syntax_version=1)
    validation.run()
    validation.get_results()
    if not validation.is_success:
        raise RuntimeError(
            "YQL validation failed: " + "; ".join(issue_messages(validation.errors))
        )

    started = time.perf_counter()
    yql = YqlClient(
        server="yql.yandex.net", port=443, token_path=str(args.token_path)
    )
    operation = yql.query(query, title="ML Camp YQL TwoTower 10M week stats")
    operation.run()
    operation_id = str(operation.operation_id)
    atomic_write_json(
        report_path,
        {
            "version": 1,
            "status": "submitted",
            "source": source,
            "source_rows": source_rows,
            "operation_id": operation_id,
            "query_sha256": query_sha256,
        },
    )
    results = operation.get_results()
    if not operation.is_success:
        raise RuntimeError(
            "YQL operation failed: " + "; ".join(issue_messages(operation.errors))
        )
    frames = [result.full_dataframe for result in results]
    if len(frames) != 1:
        raise RuntimeError(f"Expected one week-stats result, got {len(frames)}")
    table = pa.Table.from_pandas(frames[0], preserve_index=False)
    table = table.select(["week_start", "impressions", "clicks"])
    if table.num_rows == 0:
        raise RuntimeError("Week statistics are empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, output)
    rows = table.to_pylist()
    report = {
        "version": 1,
        "status": "completed",
        "source": source,
        "source_rows": source_rows,
        "output": str(output),
        "weeks": table.num_rows,
        "first_week": int(rows[0]["week_start"]),
        "last_week": int(rows[-1]["week_start"]),
        "impressions": sum(int(row["impressions"]) for row in rows),
        "clicks": sum(int(row["clicks"]) for row in rows),
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
        "operation_id": operation_id,
        "query_sha256": query_sha256,
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
