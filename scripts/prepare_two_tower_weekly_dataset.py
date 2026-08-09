#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOKEN_PATTERN = re.compile(r"(?:y[01]_|t[01]_|AQAD-)[A-Za-z0-9_\-]+")
YT_PATH_PATTERN = re.compile(r"^//[A-Za-z0-9_./-]+$")


def mask(value: object) -> str:
    return TOKEN_PATTERN.sub("***", str(value))


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
        output.append(mask(getattr(issue, "message", None) or issue))
        output.extend(issue_messages(getattr(issue, "issues", None)))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Create sorted weekly TwoTower YT data")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--token-path", type=Path, default=Path.home() / ".yql" / "token"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate syntax and live schemas without creating the YT table",
    )
    args = parser.parse_args()
    if not args.token_path.is_file():
        raise FileNotFoundError("YQL token file is unavailable")
    # The YQL client reads this path itself. The token value is never opened,
    # copied to argv, rendered, or persisted by this process.
    os.environ["YQL_TOKEN_PATH"] = str(args.token_path)
    cfg = load_config(args.config.resolve())
    sys.path.insert(0, str(cfg.paths.step2_root))
    from common.yt_data import make_client
    from two_tower_v2.data import source_fields
    from two_tower_v2.training import all_cardinalities, atomic_json
    from yql.api.v1.client import YqlClient
    from yql.client.explain import YqlSqlValidateRequest

    source = str(cfg.paths.raw_train_table)
    target = str(cfg.paths.weekly_train_table)
    started = time.perf_counter()
    if not YT_PATH_PATTERN.fullmatch(source) or not YT_PATH_PATTERN.fullmatch(target):
        raise ValueError("Unsafe YT input/output path")
    client = make_client()
    artifact_dir = Path(
        str(cfg.paths.get("dataset_artifact_dir", cfg.paths.artifact_dir))
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / "weekly_dataset.json"
    if client.exists(target):
        schema = [str(item["name"]) for item in client.get(f"{target}/@schema")]
        sorted_by = (
            [str(value) for value in client.get(f"{target}/@sorted_by")]
            if client.exists(f"{target}/@sorted_by")
            else []
        )
        required = {"week_start", "show_time", *source_fields(all_cardinalities(cfg))}
        if required <= set(schema) and sorted_by[:1] == ["week_start"]:
            report = {
                "version": 1,
                "status": "cache_hit",
                "source": source,
                "target": target,
                "rows": int(client.get(f"{target}/@row_count")),
                "sorted_by": sorted_by,
                "schema": schema,
            }
            atomic_json(report_path, report)
            print(json.dumps(report, indent=2))
            return 0
        raise RuntimeError("Existing weekly YT table has an incompatible contract")

    template = (ROOT / "yql" / "two_tower_weekly_dataset.yql.template").read_text(
        encoding="utf-8"
    )
    query = template.replace("__INPUT_TABLE__", source).replace(
        "__OUTPUT_TABLE__", target
    )
    query_path = artifact_dir / "weekly_dataset.resolved.yql"
    query_path.write_text(query, encoding="utf-8")
    validation = YqlSqlValidateRequest(query, sql=True, syntax_version=1)
    validation.run()
    validation.get_results()
    if not validation.is_success:
        raise RuntimeError("YQL validation failed: " + "; ".join(issue_messages(validation.errors)))
    if args.validate_only:
        report = {
            "version": 1,
            "status": "validated",
            "source": source,
            "target": target,
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        }
        atomic_json(report_path, report)
        print(json.dumps(report, indent=2))
        return 0

    yql = YqlClient(server="yql.yandex.net", port=443, token_path=str(args.token_path))
    operation = yql.query(query, title="ML Camp YQL TwoTower weekly dataset")
    operation.run()
    operation_id = str(operation.operation_id)
    atomic_json(
        report_path,
        {
            "version": 1,
            "status": "submitted",
            "source": source,
            "target": target,
            "operation_id": operation_id,
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        },
    )
    operation.get_results()
    if not operation.is_success:
        raise RuntimeError("YQL operation failed: " + "; ".join(issue_messages(operation.errors)))
    if not client.exists(target):
        raise RuntimeError("YQL succeeded but weekly output table is missing")
    schema = [str(item["name"]) for item in client.get(f"{target}/@schema")]
    sorted_by = (
        [str(value) for value in client.get(f"{target}/@sorted_by")]
        if client.exists(f"{target}/@sorted_by")
        else []
    )
    if sorted_by[:1] != ["week_start"]:
        raise RuntimeError(f"Weekly output is not sorted by week_start: {sorted_by}")
    report = {
        "version": 1,
        "status": "completed",
        "source": source,
        "target": target,
        "rows": int(client.get(f"{target}/@row_count")),
        "sorted_by": sorted_by,
        "schema": schema,
        "operation_id": operation_id,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
        "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
    }
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
