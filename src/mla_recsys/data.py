from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import atomic_output_path


REQUEST_SCHEMA = pa.schema(
    [
        pa.field("request_id", pa.string(), nullable=False),
        pa.field("hit_log_id", pa.uint64(), nullable=False),
        pa.field("show_time", pa.uint64(), nullable=True),
        pa.field("query", pa.string(), nullable=False),
        pa.field("region_id", pa.int32(), nullable=True),
        pa.field("crypta_id_v2", pa.uint64(), nullable=True),
        pa.field("device", pa.string(), nullable=True),
        pa.field("age", pa.int32(), nullable=True),
        pa.field("gender", pa.int32(), nullable=True),
        pa.field("clicked_banner_ids", pa.list_(pa.uint64()), nullable=False),
        pa.field("clicked_source_costs", pa.list_(pa.float64()), nullable=False),
    ]
)


def stable_partition(request_id: str, count: int) -> int:
    digest = hashlib.sha1(request_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % count


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def load_validation_requests(path: Path) -> list[dict[str, Any]]:
    columns = [
        "SearchReqId",
        "HitLogID",
        "ShowTime",
        "SearchQuery",
        "RegionID",
        "CryptaIDv2",
        "DetailedDeviceType",
        "Age",
        "Gender",
        "BannerID",
        "SourceCost",
        "IsClick",
    ]
    data = pq.read_table(path, columns=columns).to_pydict()
    groups: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row_index in range(len(data["SearchReqId"])):
        if int(data["IsClick"][row_index] or 0) != 1:
            continue
        request_id = _text(data["SearchReqId"][row_index])
        show_time = int(data["ShowTime"][row_index])
        group = groups.get(request_id)
        if group is None:
            group = {
                "request_id": request_id,
                "hit_log_id": int(data["HitLogID"][row_index]),
                "show_time": show_time,
                "show_time_min": show_time,
                "show_time_max": show_time,
                "query": _text(data["SearchQuery"][row_index]),
                "region_id": data["RegionID"][row_index],
                "crypta_id_v2": data["CryptaIDv2"][row_index],
                "device": _text(data["DetailedDeviceType"][row_index]) or None,
                "age": data["Age"][row_index],
                "gender": data["Gender"][row_index],
                "clicked_banner_ids": [],
                "clicked_source_costs": [],
            }
            groups[request_id] = group
        group["show_time_min"] = min(group["show_time_min"], show_time)
        group["show_time_max"] = max(group["show_time_max"], show_time)
        banner_id = int(data["BannerID"][row_index])
        if banner_id not in group["clicked_banner_ids"]:
            group["clicked_banner_ids"].append(banner_id)
            group["clicked_source_costs"].append(float(data["SourceCost"][row_index] or 0.0))
    return list(groups.values())


def load_test_requests(path: Path) -> list[dict[str, Any]]:
    data = pq.read_table(path).to_pydict()
    rows = []
    for row_index, hit_log_id in enumerate(data["HitLogID"]):
        rows.append(
            {
                "request_id": str(int(hit_log_id)),
                "hit_log_id": int(hit_log_id),
                "show_time": None,
                "query": _text(data["SearchQuery"][row_index]),
                "region_id": data.get("RegionID", [None] * len(data["HitLogID"]))[row_index],
                "crypta_id_v2": data.get("CryptaIDv2", [None] * len(data["HitLogID"]))[row_index],
                "device": _text(
                    data.get("DetailedDeviceType", [None] * len(data["HitLogID"]))[row_index]
                )
                or None,
                "age": data.get("Age", [None] * len(data["HitLogID"]))[row_index],
                "gender": data.get("Gender", [None] * len(data["HitLogID"]))[row_index],
                "clicked_banner_ids": [],
                "clicked_source_costs": [],
            }
        )
    return rows


def temporal_split(
    rows: Iterable[dict[str, Any]],
    *,
    boundary: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fit: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for row in rows:
        minimum = int(row.get("show_time_min", row["show_time"]))
        maximum = int(row.get("show_time_max", row["show_time"]))
        if minimum < boundary <= maximum:
            raise ValueError(f"Request crosses temporal boundary: {row['request_id']}")
        target = fit if maximum < boundary else holdout
        cleaned = {key: value for key, value in row.items() if not key.startswith("show_time_")}
        target.append(cleaned)
    fit.sort(key=lambda item: (int(item["show_time"]), item["request_id"]))
    holdout.sort(key=lambda item: (int(item["show_time"]), item["request_id"]))
    return fit, holdout


def request_table(rows: Iterable[dict[str, Any]]) -> pa.Table:
    materialized = list(rows)
    return pa.Table.from_pylist(materialized, schema=REQUEST_SCHEMA)


def write_request_parquet(path: Path, rows: Iterable[dict[str, Any]]) -> pa.Table:
    table = request_table(rows)
    with atomic_output_path(path) as temporary:
        pq.write_table(table, temporary, compression="zstd")
    return table


def read_request_parquet(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def request_example(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": row["request_id"],
        "show_time": row.get("show_time"),
        "query": row["query"],
        "context": {
            "region_id": row.get("region_id"),
            "crypta_id_v2": row.get("crypta_id_v2"),
            "device": row.get("device"),
            "age": row.get("age"),
            "gender": row.get("gender"),
        },
    }
