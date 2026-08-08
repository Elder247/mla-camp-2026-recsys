from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig

from .candidate_cache import feature_name
from .counters import CounterLookup, counter_feature_values, url_domain
from .features import extract_feature_rows, feature_names
from .training_data import attach_targets, freeze_natural_pool


def feature_aliases(cfg: DictConfig) -> list[str]:
    return [
        feature_name(cfg, str(source))
        for source, item in cfg.candidates.generators.items()
        if bool(item.get("enabled", False))
    ]


def configured_feature_names(cfg: DictConfig) -> list[str]:
    return feature_names(
        feature_aliases(cfg),
        version=str(cfg.features.version),
        enabled_groups=[str(value) for value in cfg.features.enabled],
        counter_families=[
            str(value) for value in cfg.features.get("counter_families", [])
        ],
        counter_windows_days=[
            int(value) for value in cfg.features.get("counter_windows_days", [])
        ],
    )


def feature_schema(cfg: DictConfig) -> pa.Schema:
    fields = [
        pa.field("request_id", pa.string(), nullable=False),
        pa.field("group_id", pa.uint64(), nullable=False),
        pa.field("hit_log_id", pa.uint64(), nullable=False),
        pa.field("banner_id", pa.uint64(), nullable=False),
        pa.field("pre_rank", pa.int32(), nullable=False),
        pa.field("is_positive", pa.bool_(), nullable=False),
        pa.field("group_has_positive", pa.bool_(), nullable=False),
        pa.field("target_source_cost", pa.float64(), nullable=False),
        pa.field("label_binary", pa.float32(), nullable=False),
        pa.field("label_logsc", pa.float32(), nullable=False),
        pa.field("label_raw_sc", pa.float64(), nullable=False),
    ]
    fields.extend(
        pa.field(name, pa.float32(), nullable=False)
        for name in configured_feature_names(cfg)
    )
    return pa.schema(fields)


def stable_group_id(request_id: str) -> int:
    return int.from_bytes(hashlib.sha1(request_id.encode("utf-8")).digest()[:8], "little")


class BannerIndex:
    def __init__(self, path: Path) -> None:
        table = pq.read_table(
            path,
            columns=[
                "BannerID",
                "BannerTitle",
                "BannerText",
                "BannerURL",
                "SourceCost",
                "ProductPrice",
                "GroupExportID",
                "ClientID",
            ],
        ).combine_chunks()
        ids = table["BannerID"].to_numpy(zero_copy_only=False)
        self.positions = {int(value): index for index, value in enumerate(ids)}
        self.title = table["BannerTitle"].chunk(0)
        self.text = table["BannerText"].chunk(0)
        self.url = table["BannerURL"].chunk(0)
        self.source_cost = table["SourceCost"].chunk(0)
        self.product_price = table["ProductPrice"].chunk(0)
        self.group_id = table["GroupExportID"].chunk(0)
        self.client_id = table["ClientID"].chunk(0)
        self.domains: list[str] = []
        group_stats: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
        domain_stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for group, url, source_cost in zip(
            self.group_id.to_pylist(),
            self.url.to_pylist(),
            self.source_cost.to_pylist(),
        ):
            domain = url_domain(url)
            self.domains.append(domain)
            cost = float(source_cost or 0.0)
            if group is not None:
                stats = group_stats[int(group)]
                stats[0] += 1.0
                stats[1] += cost
            if domain:
                stats = domain_stats[domain]
                stats[0] += 1.0
                stats[1] += cost
        self.group_stats = dict(group_stats)
        self.domain_stats = dict(domain_stats)

    def get(self, banner_id: int) -> dict[str, Any]:
        position = self.positions.get(int(banner_id))
        if position is None:
            return {
                "title": "",
                "text": "",
                "url": "",
                "source_cost": None,
                "product_price": None,
                "group_id": None,
                "client_id": None,
                "domain": "",
                "group_banner_count": 0,
                "domain_banner_count": 0,
                "group_source_cost_mean": 0.0,
                "domain_source_cost_mean": 0.0,
            }
        group = self.group_id[position].as_py()
        group = int(group) if group is not None else None
        domain = self.domains[position]
        group_count, group_cost = self.group_stats.get(group, (0.0, 0.0))
        domain_count, domain_cost = self.domain_stats.get(domain, (0.0, 0.0))
        return {
            "title": self.title[position].as_py() or "",
            "text": self.text[position].as_py() or "",
            "url": self.url[position].as_py() or "",
            "source_cost": (
                float(self.source_cost[position].as_py())
                if self.source_cost[position].as_py() is not None
                else None
            ),
            "product_price": (
                float(self.product_price[position].as_py())
                if self.product_price[position].as_py() is not None
                else None
            ),
            "group_id": group,
            "client_id": self.client_id[position].as_py(),
            "domain": domain,
            "group_banner_count": int(group_count),
            "domain_banner_count": int(domain_count),
            "group_source_cost_mean": group_cost / group_count if group_count else 0.0,
            "domain_source_cost_mean": domain_cost / domain_count if domain_count else 0.0,
        }


def _candidate_from_merged(
    row: dict[str, Any], aliases: list[str], index: BannerIndex
) -> dict[str, Any]:
    banner_id = int(row["banner_id"])
    candidate = {
        "banner_id": banner_id,
        "rrf_score": float(row["rrf_score"]),
        "source_count": int(row["source_count"]),
        "retrieval": {},
        **index.get(banner_id),
    }
    for alias in aliases:
        if not row[f"{alias}__present"]:
            continue
        contributions = None
        if alias == "history":
            contributions = {
                "history": {
                    name: {}
                    for name, present in (
                        ("query", row["history_query_present"]),
                        ("query_region", row["history_region_present"]),
                    )
                    if present
                },
                "click_count": int(row["history_click_count"]),
                "source_cost_sum": float(row["history_source_cost_sum"]),
            }
        rank = int(row[f"{alias}__rank"])
        candidate["retrieval"][alias] = {
            "rank": rank,
            "reciprocal_rank": 1.0 / rank,
            "score": float(row[f"{alias}__score"]),
            "contributions": contributions,
        }
    return candidate


def build_feature_partition(
    *,
    cfg: DictConfig,
    merged_path: Path,
    requests: dict[str, dict[str, Any]],
    banner_index: BannerIndex,
    counter_lookup: CounterLookup | None = None,
    frozen_counter_cutoff: int | None = None,
) -> tuple[pa.Table, dict[str, int]]:
    merged = pq.read_table(merged_path).to_pylist()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in merged:
        grouped.setdefault(str(row["request_id"]), []).append(row)
    aliases = feature_aliases(cfg)
    names = configured_feature_names(cfg)
    output: list[dict[str, Any]] = []
    groups = 0
    positive_groups = 0
    missed_positive_groups = 0
    for request_id, rows in grouped.items():
        request = requests[request_id]
        natural = freeze_natural_pool(rows, int(cfg.candidates.ranker_pool))
        clicked = {
            int(banner_id): float(source_cost)
            for banner_id, source_cost in zip(
                request["clicked_banner_ids"], request["clicked_source_costs"]
            )
        }
        labeled = attach_targets(natural, clicked)
        has_positive = any(row["is_positive"] for row in labeled)
        groups += 1
        positive_groups += int(has_positive)
        missed_positive_groups += int(bool(clicked) and not has_positive)
        candidates = [
            _candidate_from_merged(row, aliases, banner_index) for row in labeled
        ]
        example = {
            "query": request["query"],
            "show_time": request.get("show_time"),
            "context": {
                "region_id": request.get("region_id"),
                "crypta_id_v2": request.get("crypta_id_v2"),
                "device": request.get("device"),
                "age": request.get("age"),
                "gender": request.get("gender"),
            },
        }
        if counter_lookup is not None:
            if frozen_counter_cutoff is None:
                raise ValueError("frozen_counter_cutoff is required with counter_lookup")
            counter_request = {"query": request["query"], **example["context"]}
            counter_request["show_time"] = request.get("show_time")
            for candidate in candidates:
                candidate["counter_features"] = counter_feature_values(
                    counter_lookup,
                    counter_request,
                    candidate,
                    families=[
                        str(value) for value in cfg.features.counter_families
                    ],
                    windows_days=[
                        int(value) for value in cfg.features.counter_windows_days
                    ],
                    frozen_cutoff=int(frozen_counter_cutoff),
                )
        matrix = extract_feature_rows(
            example,
            candidates,
            aliases,
            version=str(cfg.features.version),
            enabled_groups=[str(value) for value in cfg.features.enabled],
            counter_families=[
                str(value) for value in cfg.features.get("counter_families", [])
            ],
            counter_windows_days=[
                int(value) for value in cfg.features.get("counter_windows_days", [])
            ],
        )
        group_id = stable_group_id(request_id)
        for labeled_row, values in zip(labeled, matrix):
            source_cost = float(labeled_row["target_source_cost"])
            is_positive = bool(labeled_row["is_positive"])
            row = {
                "request_id": request_id,
                "group_id": group_id,
                "hit_log_id": int(request["hit_log_id"]),
                "banner_id": int(labeled_row["banner_id"]),
                "pre_rank": int(labeled_row["pre_rank"]),
                "is_positive": is_positive,
                "group_has_positive": has_positive,
                "target_source_cost": source_cost,
                "label_binary": float(is_positive),
                "label_logsc": float(
                    1.0
                    + math.log1p(
                        max(0.0, source_cost) / float(cfg.ranker.log_sc_scale)
                    )
                    if is_positive
                    else 0.0
                ),
                "label_raw_sc": source_cost if is_positive else 0.0,
            }
            row.update((name, float(value)) for name, value in zip(names, values))
            output.append(row)
    stats = {
        "groups": groups,
        "positive_groups": positive_groups,
        "missed_positive_groups": missed_positive_groups,
        "rows": len(output),
    }
    return pa.Table.from_pylist(output, schema=feature_schema(cfg)), stats
