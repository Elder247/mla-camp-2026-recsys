from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from omegaconf import DictConfig

from .artifacts import fingerprint_file
from .candidate_cache import CandidatePartitionSink, candidate_row
from .data import read_request_parquet, stable_partition


TEMPORAL_SOURCE_KIND = "temporal_history"
TEMPORAL_SOURCES = {
    "history_user_v1": "user",
    "region_pop_sc_v1": "region",
    "global_pop_sc_v1": "global",
}


@dataclass
class BannerStats:
    clicks: int = 0
    source_cost_sum: float = 0.0
    last_show_time: int = 0


class TemporalHistoryState:
    def __init__(self, source: str, *, min_clicks: int, bayes_prior: float) -> None:
        if source not in TEMPORAL_SOURCES:
            raise ValueError(f"Unsupported temporal candidate source: {source}")
        self.source = source
        self.family = TEMPORAL_SOURCES[source]
        self.min_clicks = int(min_clicks)
        self.bayes_prior = float(bayes_prior)
        self.stats: dict[str, dict[int, BannerStats]] = defaultdict(dict)

    def _key(self, request: dict[str, Any]) -> str | None:
        if self.family == "user":
            value = request.get("crypta_id_v2")
            return str(int(value)) if value not in (None, 0) else None
        if self.family == "region":
            value = request.get("region_id")
            return str(int(value)) if value is not None else None
        return "global"

    def observe(self, request: dict[str, Any]) -> None:
        key = self._key(request)
        if key is None:
            return
        show_time = int(request.get("show_time") or 0)
        by_banner = self.stats[key]
        for banner_id, source_cost in zip(
            request.get("clicked_banner_ids") or (),
            request.get("clicked_source_costs") or (),
        ):
            item = by_banner.setdefault(int(banner_id), BannerStats())
            item.clicks += 1
            item.source_cost_sum += float(source_cost or 0.0)
            item.last_show_time = max(item.last_show_time, show_time)

    def rank(self, request: dict[str, Any], *, top_k: int) -> list[dict[str, Any]]:
        key = self._key(request)
        if key is None:
            return []
        ranked = []
        for banner_id, item in self.stats.get(key, {}).items():
            if item.clicks < self.min_clicks:
                continue
            if self.family == "user":
                score = item.source_cost_sum
            else:
                support = item.clicks / (item.clicks + self.bayes_prior)
                score = item.source_cost_sum * support
            ranked.append((score, item.clicks, item.last_show_time, banner_id, item))
        ranked.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
        result = []
        for score, _, _, banner_id, item in ranked[:top_k]:
            result.append(
                {
                    "banner_id": int(banner_id),
                    "score": float(score),
                    "contributions": {
                        "history": {self.family: {}},
                        "click_count": item.clicks,
                        "source_cost_sum": item.source_cost_sum,
                        "last_show_time": item.last_show_time,
                    },
                }
            )
        return result


def is_temporal_source(cfg: DictConfig, source: str) -> bool:
    return str(cfg.candidates.generators[source].get("kind") or "") == TEMPORAL_SOURCE_KIND


def reference_split(split: str) -> str | None:
    return {"holdout": "train", "test": "full_train"}.get(split)


def temporal_source_inputs(
    *, cfg: DictConfig, run_path: Path, split: str, source: str
) -> list[dict[str, Any]]:
    inputs = [fingerprint_file(run_path / "data" / f"{split}_requests.parquet")]
    warm_split = reference_split(split)
    if warm_split:
        inputs.append(fingerprint_file(run_path / "data" / f"{warm_split}_requests.parquet"))
    inputs.append(fingerprint_file(Path(__file__)))
    return inputs


def _new_state(cfg: DictConfig, source: str) -> TemporalHistoryState:
    item = cfg.candidates.generators[source]
    default_min_clicks = 1 if TEMPORAL_SOURCES[source] == "user" else 2
    return TemporalHistoryState(
        source,
        min_clicks=int(item.get("min_clicks", default_min_clicks)),
        bayes_prior=float(item.get("bayes_prior", 0.0)),
    )


def _warm_state(
    state: TemporalHistoryState, rows: Iterable[dict[str, Any]]
) -> None:
    for request in sorted(
        rows, key=lambda row: (int(row.get("show_time") or 0), str(row["request_id"]))
    ):
        state.observe(request)


def temporal_rankings(
    *,
    cfg: DictConfig,
    run_path: Path,
    split: str,
    source: str,
    requests: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    state = _new_state(cfg, source)
    warm_split = reference_split(split)
    if warm_split:
        _warm_state(
            state,
            read_request_parquet(run_path / "data" / f"{warm_split}_requests.parquet"),
        )
    ordered = sorted(
        requests, key=lambda row: (int(row.get("show_time") or 0), str(row["request_id"]))
    )
    top_k = int(cfg.candidates.generators[source].top_k)
    result: dict[str, list[dict[str, Any]]] = {}
    if warm_split:
        for request in ordered:
            result[str(request["request_id"])] = state.rank(request, top_k=top_k)
        return result

    position = 0
    while position < len(ordered):
        timestamp = int(ordered[position].get("show_time") or 0)
        end = position + 1
        while end < len(ordered) and int(ordered[end].get("show_time") or 0) == timestamp:
            end += 1
        batch = ordered[position:end]
        for request in batch:
            result[str(request["request_id"])] = state.rank(request, top_k=top_k)
        for request in batch:
            state.observe(request)
        position = end
    return result


def generate_temporal_source_candidates(
    *,
    cfg: DictConfig,
    run_path: Path,
    split: str,
    source: str,
    requests: list[dict[str, Any]],
    partitions: int,
    buffer_rows: int,
) -> dict[str, Any]:
    sink = CandidatePartitionSink(
        run_path=run_path,
        split=split,
        source=source,
        partitions=partitions,
        buffer_rows=buffer_rows,
    )
    started = time.monotonic()
    covered = 0
    quota = int(cfg.candidates.generators[source].quota)
    try:
        rankings = temporal_rankings(
            cfg=cfg,
            run_path=run_path,
            split=split,
            source=source,
            requests=requests,
        )
        for request in requests:
            request_id = str(request["request_id"])
            partition = stable_partition(request_id, partitions)
            rows = rankings.get(request_id, ())[:quota]
            covered += int(bool(rows))
            for source_rank, raw in enumerate(rows, start=1):
                sink.append(partition, candidate_row(request, raw, source_rank))
        sink.close()
    except BaseException:
        sink.abort()
        raise
    return {
        "source": source,
        "feature_name": str(cfg.candidates.generators[source].get("feature_name") or source),
        "split": split,
        "requests": len(requests),
        "covered_requests": covered,
        "coverage": covered / len(requests) if requests else 0.0,
        "rows": sum(sink.rows),
        "partition_rows": sink.rows,
        "wall_seconds": time.monotonic() - started,
        "history_semantics": (
            f"frozen_after_{reference_split(split)}"
            if reference_split(split)
            else "strictly_prior_timestamp_batches"
        ),
    }
