from __future__ import annotations

import heapq
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from omegaconf import DictConfig
import pyarrow.parquet as pq

from .artifacts import fingerprint_file
from .candidate_cache import CandidatePartitionSink, candidate_row
from .counters import scalar_key, stable_text_key
from .data import read_request_parquet, stable_partition


TEMPORAL_SOURCE_KIND = "temporal_history"
TEMPORAL_SOURCES = {
    "history_user_v1": "user",
    "region_pop_sc_v1": "region",
    "global_pop_sc_v1": "global",
    "history_query_sc_oof_v1": "query_sc",
    "history_query_region_oof_v1": "query_region_sc",
}


@dataclass
class BannerStats:
    clicks: int = 0
    source_cost_sum: float = 0.0
    last_show_time: int = 0


class TemporalHistoryState:
    def __init__(
        self,
        source: str,
        *,
        min_clicks: int,
        bayes_prior: float,
        valid_banner_ids: set[int] | None = None,
    ) -> None:
        if source not in TEMPORAL_SOURCES:
            raise ValueError(f"Unsupported temporal candidate source: {source}")
        self.source = source
        self.family = TEMPORAL_SOURCES[source]
        self.min_clicks = int(min_clicks)
        self.bayes_prior = float(bayes_prior)
        self.valid_banner_ids = valid_banner_ids
        self.stats: dict[str, dict[int, BannerStats]] = defaultdict(dict)
        self.dirty: dict[str, set[int]] = defaultdict(set)
        self.ranking_cache: dict[str, tuple[int, list[int]]] = {}

    def _key(self, request: dict[str, Any]) -> str | None:
        query = str(request.get("query_key") or stable_text_key(request.get("query")))
        region = str(request.get("region_key") or scalar_key(request.get("region_id")))
        if self.family == "query_sc":
            return query or None
        if self.family == "query_region_sc":
            return f"{query}|{region}" if query and region else None
        if self.family == "user":
            value = request.get("user_key") or request.get("crypta_id_v2")
            return str(value) if value not in (None, 0, "", "0") else None
        if self.family == "region":
            value = request.get("region_key") or request.get("region_id")
            return str(value) if value not in (None, "") else None
        return "global"

    def observe(self, request: dict[str, Any]) -> None:
        key = self._key(request)
        if key is None:
            return
        show_time = int(request.get("show_time") or 0)
        by_banner = self.stats[key]
        if request.get("banner_id") is not None:
            values = [(request["banner_id"], request.get("source_cost") or 0.0)]
        else:
            values = zip(
                request.get("clicked_banner_ids") or (),
                request.get("clicked_source_costs") or (),
            )
        for banner_id, source_cost in values:
            normalized_banner_id = int(banner_id)
            if (
                self.valid_banner_ids is not None
                and normalized_banner_id not in self.valid_banner_ids
            ):
                continue
            item = by_banner.setdefault(normalized_banner_id, BannerStats())
            item.clicks += 1
            item.source_cost_sum += float(source_cost or 0.0)
            item.last_show_time = max(item.last_show_time, show_time)
            self.dirty[key].add(normalized_banner_id)

    def _score(self, item: BannerStats) -> float:
        if self.family in {"user", "query_sc", "query_region_sc"}:
            return item.source_cost_sum
        support = item.clicks / (item.clicks + self.bayes_prior)
        return item.source_cost_sum * support

    def rank(self, request: dict[str, Any], *, top_k: int) -> list[dict[str, Any]]:
        key = self._key(request)
        if key is None or top_k <= 0:
            return []
        cached_width, cached_ids = self.ranking_cache.get(key, (0, []))
        cache_width = max(cached_width, top_k)
        by_banner = self.stats.get(key, {})
        if cached_width < top_k:
            candidate_ids = set(by_banner)
        else:
            candidate_ids = set(cached_ids)
            candidate_ids.update(self.dirty.get(key, ()))
        ranked = []
        for banner_id in candidate_ids:
            item = by_banner[banner_id]
            if item.clicks < self.min_clicks:
                continue
            score = self._score(item)
            ranked.append((score, item.clicks, item.last_show_time, banner_id, item))
        ranked.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
        self.ranking_cache[key] = (
            cache_width,
            [row[3] for row in ranked[:cache_width]],
        )
        self.dirty[key].clear()
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
    external = _external_events_path(cfg, source)
    if external is not None:
        inputs.append(fingerprint_file(external))
    banner_index = _banner_index_path(cfg)
    if banner_index is not None:
        inputs.append(fingerprint_file(banner_index))
    inputs.append(fingerprint_file(Path(__file__)))
    return inputs


def _new_state(cfg: DictConfig, source: str) -> TemporalHistoryState:
    item = cfg.candidates.generators[source]
    default_min_clicks = 1 if TEMPORAL_SOURCES[source] == "user" else 2
    banner_index = _banner_index_path(cfg)
    valid_banner_ids = None
    if banner_index is not None:
        table = pq.read_table(banner_index, columns=["BannerID"])
        valid_banner_ids = {
            int(banner_id)
            for banner_id in table.column("BannerID").to_pylist()
            if banner_id is not None
        }
    return TemporalHistoryState(
        source,
        min_clicks=int(item.get("min_clicks", default_min_clicks)),
        bayes_prior=float(item.get("bayes_prior", 0.0)),
        valid_banner_ids=valid_banner_ids,
    )


def _banner_index_path(cfg: DictConfig) -> Path | None:
    if not bool(cfg.candidates.get("temporal_restrict_to_banner_index", True)):
        return None
    paths = cfg.get("paths")
    value = paths.get("banner_index") if paths is not None else None
    return Path(str(value)) if value else None


def _warm_state(
    state: TemporalHistoryState, rows: Iterable[dict[str, Any]]
) -> None:
    for request in sorted(
        rows, key=lambda row: (int(row.get("show_time") or 0), str(row["request_id"]))
    ):
        state.observe(request)


def _external_events_path(cfg: DictConfig, source: str) -> Path | None:
    item = cfg.candidates.generators[source]
    key = item.get("external_events_path_key")
    return Path(str(cfg.paths[str(key)])) if key else None


def _external_event_rows(path: Path) -> Iterable[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    columns = [
        "show_time",
        "banner_id",
        "query_key",
        "region_key",
        "user_key",
        "source_cost",
    ]
    for batch in parquet.iter_batches(batch_size=100_000, columns=columns):
        values = batch.to_pydict()
        for index in range(batch.num_rows):
            yield {name: values[name][index] for name in columns}


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
    external_path = _external_events_path(cfg, source)
    auxiliary: Iterable[dict[str, Any]] = ()
    if external_path is not None:
        auxiliary = _external_event_rows(external_path)
    if warm_split:
        warm = sorted(
            read_request_parquet(run_path / "data" / f"{warm_split}_requests.parquet"),
            key=lambda row: (int(row.get("show_time") or 0), str(row["request_id"])),
        )
        if external_path is not None:
            warm = [row for row in warm if not str(row["request_id"]).startswith("oof:")]
        auxiliary = heapq.merge(
            auxiliary,
            warm,
            key=lambda row: (int(row.get("show_time") or 0), str(row.get("request_id") or "")),
        )
    ordered = sorted(
        requests, key=lambda row: (int(row.get("show_time") or 0), str(row["request_id"]))
    )
    top_k = int(cfg.candidates.generators[source].top_k)
    result: dict[str, list[dict[str, Any]]] = {}
    auxiliary_iterator = iter(auxiliary)
    current_auxiliary = next(auxiliary_iterator, None)
    frozen_inference = bool(warm_split) and all(
        row.get("show_time") is None for row in ordered
    )
    if frozen_inference:
        # Competition test requests are unlabeled and intentionally have no
        # timestamp. Rank them against one state frozen after every available
        # full-history event; never update that state from test requests.
        while current_auxiliary is not None:
            state.observe(current_auxiliary)
            current_auxiliary = next(auxiliary_iterator, None)
        for request in ordered:
            result[str(request["request_id"])] = state.rank(request, top_k=top_k)
        return result
    position = 0
    while position < len(ordered):
        timestamp = int(ordered[position].get("show_time") or 0)
        while (
            current_auxiliary is not None
            and int(current_auxiliary.get("show_time") or 0) < timestamp
        ):
            state.observe(current_auxiliary)
            current_auxiliary = next(auxiliary_iterator, None)
        end = position + 1
        while end < len(ordered) and int(ordered[end].get("show_time") or 0) == timestamp:
            end += 1
        batch = ordered[position:end]
        for request in batch:
            result[str(request["request_id"])] = state.rank(request, top_k=top_k)
        if not warm_split:
            for request in batch:
                if external_path is None or not str(request["request_id"]).startswith("oof:"):
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
        "index_membership_filter": bool(_banner_index_path(cfg)),
    }
