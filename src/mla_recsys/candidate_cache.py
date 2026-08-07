from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig

from .artifacts import (
    fingerprint_file,
    make_cache_key,
    validate_output_cache,
    write_output_manifest,
)
from .data import request_example, stable_partition
from .loading import load_module
from .pipeline import Generator


SOURCE_SCHEMA = pa.schema(
    [
        pa.field("request_id", pa.string(), nullable=False),
        pa.field("hit_log_id", pa.uint64(), nullable=False),
        pa.field("banner_id", pa.uint64(), nullable=False),
        pa.field("source_rank", pa.int32(), nullable=False),
        pa.field("source_score", pa.float64(), nullable=False),
        pa.field("history_click_count", pa.int64(), nullable=False),
        pa.field("history_source_cost_sum", pa.float64(), nullable=False),
        pa.field("history_query_present", pa.bool_(), nullable=False),
        pa.field("history_region_present", pa.bool_(), nullable=False),
    ]
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    feature_name: str
    generator: Generator
    code_path: Path
    artifact_dir: Path


def enabled_sources(cfg: DictConfig) -> list[str]:
    return [
        str(name)
        for name, item in cfg.candidates.generators.items()
        if bool(item.get("enabled", False))
    ]


def feature_name(cfg: DictConfig, source: str) -> str:
    return str(cfg.candidates.generators[source].get("feature_name", source))


def load_source(cfg: DictConfig, source: str) -> SourceSpec:
    item = cfg.candidates.generators[source]
    if not item.get("code_path_key") or not item.get("artifact_path_key"):
        raise ValueError(f"Candidate generator {source} has no implementation paths")
    code_path = Path(str(cfg.paths[str(item.code_path_key)]))
    artifact_dir = Path(str(cfg.paths[str(item.artifact_path_key)]))
    python_paths = [
        str(cfg.paths[str(key)]) for key in list(item.get("python_path_keys") or [])
    ]
    module = load_module(code_path, python_paths)
    model = module.load_model(artifact_dir)
    defaults = {}
    if hasattr(module, "feature_schema"):
        defaults = {
            str(value["name"]): value.get("default")
            for value in module.feature_schema()
            if "name" in value and "default" in value
        }
    generator = Generator(
        name=source,
        module=module,
        model=model,
        top_k=int(item.top_k),
        quota=int(item.quota),
        weight=float(item.weight),
        features={**defaults, **dict(item.get("features") or {})},
    )
    return SourceSpec(
        name=source,
        feature_name=feature_name(cfg, source),
        generator=generator,
        code_path=code_path,
        artifact_dir=artifact_dir,
    )


def source_input_fingerprints(spec: SourceSpec, request_path: Path) -> list[dict[str, Any]]:
    inputs = [fingerprint_file(request_path), fingerprint_file(spec.code_path)]
    if spec.artifact_dir.is_dir():
        for path in sorted(spec.artifact_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in {
                ".json",
                ".npy",
                ".parquet",
                ".pt",
                ".gz",
            }:
                inputs.append(fingerprint_file(path))
    return inputs


def source_part_path(run_path: Path, split: str, source: str, partition: int) -> Path:
    return (
        run_path
        / "candidates"
        / split
        / source
        / f"part-{partition:05d}.parquet"
    )


class CandidatePartitionSink:
    def __init__(
        self,
        *,
        run_path: Path,
        split: str,
        source: str,
        partitions: int,
        buffer_rows: int,
    ) -> None:
        self.paths = [source_part_path(run_path, split, source, index) for index in range(partitions)]
        self.buffer_rows = buffer_rows
        self.buffers: list[list[dict[str, Any]]] = [[] for _ in self.paths]
        self.writers: list[pq.ParquetWriter | None] = [None for _ in self.paths]
        self.temporary: list[Path] = []
        self.rows = [0 for _ in self.paths]
        for path in self.paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            os.close(fd)
            self.temporary.append(Path(raw))

    def append(self, partition: int, row: dict[str, Any]) -> None:
        self.buffers[partition].append(row)
        if len(self.buffers[partition]) >= self.buffer_rows:
            self._flush(partition)

    def _flush(self, partition: int) -> None:
        rows = self.buffers[partition]
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=SOURCE_SCHEMA)
        writer = self.writers[partition]
        if writer is None:
            writer = pq.ParquetWriter(
                self.temporary[partition], SOURCE_SCHEMA, compression="zstd"
            )
            self.writers[partition] = writer
        writer.write_table(table)
        self.rows[partition] += table.num_rows
        rows.clear()

    def close(self) -> None:
        for partition, path in enumerate(self.paths):
            self._flush(partition)
            writer = self.writers[partition]
            if writer is not None:
                writer.close()
            else:
                pq.write_table(
                    pa.Table.from_pylist([], schema=SOURCE_SCHEMA),
                    self.temporary[partition],
                    compression="zstd",
                )
            os.replace(self.temporary[partition], path)

    def abort(self) -> None:
        for writer in self.writers:
            if writer is not None:
                writer.close()
        for path in self.temporary:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def candidate_row(
    request: dict[str, Any], raw: dict[str, Any], source_rank: int
) -> dict[str, Any]:
    contributions = raw.get("contributions") or {}
    history_sources = contributions.get("history") or {}
    return {
        "request_id": str(request["request_id"]),
        "hit_log_id": int(request["hit_log_id"]),
        "banner_id": int(raw["banner_id"]),
        "source_rank": int(source_rank),
        "source_score": float(raw.get("score") or 0.0),
        "history_click_count": int(contributions.get("click_count") or 0),
        "history_source_cost_sum": float(contributions.get("source_cost_sum") or 0.0),
        "history_query_present": "query" in history_sources,
        "history_region_present": "query_region" in history_sources,
    }


def generate_source_candidates(
    *,
    spec: SourceSpec,
    requests: Iterable[dict[str, Any]],
    run_path: Path,
    split: str,
    partitions: int,
    buffer_rows: int,
) -> dict[str, Any]:
    sink = CandidatePartitionSink(
        run_path=run_path,
        split=split,
        source=spec.name,
        partitions=partitions,
        buffer_rows=buffer_rows,
    )
    started = time.monotonic()
    request_count = 0
    covered = 0
    try:
        for request in requests:
            request_count += 1
            ranking = spec.generator.rank(request_example(request))
            seen: set[int] = set()
            accepted = 0
            partition = stable_partition(str(request["request_id"]), partitions)
            for source_rank, raw in enumerate(ranking, start=1):
                banner_id = int(raw["banner_id"])
                if banner_id in seen:
                    continue
                seen.add(banner_id)
                if accepted >= spec.generator.quota:
                    break
                accepted += 1
                sink.append(partition, candidate_row(request, raw, source_rank))
            covered += int(accepted > 0)
        sink.close()
    except BaseException:
        sink.abort()
        raise
    return {
        "source": spec.name,
        "feature_name": spec.feature_name,
        "split": split,
        "requests": request_count,
        "covered_requests": covered,
        "coverage": covered / request_count if request_count else 0.0,
        "rows": sum(sink.rows),
        "partition_rows": sink.rows,
        "wall_seconds": time.monotonic() - started,
    }


def cache_is_valid(
    *,
    run_path: Path,
    split: str,
    source: str,
    partitions: int,
    cache_key: str,
) -> bool:
    return all(
        validate_output_cache(
            source_part_path(run_path, split, source, partition),
            expected_cache_key=cache_key,
            expected_schema=str(SOURCE_SCHEMA),
        )[0]
        for partition in range(partitions)
    )


def finalize_source_manifests(
    *,
    run_path: Path,
    split: str,
    source: str,
    partitions: int,
    rows: list[int],
    artifact_version: str,
    config_sha256: str,
    inputs: list[dict[str, Any]],
    scope: str,
) -> None:
    for partition in range(partitions):
        write_output_manifest(
            source_part_path(run_path, split, source, partition),
            stage=f"generate_candidates_{source}_{split}",
            artifact_version=artifact_version,
            config_sha256=config_sha256,
            inputs=inputs,
            rows=rows[partition],
            schema=str(SOURCE_SCHEMA),
            scope=scope,
        )

