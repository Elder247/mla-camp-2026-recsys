#!/usr/bin/env python3
"""Train a leakage-safe residual DCNv2 ranker on cached natural-pool features."""

from __future__ import annotations

import argparse
import copy
import json
import math
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import (  # noqa: E402
    atomic_output_path,
    atomic_write_json,
    fingerprint_file,
    write_output_manifest,
)
from mla_recsys.data import read_request_parquet  # noqa: E402
from mla_recsys.dcn_ranker import (  # noqa: E402
    DCNv2Ranker,
    sample_listwise_groups,
    stable_hash,
)
from mla_recsys.text import normalize  # noqa: E402


BUCKETS = [2**18, 2**18, 2**19, 2**20, 2**20, 2**18]
EMBEDDING_DIMS = [12, 8, 12, 8, 8, 6]


@dataclass
class SplitArrays:
    features: np.ndarray
    group_ids: np.ndarray
    hit_log_ids: np.ndarray
    banner_ids: np.ndarray
    pre_ranks: np.ndarray
    source_costs: np.ndarray
    categorical: np.ndarray


def feature_paths(run: Path, split: str) -> list[Path]:
    paths = sorted((run / "features" / split).glob("part-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No feature parts for {split}: {run}")
    return paths


def request_context(run: Path, split: str) -> tuple[dict[str, tuple[int, int, int]], list[dict]]:
    requests = read_request_parquet(run / "data" / f"{split}_requests.parquet")
    context = {}
    for row in requests:
        request_id = str(row["request_id"])
        query_hash = stable_hash(normalize(str(row.get("query") or "")), BUCKETS[0])
        user_hash = stable_hash(int(row.get("crypta_id_v2") or 0), BUCKETS[1])
        context[request_id] = (
            query_hash,
            user_hash,
            int(row.get("region_id") or 0),
        )
    return context, requests


def load_split(
    run: Path,
    split: str,
    feature_names: list[str],
    context: dict[str, tuple[int, int, int]],
) -> SplitArrays:
    paths = feature_paths(run, split)
    total = sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
    features = np.empty((total, len(feature_names)), dtype=np.float32)
    group_ids = np.empty(total, dtype=np.uint64)
    hit_log_ids = np.empty(total, dtype=np.uint64)
    banner_ids = np.empty(total, dtype=np.uint64)
    pre_ranks = np.empty(total, dtype=np.int32)
    source_costs = np.empty(total, dtype=np.float64)
    categorical = np.empty((total, len(BUCKETS)), dtype=np.int64)
    columns = [
        "request_id",
        "group_id",
        "hit_log_id",
        "banner_id",
        "pre_rank",
        "label_raw_sc",
        *feature_names,
    ]
    offset = 0
    for path in paths:
        table = pq.read_table(path, columns=columns)
        size = table.num_rows
        end = offset + size
        matrix = np.column_stack(
            [
                table[name].combine_chunks().to_numpy(zero_copy_only=False)
                for name in feature_names
            ]
        ).astype(np.float32, copy=False)
        np.nan_to_num(matrix, copy=False, nan=0.0, posinf=1.0e12, neginf=-1.0e12)
        features[offset:end] = matrix
        group_ids[offset:end] = table["group_id"].to_numpy(zero_copy_only=False)
        hit_log_ids[offset:end] = table["hit_log_id"].to_numpy(zero_copy_only=False)
        banners = table["banner_id"].to_numpy(zero_copy_only=False).astype(
            np.uint64, copy=False
        )
        banner_ids[offset:end] = banners
        pre_ranks[offset:end] = table["pre_rank"].to_numpy(zero_copy_only=False)
        source_costs[offset:end] = table["label_raw_sc"].to_numpy(
            zero_copy_only=False
        )
        request_ids = table["request_id"].to_pylist()
        query_hash = np.fromiter(
            (context[str(value)][0] for value in request_ids),
            dtype=np.int64,
            count=size,
        )
        user_hash = np.fromiter(
            (context[str(value)][1] for value in request_ids),
            dtype=np.int64,
            count=size,
        )
        regions = np.fromiter(
            (context[str(value)][2] for value in request_ids),
            dtype=np.int64,
            count=size,
        )
        banner_hash = (banners % BUCKETS[2]).astype(np.int64)
        query_banner = (
            (query_hash.astype(np.uint64) * np.uint64(0x9E3779B185EBCA87))
            ^ banners
        ) % np.uint64(BUCKETS[3])
        user_banner = (
            (user_hash.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F))
            ^ banners
        ) % np.uint64(BUCKETS[4])
        group_values = matrix[:, feature_names.index("group_hash_bucket")].astype(
            np.int64, copy=False
        )
        region_group = (
            (regions.astype(np.uint64) * np.uint64(0x165667B19E3779F9))
            ^ group_values.astype(np.uint64)
        ) % np.uint64(BUCKETS[5])
        categorical[offset:end] = np.column_stack(
            [
                query_hash,
                user_hash,
                banner_hash,
                query_banner.astype(np.int64),
                user_banner.astype(np.int64),
                region_group.astype(np.int64),
            ]
        )
        offset = end
    if offset != total:
        raise AssertionError("feature row preallocation mismatch")
    starts = np.r_[0, np.flatnonzero(group_ids[1:] != group_ids[:-1]) + 1]
    if np.unique(group_ids[starts]).size != starts.size:
        raise ValueError(f"{split} ranking groups are not contiguous")
    return SplitArrays(
        features=features,
        group_ids=group_ids,
        hit_log_ids=hit_log_ids,
        banner_ids=banner_ids,
        pre_ranks=pre_ranks,
        source_costs=source_costs,
        categorical=categorical,
    )


def normalize_features(
    train: SplitArrays,
    holdout: SplitArrays,
    feature_names: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    mean = train.features.mean(axis=0, dtype=np.float64)
    variance = np.maximum(
        np.square(train.features, dtype=np.float64).mean(axis=0) - np.square(mean),
        0.0,
    )
    standard_deviation = np.sqrt(variance)
    selected = np.isfinite(standard_deviation) & (standard_deviation > 1.0e-6)
    if not selected.any():
        raise ValueError("Every continuous feature is constant")
    mean = mean[selected].astype(np.float32)
    standard_deviation = standard_deviation[selected].astype(np.float32)
    train.features = train.features[:, selected]
    holdout.features = holdout.features[:, selected]
    for values in (train.features, holdout.features):
        values -= mean
        values /= standard_deviation
        np.clip(values, -8.0, 8.0, out=values)
    return (
        [name for name, keep in zip(feature_names, selected) if keep],
        mean,
        standard_deviation,
    )


def base_scores(pre_ranks: np.ndarray) -> np.ndarray:
    return (-0.5 * np.log1p(pre_ranks.astype(np.float32))).astype(np.float32)


def score_model(
    model: DCNv2Ranker,
    values: SplitArrays,
    *,
    device: torch.device,
    batch_rows: int,
) -> np.ndarray:
    model.eval()
    result = np.empty(len(values.pre_ranks), dtype=np.float32)
    bases = base_scores(values.pre_ranks)
    with torch.inference_mode():
        for start in range(0, len(result), batch_rows):
            end = min(start + batch_rows, len(result))
            continuous = torch.from_numpy(values.features[start:end]).to(
                device, non_blocking=True
            )
            categorical = torch.from_numpy(values.categorical[start:end]).to(
                device, non_blocking=True
            )
            base = torch.from_numpy(bases[start:end]).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                scores = model(continuous, categorical, base)
            result[start:end] = scores.float().cpu().numpy()
    return result


def truth_by_hit(requests: list[dict]) -> dict[int, dict[int, float]]:
    truth: dict[int, dict[int, float]] = {}
    for request in requests:
        hit_log_id = int(request["hit_log_id"])
        truth[hit_log_id] = {
            int(banner_id): float(source_cost)
            for banner_id, source_cost in zip(
                request.get("clicked_banner_ids") or (),
                request.get("clicked_source_costs") or (),
            )
        }
    return truth


def ranking_metrics(
    values: SplitArrays,
    scores: np.ndarray,
    truth: dict[int, dict[int, float]],
    allowed: set[int],
) -> dict[str, dict[str, float | int]]:
    cutoffs = (10, 50, 100, 500)
    source_total = sum(
        cost
        for hit_log_id, targets in truth.items()
        if hit_log_id in allowed
        for cost in targets.values()
    )
    click_total = sum(
        len(targets) for hit_log_id, targets in truth.items() if hit_log_id in allowed
    )
    source_hits = {cutoff: 0.0 for cutoff in cutoffs}
    click_hits = {cutoff: 0 for cutoff in cutoffs}
    starts = np.r_[
        0, np.flatnonzero(values.group_ids[1:] != values.group_ids[:-1]) + 1
    ]
    ends = np.r_[starts[1:], len(values.group_ids)]
    for start, end in zip(starts, ends):
        hit_log_id = int(values.hit_log_ids[start])
        if hit_log_id not in allowed:
            continue
        targets = truth.get(hit_log_id, {})
        if not targets:
            continue
        order = np.lexsort(
            (
                values.banner_ids[start:end],
                values.pre_ranks[start:end],
                -scores[start:end],
            )
        )
        ranked = values.banner_ids[start:end][order]
        positions = {int(banner_id): rank for rank, banner_id in enumerate(ranked, 1)}
        for banner_id, cost in targets.items():
            rank = positions.get(banner_id, 10**9)
            for cutoff in cutoffs:
                if rank <= cutoff:
                    source_hits[cutoff] += cost
                    click_hits[cutoff] += 1
    return {
        str(cutoff): {
            "sourcecost_recall": source_hits[cutoff] / source_total,
            "recall": click_hits[cutoff] / click_total,
            "sourcecost_hit": source_hits[cutoff],
            "sourcecost_total": source_total,
            "hits": click_hits[cutoff],
            "clicks": click_total,
        }
        for cutoff in cutoffs
    }


def temporal_metrics(
    values: SplitArrays,
    scores: np.ndarray,
    requests: list[dict],
) -> dict[str, dict[str, dict[str, float | int]]]:
    ordered = sorted(
        requests,
        key=lambda row: (int(row.get("show_time") or 0), str(row["request_id"])),
    )
    split = len(ordered) // 2
    early = {int(row["hit_log_id"]) for row in ordered[:split]}
    late = {int(row["hit_log_id"]) for row in ordered[split:]}
    truth = truth_by_hit(requests)
    return {
        "early": ranking_metrics(values, scores, truth, early),
        "late": ranking_metrics(values, scores, truth, late),
        "full": ranking_metrics(values, scores, truth, early | late),
    }


def metric_deltas(candidate: dict, baseline: dict) -> dict:
    return {
        segment: {
            cutoff: {
                metric: float(candidate[segment][cutoff][metric])
                - float(baseline[segment][cutoff][metric])
                for metric in ("sourcecost_recall", "recall")
            }
            for cutoff in candidate[segment]
        }
        for segment in candidate
    }


def export_ranking(
    path: Path,
    values: SplitArrays,
    scores: np.ndarray,
    *,
    top_k: int,
) -> int:
    rows = []
    starts = np.r_[
        0, np.flatnonzero(values.group_ids[1:] != values.group_ids[:-1]) + 1
    ]
    ends = np.r_[starts[1:], len(values.group_ids)]
    for start, end in zip(starts, ends):
        order = np.lexsort(
            (
                values.banner_ids[start:end],
                values.pre_ranks[start:end],
                -scores[start:end],
            )
        )
        rows.append(
            {
                "HitLogID": int(values.hit_log_ids[start]),
                "BannerID": [
                    int(value) for value in values.banner_ids[start:end][order[:top_k]]
                ],
            }
        )
    rows.sort(key=lambda row: int(row["HitLogID"]))
    schema = pa.schema(
        [
            pa.field("HitLogID", pa.uint64(), nullable=False),
            pa.field("BannerID", pa.list_(pa.uint64()), nullable=False),
        ]
    )
    with atomic_output_path(path) as temporary:
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd")
    return len(rows)


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--groups-per-batch", type=int, default=24)
    parser.add_argument("--candidates-per-group", type=int, default=128)
    parser.add_argument("--hard-fraction", type=float, default=0.75)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--score-batch-rows", type=int, default=131072)
    parser.add_argument("--top-k", type=int, default=500)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("DCNv2 screen requires CUDA")
    if args.epochs <= 0 or args.groups_per_batch <= 0:
        parser.error("epochs and groups-per-batch must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "models").mkdir()
    (args.output_dir / "predictions").mkdir()
    (args.output_dir / "metrics").mkdir()

    metadata = json.loads(
        (args.source_run / "models" / "catboost.json").read_text(encoding="utf-8")
    )
    feature_names = list(metadata["feature_names"])
    train_context, train_requests = request_context(args.source_run, "train")
    holdout_context, holdout_requests = request_context(args.source_run, "holdout")
    train_ids = set(train_context)
    holdout_ids = set(holdout_context)
    if train_ids & holdout_ids:
        raise ValueError("train and holdout request IDs overlap")
    train_max_time = max(int(row.get("show_time") or 0) for row in train_requests)
    holdout_min_time = min(int(row.get("show_time") or 0) for row in holdout_requests)
    if train_max_time > holdout_min_time:
        raise ValueError("temporal train extends beyond holdout start")

    print("loading train natural-pool features", flush=True)
    train = load_split(args.source_run, "train", feature_names, train_context)
    print("loading holdout natural-pool features", flush=True)
    holdout = load_split(args.source_run, "holdout", feature_names, holdout_context)
    selected_features, mean, standard_deviation = normalize_features(
        train, holdout, feature_names
    )
    sampled = sample_listwise_groups(
        train.group_ids,
        train.pre_ranks,
        train.source_costs,
        candidates_per_group=args.candidates_per_group,
        hard_fraction=args.hard_fraction,
        seed=args.seed,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    model = DCNv2Ranker(
        len(selected_features),
        BUCKETS,
        EMBEDDING_DIMS,
        cross_layers=3,
        deep_dims=(256, 128),
        dropout=0.05,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    baseline_scores = base_scores(holdout.pre_ranks)
    baseline = temporal_metrics(holdout, baseline_scores, holdout_requests)
    history = []
    best_key: tuple[float, float] | None = None
    best_state = None
    best_epoch = 0
    order_rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = order_rng.permutation(len(sampled.indices))
        losses = []
        for batch_start in range(0, len(permutation), args.groups_per_batch):
            group_rows = permutation[batch_start : batch_start + args.groups_per_batch]
            flat = sampled.indices[group_rows].reshape(-1)
            continuous = torch.from_numpy(train.features[flat]).to(
                device, non_blocking=True
            )
            categorical = torch.from_numpy(train.categorical[flat]).to(
                device, non_blocking=True
            )
            base = torch.from_numpy(base_scores(train.pre_ranks[flat])).to(
                device, non_blocking=True
            )
            target = torch.from_numpy(sampled.targets[group_rows]).to(
                device, non_blocking=True
            )
            weight = torch.from_numpy(sampled.weights[group_rows]).to(
                device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                scores = model(continuous, categorical, base).reshape(
                    len(group_rows), args.candidates_per_group
                )
                per_group = -(target * F.log_softmax(scores, dim=1)).sum(dim=1)
                loss = (per_group * weight).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        holdout_scores = score_model(
            model,
            holdout,
            device=device,
            batch_rows=args.score_batch_rows,
        )
        candidate = temporal_metrics(holdout, holdout_scores, holdout_requests)
        deltas = metric_deltas(candidate, baseline)
        key = (
            float(candidate["early"]["50"]["sourcecost_recall"]),
            float(candidate["early"]["50"]["recall"]),
        )
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "metrics": candidate,
                "deltas": deltas,
            }
        )
        print(json.dumps(history[-1]), flush=True)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise AssertionError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    final_scores = score_model(
        model,
        holdout,
        device=device,
        batch_rows=args.score_batch_rows,
    )
    final_metrics = temporal_metrics(holdout, final_scores, holdout_requests)
    final_deltas = metric_deltas(final_metrics, baseline)
    model_path = args.output_dir / "models" / "dcn_ranker.pt"
    checkpoint = {
        "version": 1,
        "architecture": "DCNv2_residual_listwise_v1",
        "state_dict": {key: value.detach().cpu() for key, value in best_state.items()},
        "feature_names": selected_features,
        "feature_mean": mean,
        "feature_standard_deviation": standard_deviation,
        "categorical_buckets": BUCKETS,
        "embedding_dims": EMBEDDING_DIMS,
        "best_epoch": best_epoch,
        "source_run": str(args.source_run),
    }
    with atomic_output_path(model_path) as temporary:
        torch.save(checkpoint, temporary)
    prediction_path = args.output_dir / "predictions" / "holdout_top500.parquet"
    requests_written = export_ranking(
        prediction_path, holdout, final_scores, top_k=args.top_k
    )
    report = {
        "status": "completed",
        "architecture": "DCNv2_residual_listwise_v1",
        "source_run": str(args.source_run),
        "leakage_contract": {
            "fit_split": "train",
            "selection_split": "holdout_early_half",
            "confirmation_split": "holdout_late_half",
            "train_holdout_request_overlap": 0,
            "train_max_show_time": train_max_time,
            "holdout_min_show_time": holdout_min_time,
            "candidate_pool": "cached natural pool; no positive injection",
        },
        "rows": {"train": len(train.pre_ranks), "holdout": len(holdout.pre_ranks)},
        "positive_train_groups": len(sampled.indices),
        "features": {
            "configured": len(feature_names),
            "selected_nonconstant": len(selected_features),
        },
        "parameters": vars(args) | {"source_run": str(args.source_run), "output_dir": str(args.output_dir)},
        "best_epoch": best_epoch,
        "baseline": baseline,
        "candidate": final_metrics,
        "deltas": final_deltas,
        "history": history,
        "artifacts": {
            "model": fingerprint_file(model_path),
            "holdout_ranking": fingerprint_file(prediction_path),
            "requests_written": requests_written,
        },
        "wall_seconds": time.monotonic() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "gpu": torch.cuda.get_device_name(0),
    }
    report_path = args.output_dir / "metrics" / "dcn_temporal.json"
    atomic_write_json(report_path, report)
    inputs = [
        fingerprint_file(args.source_run / "manifest.json"),
        fingerprint_file(args.source_run / "models" / "catboost.json"),
        *(fingerprint_file(path) for split in ("train", "holdout") for path in feature_paths(args.source_run, split)),
    ]
    write_output_manifest(
        model_path,
        stage="train_dcn_ranker",
        artifact_version="dcnv2_residual_listwise_v1",
        config_sha256=fingerprint_file(ROOT / "scripts" / "train_dcn_ranker.py")["sha256"],
        inputs=inputs,
        schema={"feature_names": selected_features, "categorical_buckets": BUCKETS},
        scope="offline",
    )
    write_output_manifest(
        prediction_path,
        stage="export_dcn_ranker",
        artifact_version=f"dcnv2_top{args.top_k}_v1",
        config_sha256=fingerprint_file(ROOT / "scripts" / "train_dcn_ranker.py")["sha256"],
        inputs=[fingerprint_file(model_path)],
        rows=requests_written,
        schema="HitLogID:uint64, BannerID:list<uint64>",
        scope="offline",
    )
    atomic_write_json(args.output_dir / "manifest.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
