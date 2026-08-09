from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common.text import normalize, tokenize
from two_tower_v2.data import enrich_rows, feature_bucket, pack_bags
from two_tower_v2.training import (
    EMBEDDINGS_FILENAME,
    METADATA_FILENAME,
    MODEL_FILENAME,
    bpe_limits,
    build_model,
    file_sha256,
    load_bpe_tokenizer,
)


SOLUTION_NAME = "two_tower_v2_dcn4_mlp3"
INFERENCE_CONFIG_FILENAME = "inference_config.json"


def load_logq_restore_bias(
    artifact_dir: Path,
    *,
    candidate_metadata: dict[str, list[Any]],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
    """Load a validated, train-only item-prior bias aligned to the index."""

    config_path = artifact_dir / INFERENCE_CONFIG_FILENAME
    if not config_path.is_file():
        return None, None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("version", 0)) != 1:
        raise ValueError("unsupported TwoTower inference config version")
    if config.get("kind") != "global_banner_logq_restore":
        raise ValueError("unsupported TwoTower inference transform")
    alpha = float(config.get("alpha", 0.0))
    if not -1.0 <= alpha <= 1.0:
        raise ValueError("logQ restore alpha must be in [-1, 1]")
    unseen_count = float(config.get("unseen_count", 1.0))
    if unseen_count <= 0.0:
        raise ValueError("logQ restore unseen_count must be positive")
    prior_dir = Path(str(config["prior_dir"]))
    manifest_path = prior_dir / "manifest.json"
    expected_manifest_sha = str(config.get("prior_manifest_sha256", ""))
    if not manifest_path.is_file() or not expected_manifest_sha:
        raise FileNotFoundError("validated logQ prior manifest is unavailable")
    if file_sha256(manifest_path) != expected_manifest_sha:
        raise ValueError("logQ prior manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "global_banner_frequency":
        raise ValueError("unexpected logQ prior kind")
    prior_file = prior_dir / str(manifest["file"]["name"])
    if file_sha256(prior_file) != str(manifest["file"]["sha256"]):
        raise ValueError("logQ prior data SHA-256 mismatch")

    import pyarrow.parquet as pq

    prior = pq.read_table(prior_file, columns=["banner_id", "count"])
    prior_ids = np.asarray(
        prior["banner_id"].combine_chunks().to_numpy(), dtype=np.uint64
    )
    prior_counts = np.asarray(
        prior["count"].combine_chunks().to_numpy(), dtype=np.float64
    )
    if prior_ids.size == 0 or np.any(prior_ids[1:] <= prior_ids[:-1]):
        raise ValueError("logQ prior ids must be strictly increasing")
    candidate_ids = np.asarray(candidate_metadata["banner_id"], dtype=np.uint64)
    positions = np.searchsorted(prior_ids, candidate_ids)
    safe_positions = np.minimum(positions, prior_ids.size - 1)
    matched = (positions < prior_ids.size) & (
        prior_ids[safe_positions] == candidate_ids
    )
    counts = np.full(candidate_ids.shape, unseen_count, dtype=np.float64)
    counts[matched] = prior_counts[safe_positions[matched]]
    values = (alpha * np.log(counts)).astype(np.float32)
    bias = torch.from_numpy(values).to(device=device, dtype=dtype)
    return bias, {
        "kind": "global_banner_logq_restore",
        "alpha": alpha,
        "prior_dir": str(prior_dir),
        "candidate_coverage": float(matched.mean()),
        "candidate_misses": int((~matched).sum()),
    }


def input_schema() -> list[dict[str, Any]]:
    return [
        {"name": "query", "path": "query", "type": "textarea", "primary": True},
        {
            "name": "region_id",
            "path": "context.region_id",
            "type": "integer",
            "nullable": True,
        },
        {"name": "device", "path": "context.device", "type": "string", "nullable": True},
        {"name": "age", "path": "context.age", "type": "integer", "nullable": True},
        {"name": "gender", "path": "context.gender", "type": "integer", "nullable": True},
        {
            "name": "income",
            "path": "context.income",
            "type": "integer",
            "nullable": True,
        },
        {
            "name": "crypta_id_v2",
            "path": "context.crypta_id_v2",
            "type": "integer",
            "nullable": True,
        },
    ]


def feature_schema() -> list[dict[str, Any]]:
    return []


def load_model(
    artifact_dir: Path,
    *,
    candidate_metadata: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    paths = [
        artifact_dir / MODEL_FILENAME,
        artifact_dir / EMBEDDINGS_FILENAME,
        artifact_dir / METADATA_FILENAME,
        artifact_dir / "manifest.json",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"TwoTower v2 artifact is incomplete: {missing}")
    checkpoint = torch.load(paths[0], map_location="cpu", weights_only=True)
    if int(checkpoint.get("version", 0)) != 2:
        raise ValueError(f"Unsupported TwoTower v2 checkpoint: {checkpoint.get('version')}")
    config = checkpoint["config"]
    tokenizer = load_bpe_tokenizer(config, artifact_dir=artifact_dir)
    network = build_model(config)
    network.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device).eval()
    candidate_vectors = torch.from_numpy(
        np.array(np.load(paths[1], mmap_mode="r"), dtype=np.float16, copy=True)
    ).to(device)
    metadata = (
        candidate_metadata
        if candidate_metadata is not None
        else pq.read_table(paths[2]).to_pydict()
    )
    if len(metadata["banner_id"]) != candidate_vectors.shape[0]:
        raise ValueError("candidate metadata and embeddings have different sizes")
    candidate_logq_bias, inference = load_logq_restore_bias(
        artifact_dir,
        candidate_metadata=metadata,
        device=device,
        dtype=candidate_vectors.dtype,
    )
    return {
        "network": network,
        "config": config,
        "device": device,
        "candidate_vectors": candidate_vectors,
        "candidate_metadata": metadata,
        "candidate_logq_bias": candidate_logq_bias,
        "tokenizer": tokenizer,
        "metadata": {
            "solution": SOLUTION_NAME,
            "candidates": candidate_vectors.shape[0],
            "training": checkpoint.get("training", {}),
            "inference": inference,
        },
    }


def _query_row(example: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    tokens = tokenize(example.get("query"))[:32]
    context = example.get("context") or {}
    region_id = int(context.get("region_id") or 0)
    device = str(context.get("device") or "")
    age = max(0, int(context.get("age") or 0))
    gender = max(0, int(context.get("gender") or 0))
    income_value = context.get("income")
    income = max(0, int(income_value if income_value is not None else -1) + 1)
    crypta_id_v2 = max(0, int(context.get("crypta_id_v2") or 0))
    return {
        "query_word_ids": [feature_bucket(token) for token in tokens],
        "region_ids": [feature_bucket(str(region_id))],
        "device_ids": [feature_bucket(device)],
        "age_bucket_ids": [age],
        "gender_ids": [gender],
        "income_ids": [income],
        "crypta_id_v2": crypta_id_v2,
        "query_text": normalize(example.get("query")),
        "title_text": "",
        "text_text": "",
    }, tokens


def rank_batch(
    *,
    model: dict[str, Any],
    examples: list[dict[str, Any]],
    features: dict[str, Any],
    top_k: int,
) -> list[list[dict[str, Any]]]:
    del features
    if not examples:
        return []
    rows_and_tokens = [_query_row(example) for example in examples]
    rows = [item[0] for item in rows_and_tokens]
    token_rows = [item[1] for item in rows_and_tokens]
    query_cardinalities = {
        str(k): int(v)
        for k, v in model["config"]["model"]["query_cardinalities"].items()
    }
    rows = enrich_rows(
        rows,
        cardinalities=query_cardinalities,
        tokenizer=model.get("tokenizer"),
        bpe_limits=bpe_limits(model["config"]),
    )
    bags = pack_bags(
        rows,
        cardinalities=query_cardinalities,
        device=model["device"],
    )
    with torch.inference_mode(), torch.autocast(
        device_type=model["device"].type,
        dtype=torch.bfloat16,
        enabled=model["device"].type == "cuda",
    ):
        query_vectors = model["network"].encode_query(bags)
        scores = query_vectors.to(model["candidate_vectors"].dtype) @ model[
            "candidate_vectors"
        ].T
        if model.get("candidate_logq_bias") is not None:
            scores = scores + model["candidate_logq_bias"].unsqueeze(0)
        count = min(top_k, scores.shape[1])
        values, indices = torch.topk(scores, k=count, dim=1)
        selected_bias = (
            model["candidate_logq_bias"][indices]
            if model.get("candidate_logq_bias") is not None
            else torch.zeros_like(values)
        )
    metadata = model["candidate_metadata"]
    output: list[list[dict[str, Any]]] = []
    for tokens, score_row, index_row, bias_row in zip(
        token_rows,
        values.float().cpu().tolist(),
        indices.cpu().tolist(),
        selected_bias.float().cpu().tolist(),
    ):
        ranked = []
        for score, index, logq_bias in zip(score_row, index_row, bias_row):
            title = metadata["title"][index] or ""
            text = metadata["text"][index] or ""
            candidate_tokens = set(tokenize(title)) | set(tokenize(text))
            ranked.append(
                {
                    "banner_id": int(metadata["banner_id"][index]),
                    "title": title,
                    "text": text,
                    "url": metadata["url"][index] or "",
                    "source_cost": float(metadata["source_cost"][index] or 0.0),
                    "score": float(score),
                    "contributions": {
                        "dot_product": float(score - logq_bias),
                        "logq_restore": float(logq_bias),
                    },
                    "matched_tokens": [token for token in tokens if token in candidate_tokens],
                }
            )
        output.append(ranked)
    return output


def rank(
    *,
    model: dict[str, Any],
    example: dict[str, Any],
    features: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    return rank_batch(
        model=model,
        examples=[example],
        features=features,
        top_k=top_k,
    )[0]
