from __future__ import annotations

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
    load_bpe_tokenizer,
)


SOLUTION_NAME = "two_tower_v2_dcn4_mlp3"


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
    return {
        "network": network,
        "config": config,
        "device": device,
        "candidate_vectors": candidate_vectors,
        "candidate_metadata": metadata,
        "tokenizer": tokenizer,
        "metadata": {
            "solution": SOLUTION_NAME,
            "candidates": candidate_vectors.shape[0],
            "training": checkpoint.get("training", {}),
        },
    }


def _query_row(example: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    tokens = tokenize(example.get("query"))[:32]
    context = example.get("context") or {}
    region_id = int(context.get("region_id") or 0)
    device = str(context.get("device") or "")
    age = max(0, int(context.get("age") or 0))
    gender = max(0, int(context.get("gender") or 0))
    return {
        "query_word_ids": [feature_bucket(token) for token in tokens],
        "region_ids": [feature_bucket(str(region_id))],
        "device_ids": [feature_bucket(device)],
        "age_bucket_ids": [age],
        "gender_ids": [gender],
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
        count = min(top_k, scores.shape[1])
        values, indices = torch.topk(scores, k=count, dim=1)
    metadata = model["candidate_metadata"]
    output: list[list[dict[str, Any]]] = []
    for tokens, score_row, index_row in zip(
        token_rows,
        values.float().cpu().tolist(),
        indices.cpu().tolist(),
    ):
        ranked = []
        for score, index in zip(score_row, index_row):
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
                    "contributions": {"dot_product": float(score)},
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
