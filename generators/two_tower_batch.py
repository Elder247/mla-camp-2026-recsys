from __future__ import annotations

from typing import Any

import torch

from common.text import tokenize
from code_maxim.step2_ce import inference as base
from code_maxim.step2_ce.model import pack_features, query_features


input_schema = base.input_schema
feature_schema = base.feature_schema
load_model = base.load_model
rank = base.rank


def rank_batch(
    *,
    model: dict[str, Any],
    examples: list[dict[str, Any]],
    features: dict[str, Any],
    top_k: int,
) -> list[list[dict[str, Any]]]:
    """Mathematically equivalent batched query encoding and exact top-k scan."""

    del features
    if not examples:
        return []
    device = model["device"]
    config = model["config"]
    token_rows = [tokenize(example.get("query"))[:32] for example in examples]
    rows = [
        {
            "query_tokens": tokens,
            "region_id": (example.get("context") or {}).get("region_id") or 0,
        }
        for example, tokens in zip(examples, token_rows)
    ]
    ids, offsets = pack_features(
        [query_features(row) for row in rows],
        hash_buckets=int(config["hash_buckets"]),
        device=device,
    )
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        query_vectors = model["network"].encode_query(ids, offsets)
        scores = query_vectors.to(model["candidate_vectors"].dtype) @ model[
            "candidate_vectors"
        ].T
        count = min(top_k, scores.shape[1])
        values, indices = torch.topk(scores, k=count, dim=1)
    values_rows = values.float().cpu().tolist()
    index_rows = indices.cpu().tolist()
    metadata = model["candidate_metadata"]
    output = []
    for query_tokens, score_row, index_row in zip(token_rows, values_rows, index_rows):
        query_token_set = set(query_tokens)
        ranked = []
        for score, index in zip(score_row, index_row):
            title = metadata["title"][index] or ""
            text = metadata["text"][index] or ""
            matched = [
                token
                for token in query_tokens
                if token in set(tokenize(title)) or token in set(tokenize(text))
            ]
            ranked.append(
                {
                    "banner_id": int(metadata["banner_id"][index]),
                    "title": title,
                    "text": text,
                    "url": metadata["url"][index] or "",
                    "source_cost": float(metadata["source_cost"][index] or 0.0),
                    "score": float(score),
                    "contributions": {"dot_product": float(score)},
                    "matched_tokens": [
                        token for token in matched if token in query_token_set
                    ],
                }
            )
        output.append(ranked)
    return output
