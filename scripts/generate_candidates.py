#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import atomic_write_json, make_cache_key  # noqa: E402
from mla_recsys.candidate_cache import (  # noqa: E402
    SOURCE_SCHEMA,
    cache_is_valid,
    finalize_source_manifests,
    generate_source_candidates,
    load_source,
    source_input_fingerprints,
)
from mla_recsys.command import load_stage_context, require_choice  # noqa: E402
from mla_recsys.config import config_fingerprint  # noqa: E402
from mla_recsys.data import read_request_parquet  # noqa: E402
from mla_recsys.temporal_candidates import (  # noqa: E402
    generate_temporal_source_candidates,
    is_temporal_source,
    temporal_source_inputs,
)


def main() -> int:
    context = load_stage_context(
        "Generate and cache one candidate source",
        extra_keys=("cg", "split", "force"),
    )
    cfg = context.cfg
    cg = require_choice(context, "cg", cfg.candidates.generators.keys())
    split = require_choice(context, "split", ("train", "holdout", "full_train", "test"))
    if not bool(cfg.candidates.generators[cg].get("enabled", False)):
        raise ValueError(f"Candidate generator is disabled: {cg}")
    request_path = context.store.path / "data" / f"{split}_requests.parquet"
    temporal = is_temporal_source(cfg, cg)
    spec = None if temporal else load_source(cfg, cg)
    inputs = (
        temporal_source_inputs(
            cfg=cfg,
            run_path=context.store.path,
            split=split,
            source=cg,
        )
        if temporal
        else source_input_fingerprints(spec, request_path)
    )
    config_sha = config_fingerprint(cfg)
    artifact_version = f"{cg}_candidates_v1"
    cache_key = make_cache_key(
        stage=f"generate_candidates_{cg}_{split}",
        artifact_version=artifact_version,
        config_sha256=config_sha,
        inputs=inputs,
    )
    partitions = int(cfg.data.partition_count)
    force = context.values.get("force", "false").lower() == "true"
    if not force and cache_is_valid(
        run_path=context.store.path,
        split=split,
        source=cg,
        partitions=partitions,
        cache_key=cache_key,
    ):
        print(json.dumps({"source": cg, "split": split, "status": "cache_hit"}))
        return 0
    requests = read_request_parquet(request_path)
    if temporal:
        report = generate_temporal_source_candidates(
            cfg=cfg,
            run_path=context.store.path,
            split=split,
            source=cg,
            requests=requests,
            partitions=partitions,
            buffer_rows=int(cfg.data.candidate_buffer_rows),
        )
    else:
        assert spec is not None
        report = generate_source_candidates(
            spec=spec,
            requests=requests,
            run_path=context.store.path,
            split=split,
            partitions=partitions,
            buffer_rows=int(cfg.data.candidate_buffer_rows),
        )
    finalize_source_manifests(
        run_path=context.store.path,
        split=split,
        source=cg,
        partitions=partitions,
        rows=report["partition_rows"],
        artifact_version=artifact_version,
        config_sha256=config_sha,
        inputs=inputs,
        scope=str(cfg.runtime.scope),
    )
    metric_path = context.store.path / "metrics" / f"generate_{split}_{cg}.json"
    atomic_write_json(metric_path, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
