#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import resource
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the shared TwoTower BPE tokenizer")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    cfg = OmegaConf.load(args.config.resolve())
    sys.path.insert(0, str(cfg.paths.step2_root))
    from common.yt_data import make_client
    from mla_recsys.tracking import UnderdeepTracker, numeric_metrics
    from two_tower_v2.training import atomic_json
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    artifact_dir = Path(str(cfg.paths.artifact_dir))
    output = artifact_dir / "tokenizer.json"
    manifest_path = artifact_dir / "manifest.json"
    if manifest_path.is_file() and output.is_file():
        print(manifest_path.read_text(encoding="utf-8"))
        return 0
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite incomplete tokenizer: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(artifact_dir / "train.log", encoding="utf-8"),
        ],
    )
    resolved = OmegaConf.to_yaml(cfg, resolve=True)
    (artifact_dir / "config.resolved.yaml").write_text(resolved, encoding="utf-8")
    tracker = UnderdeepTracker(
        artifact_dir=artifact_dir,
        tracking_cfg=cfg.tracking.underdeep,
        run_name=str(cfg.tracking.underdeep.run_name),
        description="Shared lowercase BPE tokenizer for TwoTower query/title/text",
        parameters={
            "corpus_table": str(cfg.paths.corpus_table),
            "vocab_size": int(cfg.tokenizer.vocab_size),
            "min_frequency": int(cfg.tokenizer.min_frequency),
        },
        tags=["mla-camp", "two-tower-v3", "bpe"],
    )
    try:
        client = make_client()
        table = str(cfg.paths.corpus_table)
        if not client.exists(table):
            raise FileNotFoundError(f"BPE corpus table does not exist: {table}")
        rows = int(client.get(f"{table}/@row_count"))
        batch_texts = int(cfg.tokenizer.batch_texts)

        def text_batches():
            batch: list[str] = []
            for row in client.read_table(
                table, unordered=True, enable_read_parallel=True
            ):
                for name in ("query_text", "title_text", "text_text"):
                    value = row.get(name)
                    if value:
                        batch.append(str(value))
                if len(batch) >= batch_texts:
                    yield batch
                    batch = []
            if batch:
                yield batch

        started = time.perf_counter()
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.normalizer = normalizers.Sequence(
            [normalizers.NFKC(), normalizers.Lowercase()]
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.decoder = decoders.BPEDecoder()
        trainer = trainers.BpeTrainer(
            vocab_size=int(cfg.tokenizer.vocab_size),
            min_frequency=int(cfg.tokenizer.min_frequency),
            special_tokens=["[PAD]", "[UNK]"],
            show_progress=True,
        )
        tokenizer.train_from_iterator(
            text_batches(), trainer=trainer, length=rows * 3
        )
        temporary = output.with_suffix(".json.tmp")
        tokenizer.save(str(temporary))
        os.replace(temporary, output)
        wall_seconds = time.perf_counter() - started
        report = {
            "version": 1,
            "status": "completed",
            "corpus_table": table,
            "corpus_rows": rows,
            "vocab_size": tokenizer.get_vocab_size(),
            "wall_seconds": wall_seconds,
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            * 1024,
            "files": {
                "tokenizer.json": {
                    "bytes": output.stat().st_size,
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                }
            },
        }
        atomic_json(artifact_dir / "metrics.json", report)
        atomic_json(manifest_path, report)
        tracker.log_summary(numeric_metrics(report, prefix="tokenizer"))
        tracker.close()
        logging.info("BPE tokenizer completed: %s", output)
    except Exception as error:
        tracker.close(error=type(error).__name__)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
