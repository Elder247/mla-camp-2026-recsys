from __future__ import annotations

import hashlib
import math
import queue
import random
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

import torch


QUERY_FIELDS = ("query_word_ids", "region_ids")
BANNER_FIELDS = (
    "banner_id_ids",
    "ad_group_id_ids",
    "title_word_ids",
    "text_word_ids",
)
ALL_FIELDS = QUERY_FIELDS + BANNER_FIELDS
TEXT_SOURCE_FIELDS = ("query_text", "title_text", "text_text", "banner_url")
NUMERIC_SOURCE_FIELDS = ("source_cost", "product_price")
INTEGER_SOURCE_FIELDS = ("banner_id", "crypta_id_v2")
BPE_FIELDS = ("query_bpe_ids", "title_bpe_ids", "text_bpe_ids")
DIRECT_SOURCE_FIELDS = (
    "device_ids",
    "age_bucket_ids",
    "gender_ids",
    "income_ids",
    "client_id_ids",
    "order_id_ids",
    "caesar_model_id_ids",
    "caesar_sku_id_ids",
)
DERIVED_FIELDS = (
    "query_region_ids",
    "crypta_id_hash1_ids",
    "crypta_id_hash2_ids",
    "source_cost_bucket_ids",
    "product_price_bucket_ids",
    "url_domain_ids",
    "banner_id_hash2_ids",
)


def source_fields(cardinalities: Mapping[str, int]) -> tuple[str, ...]:
    """Return only YT columns needed to construct the configured model inputs."""

    fields = list(ALL_FIELDS)
    if any(name in cardinalities for name in BPE_FIELDS):
        fields.extend(TEXT_SOURCE_FIELDS[:3])
    fields.extend(name for name in DIRECT_SOURCE_FIELDS if name in cardinalities)
    if "source_cost_bucket_ids" in cardinalities:
        fields.append("source_cost")
    if "product_price_bucket_ids" in cardinalities:
        fields.append("product_price")
    if "url_domain_ids" in cardinalities:
        fields.append("banner_url")
    if "banner_id_hash2_ids" in cardinalities:
        fields.append("banner_id")
    if any(
        name in cardinalities
        for name in ("crypta_id_hash1_ids", "crypta_id_hash2_ids")
    ):
        fields.append("crypta_id_v2")
    return tuple(fields)


def _source_value(raw: Mapping[str, Any], name: str) -> Any:
    if name in TEXT_SOURCE_FIELDS:
        return str(raw.get(name) or "")
    if name in NUMERIC_SOURCE_FIELDS:
        return float(raw.get(name) or 0.0)
    if name in INTEGER_SOURCE_FIELDS:
        return int(raw.get(name) or 0)
    return [int(value) for value in raw.get(name) or () if value is not None]


def feature_bucket(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:2], "little", signed=False)


def wide_feature_bucket(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def deterministic_sample(value: object, *, fraction: float, seed: int) -> bool:
    """Return a stable request-level Bernoulli sample decision.

    Sampling by ``uniq_id`` keeps all clicked banners of one request together
    and spreads the retained training pairs across the complete chronological
    week.  The full stream is still used for OOF labels and history features.
    """

    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("sample fraction must be in (0, 1]")
    if float(fraction) == 1.0:
        return True
    payload = f"{int(seed)}\x1f{value}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    value_hash = int.from_bytes(digest, "little", signed=False)
    return value_hash < int(float(fraction) * (1 << 64))


def yt_read_options(*, ordered: bool) -> dict[str, bool]:
    """Return the YT reader contract for reproducible chronological streams."""

    return {
        "unordered": not ordered,
        "enable_read_parallel": not ordered,
    }


class YtTableSource:
    def __init__(
        self,
        table: str,
        proxy: str,
        *,
        ordered: bool = False,
        fields: Sequence[str] | None = None,
        allow_missing_fields: bool = False,
    ) -> None:
        from common.yt_data import make_client

        self.table = table
        self.proxy = proxy
        self.ordered = bool(ordered)
        self.client = make_client()
        if not self.client.exists(table):
            raise FileNotFoundError(f"YT table does not exist: {table}")
        self.row_count = int(self.client.get(f"{table}/@row_count"))
        schema = self.client.get(f"{table}/@schema")
        columns = {str(item["name"]) for item in schema}
        self.columns = columns
        self.fields = tuple(fields or ALL_FIELDS)
        missing = set(self.fields) - columns - set(NUMERIC_SOURCE_FIELDS)
        if allow_missing_fields:
            missing.clear()
        if missing:
            raise ValueError(f"YT table {table} misses fields: {sorted(missing)}")
        self.read_fields = tuple(name for name in self.fields if name in columns)
        order = "chronological" if self.ordered else "parallel"
        self.description = f"YT {proxy}:{table} ({order})"

    def rows(self) -> Iterator[dict[str, Any]]:
        import yt.wrapper as yt

        path = yt.TablePath(self.table, columns=list(self.read_fields))
        for raw in self.client.read_table(
            path,
            **yt_read_options(ordered=self.ordered),
        ):
            yield {name: _source_value(raw, name) for name in self.fields}


class YtWeekTableSource(YtTableSource):
    def __init__(
        self,
        table: str,
        proxy: str,
        *,
        start: int,
        end: int,
        fields: Sequence[str] | None = None,
    ) -> None:
        super().__init__(table, proxy, fields=fields)
        sorted_by_path = f"{table}/@sorted_by"
        sorted_by = (
            [str(value) for value in self.client.get(sorted_by_path)]
            if self.client.exists(sorted_by_path)
            else []
        )
        if not sorted_by or sorted_by[0] != "week_start":
            raise ValueError(f"YT weekly table must be sorted by week_start: {table}")
        self.start = int(start)
        self.end = int(end)
        self.row_count = 0
        self.description = f"YT {proxy}:{table}[{self.start}:{self.end})"

    def rows(
        self,
        *,
        sample_fraction: float = 1.0,
        sample_seed: int = 0,
    ) -> Iterator[dict[str, Any]]:
        import yt.wrapper as yt

        fraction = float(sample_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("sample fraction must be in (0, 1]")
        columns = list(self.read_fields)
        if fraction < 1.0:
            if "uniq_id" not in self.columns:
                raise ValueError(
                    f"YT weekly table requires uniq_id for sampling: {self.table}"
                )
            columns.append("uniq_id")
        path = yt.TablePath(
            self.table,
            columns=columns,
            ranges=[
                {
                    "lower_limit": {"key": [self.start]},
                    "upper_limit": {"key": [self.end]},
                }
            ],
        )
        for raw in self.client.read_table(
            path,
            unordered=False,
            enable_read_parallel=True,
        ):
            if fraction < 1.0 and not deterministic_sample(
                raw.get("uniq_id"),
                fraction=fraction,
                seed=int(sample_seed),
            ):
                continue
            yield {name: _source_value(raw, name) for name in self.fields}


def enrich_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    cardinalities: Mapping[str, int],
    tokenizer: Any | None,
    bpe_limits: Mapping[str, int] | None = None,
    source_cost_log1p_scale: float = 1.0,
    product_price_log1p_scale: float = 1.0,
) -> list[dict[str, Any]]:
    """Add config-gated BPE and query-region inputs without changing YT data."""

    enriched = [dict(row) for row in rows]
    limits = dict(bpe_limits or {})
    text_to_feature = {
        "query_text": "query_bpe_ids",
        "title_text": "title_bpe_ids",
        "text_text": "text_bpe_ids",
    }
    for text_name, feature_name in text_to_feature.items():
        if feature_name not in cardinalities:
            continue
        if tokenizer is None:
            raise ValueError(f"{feature_name} requires a configured BPE tokenizer")
        encoded = tokenizer.encode_batch(
            [str(row.get(text_name) or "") for row in enriched]
        )
        limit = int(limits.get(feature_name, 0))
        cardinality = int(cardinalities[feature_name])
        for row, item in zip(enriched, encoded):
            ids = item.ids[:limit] if limit > 0 else item.ids
            row[feature_name] = [int(value) % cardinality for value in ids]

    if "query_region_ids" in cardinalities:
        cardinality = int(cardinalities["query_region_ids"])
        for row in enriched:
            region = int(next(iter(row.get("region_ids") or (0,))))
            query = row.get("query_word_ids") or (0,)
            row["query_region_ids"] = [
                ((int(token) * 16_777_619) ^ region) % cardinality
                for token in query
            ]
    for feature_name, namespace in (
        ("crypta_id_hash1_ids", "crypta1"),
        ("crypta_id_hash2_ids", "crypta2"),
    ):
        if feature_name not in cardinalities:
            continue
        cardinality = int(cardinalities[feature_name])
        if cardinality <= 1:
            raise ValueError(f"{feature_name} cardinality must exceed one")
        for row in enriched:
            crypta_id = int(row.get("crypta_id_v2") or 0)
            bucket = (
                0
                if crypta_id <= 0
                else 1
                + wide_feature_bucket(f"{namespace}:{crypta_id}")
                % (cardinality - 1)
            )
            row[feature_name] = [bucket]
    if "source_cost_bucket_ids" in cardinalities:
        if source_cost_log1p_scale <= 0.0:
            raise ValueError("source_cost_log1p_scale must be positive")
        cardinality = int(cardinalities["source_cost_bucket_ids"])
        for row in enriched:
            source_cost = max(0.0, float(row.get("source_cost") or 0.0))
            bucket = min(
                cardinality - 1,
                int(math.log1p(source_cost) * source_cost_log1p_scale),
            )
            row["source_cost_bucket_ids"] = [bucket]
    if "product_price_bucket_ids" in cardinalities:
        if product_price_log1p_scale <= 0.0:
            raise ValueError("product_price_log1p_scale must be positive")
        cardinality = int(cardinalities["product_price_bucket_ids"])
        for row in enriched:
            price = max(0.0, float(row.get("product_price") or 0.0))
            bucket = min(
                cardinality - 1,
                int(math.log1p(price) * product_price_log1p_scale),
            )
            row["product_price_bucket_ids"] = [bucket]
    if "url_domain_ids" in cardinalities:
        cardinality = int(cardinalities["url_domain_ids"])
        for row in enriched:
            value = str(row.get("banner_url") or "").lower()
            host = value.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
            row["url_domain_ids"] = [feature_bucket(host) % cardinality]
    if "banner_id_hash2_ids" in cardinalities:
        cardinality = int(cardinalities["banner_id_hash2_ids"])
        for row in enriched:
            banner_id = int(row.get("banner_id") or 0)
            row["banner_id_hash2_ids"] = [
                wide_feature_bucket(f"banner2:{banner_id}") % cardinality
            ]
    return enriched


def shuffled_rows(
    rows: Iterable[dict[str, list[int]]],
    *,
    buffer_size: int,
    seed: int,
) -> Iterator[dict[str, list[int]]]:
    if buffer_size <= 1:
        yield from rows
        return
    generator = random.Random(seed)
    buffer: list[dict[str, list[int]]] = []
    for row in rows:
        if len(buffer) < buffer_size:
            buffer.append(row)
            continue
        index = generator.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = row
    generator.shuffle(buffer)
    yield from buffer


def batches(rows: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


class _PrefetchFailure:
    def __init__(self, error: BaseException) -> None:
        self.error = error


def prefetch_batches(
    values: Iterable[list[dict[str, Any]]], depth: int
) -> Iterator[list[dict[str, Any]]]:
    """Preserve batch order while overlapping remote reads with GPU work."""

    if depth <= 0:
        yield from values
        return
    pending: queue.Queue[object] = queue.Queue(maxsize=int(depth))
    stop = threading.Event()
    sentinel = object()

    def put(value: object) -> bool:
        while not stop.is_set():
            try:
                pending.put(value, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        try:
            for value in values:
                if not put(value):
                    return
        except BaseException as error:
            put(_PrefetchFailure(error))
        else:
            put(sentinel)

    worker = threading.Thread(target=produce, name="two-tower-prefetch", daemon=True)
    worker.start()
    try:
        while True:
            value = pending.get()
            if value is sentinel:
                break
            if isinstance(value, _PrefetchFailure):
                raise value.error
            yield value  # type: ignore[misc]
    finally:
        stop.set()
        worker.join(timeout=1.0)


def pack_bags(
    rows: Sequence[Mapping[str, Sequence[int]]],
    *,
    cardinalities: Mapping[str, int],
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    packed: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, cardinality in cardinalities.items():
        values: list[int] = []
        offsets: list[int] = []
        for row in rows:
            offsets.append(len(values))
            raw = row.get(name) or (0,)
            values.extend(int(value) % int(cardinality) for value in raw)
        packed[name] = (
            torch.tensor(values, dtype=torch.long, device=device),
            torch.tensor(offsets, dtype=torch.long, device=device),
        )
    return packed
