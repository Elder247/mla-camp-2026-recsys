from __future__ import annotations

import hashlib
import random
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


def feature_bucket(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:2], "little", signed=False)


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


class YtTableSource:
    def __init__(self, table: str, proxy: str) -> None:
        from common.yt_data import make_client

        self.table = table
        self.proxy = proxy
        self.client = make_client()
        if not self.client.exists(table):
            raise FileNotFoundError(f"YT table does not exist: {table}")
        self.row_count = int(self.client.get(f"{table}/@row_count"))
        schema = self.client.get(f"{table}/@schema")
        columns = {str(item["name"]) for item in schema}
        self.columns = columns
        missing = set(ALL_FIELDS) - columns
        if missing:
            raise ValueError(f"YT table {table} misses fields: {sorted(missing)}")
        self.description = f"YT {proxy}:{table}"

    def rows(self) -> Iterator[dict[str, list[int]]]:
        import yt.wrapper as yt

        path = yt.TablePath(self.table, columns=list(ALL_FIELDS))
        for raw in self.client.read_table(
            path,
            unordered=True,
            enable_read_parallel=True,
        ):
            yield {
                name: [int(value) for value in raw.get(name) or ()]
                for name in ALL_FIELDS
            }


class YtWeekTableSource(YtTableSource):
    def __init__(self, table: str, proxy: str, *, start: int, end: int) -> None:
        super().__init__(table, proxy)
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
    ) -> Iterator[dict[str, list[int]]]:
        import yt.wrapper as yt

        fraction = float(sample_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("sample fraction must be in (0, 1]")
        columns = list(ALL_FIELDS)
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
            yield {
                name: [int(value) for value in raw.get(name) or ()]
                for name in ALL_FIELDS
            }


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
