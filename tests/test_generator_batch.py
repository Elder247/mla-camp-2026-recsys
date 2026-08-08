from __future__ import annotations

from mla_recsys.pipeline import Generator


class BatchedModule:
    @staticmethod
    def rank_batch(*, model, examples, features, top_k):
        del model, features, top_k
        return [[{"banner_id": index}] for index, _ in enumerate(examples)]


class SingleModule:
    @staticmethod
    def rank(*, model, example, features, top_k):
        del model, features, top_k
        return [{"banner_id": int(example["value"])}]


def generator(module) -> Generator:
    return Generator(
        name="test",
        module=module,
        model={},
        top_k=1,
        quota=1,
        weight=1.0,
        features={},
        batch_size=8,
    )


def test_native_batch_contract() -> None:
    assert generator(BatchedModule).rank_batch([{}, {}]) == [
        [{"banner_id": 0}],
        [{"banner_id": 1}],
    ]


def test_batch_falls_back_to_single_rank() -> None:
    assert generator(SingleModule).rank_batch([{"value": 7}, {"value": 8}]) == [
        [{"banner_id": 7}],
        [{"banner_id": 8}],
    ]
