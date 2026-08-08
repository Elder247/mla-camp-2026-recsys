from __future__ import annotations

from pathlib import Path

from mla_recsys.counters import week_start

from generators import two_tower_v2_walk_forward as generator


def test_snapshot_selection_uses_preupdate_week_and_final_fallback() -> None:
    first = week_start(1_780_000_000)
    model = {
        "snapshots": {first: Path("/snapshot/first")},
        "final": Path("/snapshot/final"),
    }
    assert generator._artifact_for(
        model, {"show_time": first + 123}
    ) == Path("/snapshot/first")
    assert generator._artifact_for(
        model, {"show_time": first + 604800 + 123}
    ) == Path("/snapshot/final")
    assert generator._artifact_for(model, {"show_time": None}) == Path(
        "/snapshot/final"
    )


def test_rank_batch_preserves_order_across_snapshot_boundaries(monkeypatch) -> None:
    first = week_start(1_780_000_000)
    second = first + 604800
    model = {
        "snapshots": {first: Path("/first"), second: Path("/second")},
        "final": Path("/final"),
        "active_path": None,
        "active_model": None,
    }

    monkeypatch.setattr(
        generator,
        "_activate",
        lambda model, path: {"path": str(path)},
    )

    def fake_rank_batch(*, model, examples, features, top_k):
        del features, top_k
        return [
            [{"banner_id": int(example["request_id"]), "path": model["path"]}]
            for example in examples
        ]

    monkeypatch.setattr(generator.base, "rank_batch", fake_rank_batch)
    examples = [
        {"request_id": "1", "show_time": first + 1},
        {"request_id": "2", "show_time": first + 2},
        {"request_id": "3", "show_time": second + 1},
    ]
    result = generator.rank_batch(
        model=model, examples=examples, features={}, top_k=50
    )
    assert [row[0]["banner_id"] for row in result] == [1, 2, 3]
    assert [row[0]["path"] for row in result] == ["/first", "/first", "/second"]
