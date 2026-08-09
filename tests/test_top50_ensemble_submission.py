from __future__ import annotations

from scripts.make_top50_ensemble_submission import ensemble_rows


def test_ensemble_rows_materialize_unique_top50_in_request_order() -> None:
    requests = [
        {"hit_log_id": 2},
        {"hit_log_id": 1},
    ]
    sources = [
        {hit: list(range(1, 71)) for hit in (1, 2)},
        {hit: list(range(31, 101)) for hit in (1, 2)},
    ]
    rows = ensemble_rows(
        requests,
        sources,
        (0.6, 0.4),
        rrf_constant=10.0,
        exponent=0.2,
        rerank_top_n=75,
        source_costs={banner: float(banner) for banner in range(1, 101)},
    )

    assert [row["HitLogID"] for row in rows] == [2, 1]
    assert all(len(row["BannerID"]) == 50 for row in rows)
    assert all(len(set(row["BannerID"])) == 50 for row in rows)
