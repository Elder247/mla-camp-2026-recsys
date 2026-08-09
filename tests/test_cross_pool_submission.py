from __future__ import annotations

from scripts.make_cross_pool_submission import submission_rows


def test_submission_rows_preserve_request_order_and_unique_top50() -> None:
    requests = [
        {"request_id": "b", "hit_log_id": 2},
        {"request_id": "a", "hit_log_id": 1},
    ]
    old = {
        request: [(banner, hit_log, float(banner)) for banner in range(1, 61)]
        for request, hit_log in (("a", 1), ("b", 2))
    }
    new = {
        request: [(banner, hit_log, float(banner)) for banner in range(30, 91)]
        for request, hit_log in (("a", 1), ("b", 2))
    }
    rows = submission_rows(
        requests,
        old,
        new,
        new_weight=0.4,
        rrf_constant=10.0,
        exponent=0.2,
        rerank_top_n=75,
    )
    assert [row["HitLogID"] for row in rows] == [2, 1]
    assert all(len(row["BannerID"]) == 50 for row in rows)
    assert all(len(set(row["BannerID"])) == 50 for row in rows)
