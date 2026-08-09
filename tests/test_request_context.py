from __future__ import annotations

from mla_recsys.data import REQUEST_SCHEMA, request_example


def test_income_survives_request_contract_and_generator_mapping() -> None:
    assert "income" in REQUEST_SCHEMA.names
    example = request_example(
        {
            "request_id": "request-1",
            "show_time": 123,
            "query": "example",
            "income": 4,
        }
    )
    assert example["context"]["income"] == 4
