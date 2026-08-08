from __future__ import annotations

from mla_recsys.text import normalize, tokenize


def test_local_text_contract_matches_frozen_normalization_semantics() -> None:
    assert normalize("  Ёлка, CAFÉ 42! ") == "елка caf 42"
    assert tokenize(b"Foo-bar_17") == ["foo", "bar", "17"]
