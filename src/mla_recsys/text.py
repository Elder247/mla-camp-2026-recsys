from __future__ import annotations

import re
import unicodedata
from typing import Any


TOKEN_RE = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def normalize(value: Any) -> str:
    """Project-owned copy of the stable common.text normalization contract."""

    text = unicodedata.normalize("NFKC", as_text(value)).lower().replace("ё", "е")
    return " ".join(TOKEN_RE.findall(text))


def tokenize(value: Any) -> list[str]:
    text = normalize(value)
    return text.split() if text else []
