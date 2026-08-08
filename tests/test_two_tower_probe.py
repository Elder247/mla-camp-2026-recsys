from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.continue_two_tower_probe import main  # noqa: F401,E402


def test_probe_supervisor_is_importable() -> None:
    assert callable(main)
