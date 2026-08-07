from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
for path in (PROJECT_ROOT, CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from code_maxim.step2_ce import inference as base

SOLUTION_NAME = "step3_fps"
input_schema = base.input_schema
feature_schema = base.feature_schema
rank = base.rank


def load_model(artifact_dir: Path) -> dict[str, Any]:
    model = base.load_model(artifact_dir)
    model["metadata"]["solution"] = SOLUTION_NAME
    return model
