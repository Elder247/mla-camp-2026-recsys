from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
PATHS = OmegaConf.load(ROOT / "configs" / "paths.yaml")
for path in (ROOT, ROOT / "src", Path(str(PATHS.paths.common_root)).parent):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

