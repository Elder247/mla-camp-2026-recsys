#!/usr/bin/env python3
from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
for path in (PROJECT_ROOT, CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from code_maxim.step2_ce.export_candidates import arguments, run  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(arguments("step3_fps"))


if __name__ == "__main__":
    main()
