#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a run without loading ML artifacts")
    parser.add_argument("run", type=Path, help="runs/<run_id> directory")
    args = parser.parse_args()
    result = read_json(args.run / "result.json")
    manifest = read_json(args.run / "manifest.json")
    if result is None or manifest is None:
        print(f"invalid run directory: {args.run}", file=sys.stderr)
        return 1
    summary = {
        "run_id": result.get("run_id"),
        "status": result.get("status"),
        "experiment": manifest.get("experiment"),
        "scope": manifest.get("scope"),
        "git": manifest.get("git"),
        "compute": manifest.get("compute"),
        "stages": result.get("stages"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
