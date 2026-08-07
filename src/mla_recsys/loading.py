from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_module(path: str | Path, python_paths: list[str] | None = None) -> ModuleType:
    """Load a solution module while making its legacy project importable.

    Teacher baselines use absolute imports such as ``common.text`` and
    ``code_maxim.step2_ce``.  The configured project roots are therefore added
    to ``sys.path`` before executing the module.  Large artifacts stay outside
    this repository.
    """

    for value in reversed(python_paths or []):
        resolved = str(Path(value).expanduser().resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)

    code_path = Path(path).expanduser().resolve()
    if not code_path.is_file():
        raise FileNotFoundError(f"Solution module does not exist: {code_path}")
    digest = hashlib.sha1(str(code_path).encode("utf-8")).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(f"mla_generator_{digest}", code_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import solution module: {code_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for required in ("load_model", "rank"):
        if not hasattr(module, required):
            raise TypeError(f"{code_path} does not implement {required}()")
    return module

