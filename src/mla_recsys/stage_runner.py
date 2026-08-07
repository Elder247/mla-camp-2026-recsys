from __future__ import annotations

import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

from .artifacts import RunStore, mask_secrets, utc_now


def _rss_bytes(value: int) -> int:
    return value if sys.platform == "darwin" else value * 1024


class StageRunner:
    """Run one heavy stage as a child process and persist its atomic status."""

    def __init__(self, store: RunStore, *, echo: bool = True) -> None:
        self.store = store
        self.echo = echo

    def run(
        self,
        stage: str,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        if not command:
            raise ValueError("Stage command cannot be empty")
        log_path = self.store.path / "logs" / f"{stage}.log"
        started_at = utc_now()
        started = time.monotonic()
        before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        rendered_command = [mask_secrets(item) for item in command]
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                env=dict(env) if env is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            with process.stdout:
                for raw_line in process.stdout:
                    line = mask_secrets(raw_line)
                    log.write(line)
                    log.flush()
                    if self.echo:
                        print(line, end="", flush=True)
            return_code = process.wait()
        after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        value: dict[str, object] = {
            "stage": stage,
            "status": "completed" if return_code == 0 else "failed",
            "started_at": started_at,
            "finished_at": utc_now(),
            "wall_seconds": round(time.monotonic() - started, 6),
            "peak_rss_bytes": _rss_bytes(max(before, after)),
            "return_code": return_code,
            "command": rendered_command,
            "log": str(log_path),
        }
        self.store.record_stage(stage, value)
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
        return value
