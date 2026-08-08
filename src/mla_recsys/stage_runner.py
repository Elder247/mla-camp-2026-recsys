from __future__ import annotations

import resource
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Mapping, Sequence

from .artifacts import RunStore, mask_secrets, utc_now


def _rss_bytes(value: int) -> int:
    return value if sys.platform == "darwin" else value * 1024


def _linux_process_rss(pid: int) -> int:
    status = Path(f"/proc/{pid}/status")
    if not status.is_file():
        return 0
    text = status.read_text(encoding="utf-8", errors="replace")
    values = []
    for name in ("VmRSS", "VmHWM"):
        match = re.search(rf"^{name}:\s*(\d+)\s+kB$", text, flags=re.MULTILINE)
        if match:
            values.append(int(match.group(1)) * 1024)
    return max(values, default=0)


def _linux_process_tree_rss(pid: int) -> int:
    """Sample current RSS for a stage and all living descendants."""

    total = 0
    seen: set[int] = set()
    pending = [int(pid)]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        status = Path(f"/proc/{current}/status")
        if status.is_file():
            text = status.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"^VmRSS:\s*(\d+)\s+kB$", text, flags=re.MULTILINE)
            if match:
                total += int(match.group(1)) * 1024
        children = Path(f"/proc/{current}/task/{current}/children")
        if children.is_file():
            try:
                pending.extend(int(value) for value in children.read_text().split())
            except (FileNotFoundError, ProcessLookupError, ValueError):
                pass
    return total


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
        time_path = self.store.path / "logs" / f"{stage}.time"
        started_at = utc_now()
        started = time.monotonic()
        before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        rendered_command = [mask_secrets(item) for item in command]
        actual_command = list(command)
        use_gnu_time = sys.platform != "darwin" and Path("/usr/bin/time").is_file()
        if use_gnu_time:
            actual_command = ["/usr/bin/time", "-v", "-o", str(time_path), *command]
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                actual_command,
                cwd=cwd,
                env=dict(env) if env is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            sampled_peak = [0]
            stop_sampling = threading.Event()

            def sample_memory() -> None:
                while not stop_sampling.wait(0.1):
                    sampled_peak[0] = max(
                        sampled_peak[0], _linux_process_tree_rss(process.pid)
                    )

            sampler = None
            if sys.platform.startswith("linux") and not use_gnu_time:
                sampled_peak[0] = _linux_process_tree_rss(process.pid)
                sampler = threading.Thread(target=sample_memory, daemon=True)
                sampler.start()
            assert process.stdout is not None
            with process.stdout:
                for raw_line in process.stdout:
                    line = mask_secrets(raw_line)
                    log.write(line)
                    log.flush()
                    if self.echo:
                        try:
                            print(line, end="", flush=True)
                        except BrokenPipeError:
                            # The stage must survive a dropped SSH/client stdout;
                            # its durable log remains the source of truth.
                            self.echo = False
            return_code = process.wait()
            stop_sampling.set()
            if sampler is not None:
                sampler.join(timeout=1.0)
        after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        peak_rss_bytes = sampled_peak[0] or _rss_bytes(max(before, after))
        peak_rss_measurement = (
            "linux_proc_tree_stage" if sampled_peak[0] else "children_upper_bound"
        )
        if time_path.is_file():
            rendered_time = time_path.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", rendered_time)
            if match:
                peak_rss_bytes = int(match.group(1)) * 1024
                peak_rss_measurement = "gnu_time_stage"
        value: dict[str, object] = {
            "stage": stage,
            "status": "completed" if return_code == 0 else "failed",
            "started_at": started_at,
            "finished_at": utc_now(),
            "wall_seconds": round(time.monotonic() - started, 6),
            "peak_rss_bytes": peak_rss_bytes,
            "peak_rss_measurement": peak_rss_measurement,
            "peak_gpu_memory_bytes": None,
            "return_code": return_code,
            "command": rendered_command,
            "log": str(log_path),
        }
        self.store.record_stage(stage, value)
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
        return value
