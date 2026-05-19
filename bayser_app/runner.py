from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO


def build_bayser_command(
    *,
    features_path: Path,
    results_dir: Path,
    plot_dir: Path,
    c14_path: Path | None = None,
    intcal20_path: Path | None = None,
    feature_id_col: str | None = None,
    c14_id_col: str | None = None,
    bp_col: str | None = None,
    error_col: str | None = None,
    draws: int = 800,
    tune: int = 1200,
    chains: int = 4,
    target_accept: float = 0.96,
    random_seed: int = 123,
    max_treedepth: int = 12,
    min_type_count: int = 2,
    min_grave_count: int = 2,
    include_richness: bool = True,
    filter_data: bool = True,
    repulsion_strength: float = 0.35,
    outliers: list[str] | None = None,
    outlier_all: float | None = None,
) -> list[str]:
    """Build the Bayser command for the current Python environment."""

    cmd = [
        sys.executable,
        "-m",
        "bayser.cli",
        "--features",
        str(features_path),
        "--results-dir",
        str(results_dir),
        "--plot-dir",
        str(plot_dir),
        "--draws",
        str(draws),
        "--tune",
        str(tune),
        "--chains",
        str(chains),
        "--target-accept",
        str(target_accept),
        "--random-seed",
        str(random_seed),
        "--max-treedepth",
        str(max_treedepth),
        "--min-type-count",
        str(min_type_count),
        "--min-grave-count",
        str(min_grave_count),
        "--repulsion-strength",
        str(repulsion_strength),
        "--quiet",
    ]

    if filter_data:
        cmd.append("--filter")
    else:
        cmd.append("--no-filter")

    if include_richness:
        cmd.append("--include-richness")
    else:
        cmd.append("--no-include-richness")

    if feature_id_col:
        cmd.extend(["--feature-id-col", feature_id_col])

    if c14_path is not None:
        cmd.extend(["--c14", str(c14_path)])

    if intcal20_path is not None:
        cmd.extend(["--intcal20", str(intcal20_path)])

    if c14_id_col:
        cmd.extend(["--c14-id-col", c14_id_col])

    if bp_col:
        cmd.extend(["--bp-col", bp_col])

    if error_col:
        cmd.extend(["--error-col", error_col])

    for spec in outliers or []:
        cmd.extend(["--outlier", spec])

    if outlier_all is not None:
        cmd.extend(["--outlier-all", str(outlier_all)])

    return cmd


def _reader_thread(
    stream: TextIO | None,
    stream_name: str,
    out_queue: queue.Queue[dict],
) -> None:
    if stream is None:
        return

    try:
        for line in iter(stream.readline, ""):
            out_queue.put(
                {
                    "type": "line",
                    "stream": stream_name,
                    "text": line,
                }
            )
    finally:
        stream.close()


def stream_bayser_run(
    cmd: list[str],
    *,
    timeout_seconds: int = 3600,
    heartbeat_seconds: float = 0.5,
    env: dict[str, str] | None = None,
) -> Iterator[dict]:
    """Run Bayser as a subprocess and yield stdout/stderr/progress events."""

    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=run_env,
    )

    q: queue.Queue[dict] = queue.Queue()

    stdout_thread = threading.Thread(
        target=_reader_thread,
        args=(process.stdout, "stdout", q),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_reader_thread,
        args=(process.stderr, "stderr", q),
        daemon=True,
    )

    stdout_thread.start()
    stderr_thread.start()

    start = time.time()
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def drain_queue() -> None:
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                break

            if item.get("type") == "line":
                if item.get("stream") == "stdout":
                    stdout_lines.append(item.get("text", ""))
                else:
                    stderr_lines.append(item.get("text", ""))

    while True:
        elapsed = time.time() - start

        if elapsed > timeout_seconds and process.poll() is None:
            process.kill()

            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            drain_queue()

            yield {
                "type": "timeout",
                "elapsed": elapsed,
                "stdout": "".join(stdout_lines),
                "stderr": "".join(stderr_lines),
            }
            return

        try:
            item = q.get(timeout=heartbeat_seconds)
            item["elapsed"] = elapsed

            if item["type"] == "line":
                if item["stream"] == "stdout":
                    stdout_lines.append(item["text"])
                else:
                    stderr_lines.append(item["text"])

            yield item

        except queue.Empty:
            yield {
                "type": "heartbeat",
                "elapsed": elapsed,
            }

        if process.poll() is not None:
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)

            while True:
                try:
                    item = q.get_nowait()
                    item["elapsed"] = time.time() - start

                    if item["type"] == "line":
                        if item["stream"] == "stdout":
                            stdout_lines.append(item["text"])
                        else:
                            stderr_lines.append(item["text"])

                    yield item
                except queue.Empty:
                    break

            yield {
                "type": "done",
                "elapsed": time.time() - start,
                "returncode": process.returncode,
                "stdout": "".join(stdout_lines),
                "stderr": "".join(stderr_lines),
            }
            return