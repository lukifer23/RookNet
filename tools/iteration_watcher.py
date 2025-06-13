#!/usr/bin/env python3
"""
Iteration Watcher
=================
Monitors the `training_progress.json` file produced by `dynamic_self_play.py` and
writes the latest iteration number to `logs/alpha_zero_training/iteration_counter.txt`.

On the first successful read it also creates a freeze-tag file
`models/alpha_zero_checkpoints/iteration_freeze_start.tag` so that other tooling
can verify how many iterations have elapsed since this watcher started.

The script is **strictly read-only** with respect to training artefacts – it does
not modify checkpoints or running processes. It imports only the Python
standard library and therefore cannot interfere with the heavy-weight training
runtime (PyTorch, multiprocessing, etc.).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROGRESS_PATH = PROJECT_ROOT / "logs" / "alpha_zero_training" / "training_progress.json"
COUNTER_PATH = PROJECT_ROOT / "logs" / "alpha_zero_training" / "iteration_counter.txt"
FREEZE_TAG_PATH = PROJECT_ROOT / "models" / "alpha_zero_checkpoints" / "iteration_freeze_start.tag"

POLL_INTERVAL_SECONDS = 15.0  # Adjust if you need finer granularity

def _safe_read_progress() -> Optional[int]:
    """Read the current iteration from the progress file.

    Returns None if the file does not exist or cannot be parsed.
    """
    if not PROGRESS_PATH.exists():
        return None
    try:
        with PROGRESS_PATH.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        return int(data.get("iteration", 0))
    except Exception:
        # File might be mid-write; skip this cycle.
        return None


def _write_counter(iteration: int) -> None:
    COUNTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with COUNTER_PATH.open("w", encoding="utf-8") as fp:
        fp.write(f"{iteration}\n")


def _write_freeze_tag(iteration: int) -> None:
    FREEZE_TAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tag_data = {
        "iteration": iteration,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with FREEZE_TAG_PATH.open("w", encoding="utf-8") as fp:
        json.dump(tag_data, fp, indent=2)


def main() -> None:
    last_iteration: Optional[int] = None

    while True:
        iteration = _safe_read_progress()
        if iteration is not None:
            # Write/update simple counter
            _write_counter(iteration)

            # Emit freeze tag once
            if not FREEZE_TAG_PATH.exists():
                _write_freeze_tag(iteration)
                print(f"[iteration_watcher] Freeze tag created at iteration {iteration}.")

            # Log to stdout on change
            if last_iteration != iteration:
                print(f"[iteration_watcher] Iteration updated → {iteration}")
                last_iteration = iteration
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[iteration_watcher] Stopped by user.") 