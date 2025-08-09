"""Centralised GPU inference server for AlphaZero self-play workers.

This process owns the compiled `ChessTransformer` on the CUDA GPU and
services batched inference requests coming from CPU workers via two
multiprocessing queues:

    request_q.put((uuid_bytes, board_numpy))
    response_q.get() -> (uuid_bytes, policy_numpy, value_float)

The caller must ensure `request_q` and `response_q` are `multiprocessing.Queue`
objects created by a `multiprocessing.Manager()` to allow cross-process
transfer.
"""

from __future__ import annotations

import logging
import os
import queue
import signal
import time
import warnings
from contextlib import nullcontext
from types import FrameType
from typing import Tuple

import numpy as np
import torch
from torch.cuda.amp import autocast

from models.chess_transformer import ChessTransformer

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

BATCH_LIMIT = int(os.environ.get("NCT_GPU_BATCH", 128))  # max requests per batch
SLEEP_MS = 0.1  # polling interval when no requests (Manager.Queue latency)
LOGGER = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Silence verbose Inductor/Dynamo warnings – keep logs readable
# ------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
for noisy in ("torch._dynamo", "torch._inductor"):
    logging.getLogger(noisy).setLevel(logging.ERROR)


def _load_model(checkpoint_path: str | None, device: torch.device) -> ChessTransformer:
    """Create and compile the model, optionally loading checkpoint weights."""
    # Minimal config – adjust if you expose CLI flags later
    from utils.config_loader import load_config

    cfg = load_config("configs/config.v2.yaml")
    model_cfg = cfg["model"]["chess_transformer"]

    model = ChessTransformer(
        input_channels=model_cfg["input_channels"],
        cnn_channels=model_cfg["cnn_channels"],
        cnn_blocks=model_cfg["cnn_blocks"],
        transformer_layers=model_cfg["transformer_layers"],
        attention_heads=model_cfg["attention_heads"],
        policy_head_output_size=4096,
    ).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        LOGGER.info("Loading weights from %s", checkpoint_path)
        state = torch.load(checkpoint_path, map_location=device)
        sd = state.get("model_state_dict", state)
        model.load_state_dict(sd, strict=False)

    # ------------------------------------------------------------------
    # Compile only on CUDA (A100 / RTX etc.).
    # ------------------------------------------------------------------
    if device.type == "cuda":
        try:
            import torch._dynamo as _td

            _td.config.suppress_errors = True
            model = torch.compile(model, backend="inductor", mode="reduce-overhead")
            LOGGER.info("torch.compile succeeded → using compiled graph (cuda)")
        except Exception as exc:
            LOGGER.warning(
                "torch.compile failed on cuda: %s – falling back to eager", exc
            )
            torch._dynamo.reset()

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main server loop
# ---------------------------------------------------------------------------


def run_inference_server(
    request_q,  # multiprocessing.Queue
    response_q,  # multiprocessing.Queue
    checkpoint_path: str | None = None,
    device_str: str = "cuda",
):
    """Entry point for the dedicated GPU inference process."""
    # Setup logging inside subprocess
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - gpu_server - %(levelname)s - %(message)s",
    )

    device = torch.device(
        device_str if device_str == "cuda" and torch.cuda.is_available() else "cpu"
    )
    LOGGER.info("GPU inference server starting on device %s", device)

    model = _load_model(checkpoint_path, device)

    # -------------------------------------------------------------------
    # One-time warm-up to trigger JIT/Inductor compile before clients arrive
    # -------------------------------------------------------------------
    try:
        dummy = torch.zeros(1, 12, 8, 8, device=device)
        with torch.no_grad():
            _ = model(dummy)
        LOGGER.info("Warm-up inference complete – kernels cached")
    except Exception as exc:
        LOGGER.warning("Warm-up inference failed: %s (continuing)", exc)

    running = True

    def _graceful_shutdown(signum: int, frame: FrameType | None):  # noqa: D401
        nonlocal running
        LOGGER.info("Received signal %s – shutting down inference server", signum)
        running = False

    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)

    pending: list[Tuple[bytes, np.ndarray]] = []

    while running:
        # Blocking read for the first item so we don't spin when the queue is empty.
        if not pending:
            try:
                uid, board_np = request_q.get(timeout=SLEEP_MS)
                pending.append((uid, board_np))
            except queue.Empty:
                continue  # no requests yet

        # Drain any additional requests without blocking
        try:
            while len(pending) < BATCH_LIMIT:
                uid, board_np = request_q.get_nowait()
                pending.append((uid, board_np))
        except queue.Empty:
            pass

        if not pending:
            time.sleep(SLEEP_MS)
            continue

        # Build batch tensor
        uids, boards = zip(*pending)
        LOGGER.debug("Running inference on batch size %d", len(boards))
        batch = torch.from_numpy(np.stack(boards)).float().to(device)

        # Use autocast only on CUDA to gain speed; avoid warnings on MPS/CPU
        amp_ctx = autocast() if device.type == "cuda" else nullcontext()

        with torch.no_grad(), amp_ctx:
            # ChessTransformer forward returns (value, policy)
            value_logits, policy_logits = model(batch)

        policies = torch.softmax(policy_logits, dim=1).cpu().numpy().astype(np.float16)
        # Ensure values is 1-D array with length = batch size
        values = torch.tanh(value_logits).view(-1).cpu().numpy().astype(np.float16)

        # Send responses
        for uid, pol, val in zip(uids, policies, values):
            response_q.put((uid, pol, float(val)))

        pending.clear()

    LOGGER.info("Inference server terminated")
