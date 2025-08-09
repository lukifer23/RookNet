"""Quick smoke-test for the GPU inference server pipeline.

Run with:
    python tools/smoke_test_gpu_server.py
Expect:
    ✔ policy shape (4096,) and value scalar printed in <2 s.
"""

import pathlib
import sys

# Ensure src package resolvable before project imports
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root / "src"))

import multiprocessing as mp  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import chess  # noqa: E402

from engine.gpu_inference_server import run_inference_server  # noqa: E402
from utils.board_utils import board_to_tensor  # noqa: E402


def main():
    mgr = mp.Manager()
    req_q, res_q = mgr.Queue(), mgr.Queue()

    # Launch server (uses best opponent checkpoint if available)
    ckpt = os.path.join("models", "alpha_zero_checkpoints", "best_opponent.pt")
    srv = mp.Process(
        target=run_inference_server, args=(req_q, res_q, ckpt, "cuda"), daemon=False
    )
    srv.start()

    # Prepare random board tensor
    board = chess.Board()
    tensor = board_to_tensor(board).numpy()

    uid = os.urandom(16)
    req_q.put((uid, tensor))

    start = time.time()
    while True:
        rid, policy, value = res_q.get()
        if rid == uid:
            break
    elapsed = time.time() - start

    print(
        f"✔ received in {elapsed*1000:.1f} ms → policy {policy.shape}, value {value:.3f}"
    )

    srv.terminate()
    srv.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
