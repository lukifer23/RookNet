import random
from typing import Dict

import chess
import numpy as np
from utils.move_encoder import encode_move


def select_move(policy: np.ndarray, temperature: float, board: chess.Board) -> chess.Move:
    """Select a legal move using the given policy and temperature."""
    legal_moves_map: Dict[int, chess.Move] = {
        encode_move(mv): mv for mv in board.legal_moves if encode_move(mv) is not None
    }
    if not legal_moves_map:
        return random.choice(list(board.legal_moves)) if list(board.legal_moves) else None

    legal_indices = list(legal_moves_map.keys())
    legal_policy = policy[legal_indices]

    if np.sum(legal_policy) == 0:
        return legal_moves_map.get(random.choice(legal_indices))

    if temperature == 0:
        move_idx = legal_indices[int(np.argmax(legal_policy))]
    else:
        distribution = np.power(legal_policy, 1.0 / temperature)
        distribution /= np.sum(distribution)
        move_idx = np.random.choice(legal_indices, p=distribution)

    return legal_moves_map.get(move_idx)


def create_remote_evaluator(req_q, res_q, device="cpu"):
    """Factory that returns an evaluator(board_tensor) callable using queues."""
    import os
    import torch
    import numpy as np

    def _eval(board_tensor: torch.Tensor):
        uid = os.urandom(16)
        req_q.put((uid, board_tensor.squeeze(0).cpu().numpy()))
        while True:
            try:
                resp_uid, policy_np, value = res_q.get(timeout=120)
            except Exception:
                raise RuntimeError("Inference server timeout after 120s; check GPU server logs")
            if resp_uid == uid:
                policy = torch.from_numpy(np.asarray(policy_np)).float()
                val = torch.tensor(value, dtype=torch.float32)
                return val, policy

    return _eval
