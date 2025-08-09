"""
Chess Utilities for AI Training
Shared functions for chess position processing, data handling, and model utilities
"""

from typing import Dict, Tuple

import chess
import numpy as np

from utils.chess_env import ChessEnvironment


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Convert chess board to tensor representation
    Uses the ChessEnvironment board_to_tensor method

    Args:
        board: chess.Board object

    Returns:
        np.ndarray: Board tensor [12, 8, 8]
    """
    env = ChessEnvironment()
    return env.board_to_tensor(board)


class ChessDataProcessor:
    """Chess data processing utilities"""

    def __init__(self):
        self.env = ChessEnvironment()

    def process_position(self, board: chess.Board) -> Dict:
        """Process chess position into training format"""
        return {
            "tensor": self.env.board_to_tensor(board),
            "fen": board.fen(),
            "legal_moves": list(board.legal_moves),
            "is_terminal": board.is_game_over(),
        }

    def move_to_index(self, move: chess.Move) -> int:
        """Convert move to index (simplified encoding)"""
        return move.from_square * 64 + move.to_square

    def index_to_move(self, index: int) -> Tuple[int, int]:
        """Convert index back to move squares"""
        from_square = index // 64
        to_square = index % 64
        return from_square, to_square
