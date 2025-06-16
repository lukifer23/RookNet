"""
Chess Environment Utilities

This module provides core utilities for chess game handling, board representation,
and integration with chess engines like Stockfish.
"""

import chess
import chess.engine
import chess.pgn
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path
import torch
from .config_loader import load_config

logger = logging.getLogger(__name__)


class ChessEnvironment:
    """
    Manages the chess board state and engine interaction.
    """
    def __init__(self, config=None):
        """Lightweight wrapper around python-chess board/engine utilities.

        The *config* argument is optional now that the project is an installable
        package.  When omitted we lazily load the default `configs/config.v2.yaml`
        to keep backwards compatibility.
        """
        if config is None:
            try:
                config = load_config("configs/config.v2.yaml")
            except Exception:
                config = {}

        self.config = config
        self.board = chess.Board()
        self.engine = None
        
        # Piece mappings for tensor representation
        self.piece_to_idx = {
            chess.PAWN: 0, chess.ROOK: 1, chess.KNIGHT: 2,
            chess.BISHOP: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        self.idx_to_piece = {v: k for k, v in self.piece_to_idx.items()}
        
    def start_engine(self):
        """Initializes and starts the Stockfish engine."""
        if self.engine:
            self.stop_engine()
        
        stockfish_path = self.config.get('evaluation', {}).get('stockfish', {}).get('path', 'stockfish')
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Stockfish engine not found at '{stockfish_path}'. Please install Stockfish or update the path in your config file.")

    def stop_engine(self):
        """Stops the chess engine."""
        if self.engine:
            self.engine.quit()
            self.engine = None

    def get_engine_evaluation(self, time_limit=0.1):
        """Gets the evaluation of the current position from the engine."""
        if not self.engine:
            self.start_engine()
        
        info = self.engine.analyse(self.board, chess.engine.Limit(time=time_limit))
        return info['score'].relative.score(mate_score=10000)

    def get_engine_move(self, time_limit=1.0):
        """Gets the best move from the engine."""
        if not self.engine:
            self.start_engine()
            
        result = self.engine.play(self.board, chess.engine.Limit(time=time_limit))
        return result.move, None

    def __del__(self):
        self.stop_engine()

    # ------------------------------------------------------------------
    # Static / helper methods expected by training + MCTS pipelines
    # ------------------------------------------------------------------

    @staticmethod
    def board_to_tensor(board: chess.Board):
        """Convert *board* to a NumPy (12, 8, 8) float32 array.

        Delegates to *board_utils.board_to_tensor* (which returns a Torch
        tensor) and converts to NumPy to match the replay-buffer pipeline.
        """
        from utils.board_utils import board_to_tensor as _to_tensor

        tensor = _to_tensor(board)  # torch.Tensor [12,8,8]
        return tensor.numpy()

    # ------------------------------------------------------------------
    def select_move_with_temperature(self, board: chess.Board, policy: np.ndarray, temperature: float = 1.0):
        """Sample a legal move from *policy* using Boltzmann/temperature.

        Args
        ----
        board:       current python-chess *Board*.
        policy:      1-D NumPy probability vector sized to *policy_head_output*.
        temperature: >0 for stochastic; ==0 picks arg-max (greedy).
        """
        from utils.move_encoder import move_to_index, get_policy_vector_size

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        if temperature <= 0:
            # Greedy selection
            best_move = max(legal_moves, key=lambda m: policy[move_to_index(m)])
            return best_move

        # Convert logits to probabilities with temperature scaling
        probs = np.array([policy[move_to_index(m)] for m in legal_moves], dtype=np.float64)
        if probs.sum() == 0:
            # Fallback to uniform if NN gave zero to all legal moves
            probs = np.ones(len(legal_moves), dtype=np.float64)

        # Apply temperature: p_i ^ (1/temperature)
        if temperature != 1.0:
            probs = np.power(probs, 1.0 / temperature)
        probs /= probs.sum()

        move_idx = np.random.choice(len(legal_moves), p=probs)
        return legal_moves[move_idx]

    # ------------------------------------------------------------------
    @staticmethod
    def get_result(board: chess.Board) -> int:
        """Return game result from White's perspective: 1, ‑1 or 0."""
        outcome = board.result(claim_draw=True)
        if outcome == "1-0":
            return 1
        if outcome == "0-1":
            return -1
        return 0


# Example usage and testing
if __name__ == "__main__":
    # Basic test
    env = ChessEnvironment()
    
    print("Chess Environment Test")
    print("=" * 30)
    
    # Test board representation
    tensor = env.board_to_tensor()
    print(f"Board tensor shape: {tensor.shape}")
    print(f"FEN: {env.board_to_fen()}")
    
    # Test legal moves
    legal_moves = env.get_legal_moves()
    print(f"Number of legal moves: {len(legal_moves)}")
    print(f"First few moves: {[move.uci() for move in legal_moves[:5]]}")
    
    # Test engine (if available)
    try:
        env.start_engine()
        best_move, info = env.get_engine_move(time_limit=0.1)
        eval_score = env.get_engine_evaluation(time_limit=0.1)
        
        print(f"Engine best move: {best_move}")
        print(f"Position evaluation: {eval_score} centipawns")
        
        env.stop_engine()
        
    except Exception as e:
        print(f"Engine test failed: {e}")
        print("Install Stockfish with: brew install stockfish")
    
    print("\nChess environment ready for neural network training! 🚀")
