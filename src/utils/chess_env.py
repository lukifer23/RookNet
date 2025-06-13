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

logger = logging.getLogger(__name__)


class ChessEnvironment:
    """
    A comprehensive chess environment for neural network training and evaluation.
    
    Features:
    - Board state representation (multiple formats)
    - Move encoding/decoding
    - Stockfish integration
    - Game state management
    """
    
    def __init__(self, stockfish_path: Optional[str] = None):
        """
        Initialize the chess environment.
        
        Args:
            stockfish_path: Path to Stockfish engine binary
        """
        self.board = chess.Board()
        self.stockfish_path = stockfish_path or self._find_stockfish()
        self.engine = None
        
        # Piece mappings for tensor representation
        self.piece_to_idx = {
            chess.PAWN: 0, chess.ROOK: 1, chess.KNIGHT: 2,
            chess.BISHOP: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        self.idx_to_piece = {v: k for k, v in self.piece_to_idx.items()}
        
    def _find_stockfish(self) -> str:
        """Try to find Stockfish installation automatically."""
        possible_paths = [
            "/opt/homebrew/bin/stockfish",  # Homebrew on M1/M2/M3 Macs
            "/usr/local/bin/stockfish",     # Homebrew on Intel Macs
            "/usr/bin/stockfish",           # System installation
            "stockfish"                     # In PATH
        ]
        
        for path in possible_paths:
            if Path(path).exists() or path == "stockfish":
                return path
                
        raise FileNotFoundError("Stockfish not found. Please install with 'brew install stockfish'")
    
    def start_engine(self, **engine_options) -> None:
        """Start the Stockfish engine with specified options."""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            
            # Configure engine options (avoid automatically managed options)
            default_options = {
                "Threads": 4,
                "Hash": 128,  # MB
            }
            default_options.update(engine_options)
            
            # Only configure options that are available and not automatically managed
            configurable_options = {}
            for option, value in default_options.items():
                if option in self.engine.options and not self.engine.options[option].is_managed():
                    configurable_options[option] = value
            
            if configurable_options:
                self.engine.configure(configurable_options)
                    
            logger.info(f"Stockfish engine started: {self.engine.id}")
            
        except Exception as e:
            logger.error(f"Failed to start engine: {e}")
            raise
    
    def stop_engine(self) -> None:
        """Stop the Stockfish engine."""
        if self.engine:
            self.engine.quit()
            self.engine = None
    
    def reset_board(self) -> None:
        """Reset the chess board to starting position."""
        self.board.reset()
    
    def board_to_tensor(self, board: Optional[chess.Board] = None) -> np.ndarray:
        """
        Convert chess board to tensor representation.
        
        Args:
            board: Chess board to convert (uses self.board if None)
            
        Returns:
            numpy array of shape (12, 8, 8) representing the board state
            Channels 0-5: White pieces (Pawn, Rook, Knight, Bishop, Queen, King)
            Channels 6-11: Black pieces (Pawn, Rook, Knight, Bishop, Queen, King)
        """
        if board is None:
            board = self.board
            
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                # Get piece type and color
                piece_idx = self.piece_to_idx[piece.piece_type]
                color_offset = 0 if piece.color == chess.WHITE else 6
                channel = piece_idx + color_offset
                
                # Convert square to row, col (chess uses different convention)
                row = 7 - (square // 8)  # Flip row (chess uses rank 1-8)
                col = square % 8
                
                tensor[channel, row, col] = 1.0
        
        return tensor
    
    def board_to_fen(self, board: Optional[chess.Board] = None) -> str:
        """Get FEN representation of board."""
        if board is None:
            board = self.board
        return board.fen()
    
    def move_to_uci(self, move: chess.Move) -> str:
        """Convert move to UCI notation."""
        return move.uci()
    
    def uci_to_move(self, uci: str) -> chess.Move:
        """Convert UCI notation to move object."""
        return chess.Move.from_uci(uci)
    
    def get_legal_moves(self, board: Optional[chess.Board] = None) -> List[chess.Move]:
        """Get all legal moves for current position."""
        if board is None:
            board = self.board
        return list(board.legal_moves)
    
    def move_to_indices(self, move: chess.Move) -> Tuple[int, int]:
        """
        Convert move to from/to square indices.
        
        Returns:
            Tuple of (from_square, to_square) indices (0-63)
        """
        return move.from_square, move.to_square
    
    def indices_to_move(self, from_sq: int, to_sq: int, 
                       promotion: Optional[int] = None) -> chess.Move:
        """Convert square indices to move object."""
        return chess.Move(from_sq, to_sq, promotion)
    
    def move_to_index(self, move: chess.Move) -> int:
        """
        Convert move to single index for neural network output.
        Maps from_square (0-63) and to_square (0-63) to index (0-4095).
        
        Args:
            move: Chess move to convert
            
        Returns:
            Move index (0-4095)
        """
        return move.from_square * 64 + move.to_square
    
    def index_to_move(self, index: int) -> Tuple[int, int]:
        """
        Convert single index back to from/to squares.
        
        Args:
            index: Move index (0-4095)
            
        Returns:
            Tuple of (from_square, to_square)
        """
        from_square = index // 64
        to_square = index % 64
        return from_square, to_square
    
    def make_move(self, move: chess.Move, board: Optional[chess.Board] = None) -> bool:
        """
        Make a move on the board.
        
        Args:
            move: Move to make
            board: Board to move on (uses self.board if None)
            
        Returns:
            True if move was legal and made, False otherwise
        """
        if board is None:
            board = self.board
            
        if move in board.legal_moves:
            board.push(move)
            return True
        return False
    
    def unmake_move(self, board: Optional[chess.Board] = None) -> Optional[chess.Move]:
        """Unmake the last move."""
        if board is None:
            board = self.board
            
        if board.move_stack:
            return board.pop()
        return None
    
    def get_engine_move(self, time_limit: float = 1.0, 
                       depth: Optional[int] = None) -> Tuple[Optional[chess.Move], Dict[str, Any]]:
        """
        Get best move from Stockfish engine.
        
        Args:
            time_limit: Time limit in seconds
            depth: Search depth (if None, uses time limit)
            
        Returns:
            Tuple of (best_move, info_dict)
        """
        if not self.engine:
            raise RuntimeError("Engine not started. Call start_engine() first.")
        
        if depth:
            result = self.engine.analyse(self.board, chess.engine.Limit(depth=depth))
        else:
            result = self.engine.analyse(self.board, chess.engine.Limit(time=time_limit))
        
        best_move = result.get("pv", [None])[0] if result.get("pv") else None
        
        return best_move, dict(result)
    
    def get_engine_evaluation(self, time_limit: float = 1.0) -> float:
        """
        Get position evaluation from Stockfish in centipawns.
        
        Returns:
            Evaluation in centipawns (positive = White advantage)
        """
        if not self.engine:
            raise RuntimeError("Engine not started. Call start_engine() first.")
        
        result = self.engine.analyse(self.board, chess.engine.Limit(time=time_limit))
        
        score = result.get("score")
        if score:
            white_score = score.white()
            if white_score.is_mate():
                # Convert mate scores to large centipawn values
                mate_in = white_score.mate()
                return 10000 * (1 if mate_in and mate_in > 0 else -1)
            else:
                # Return centipawn score from White's perspective
                cp_score = white_score.score()
                return cp_score if cp_score is not None else 0
        
        return 0
    
    def is_game_over(self, board: Optional[chess.Board] = None) -> bool:
        """Check if game is over."""
        if board is None:
            board = self.board
        return board.is_game_over()
    
    def get_game_result(self, board: Optional[chess.Board] = None) -> Optional[str]:
        """Get game result (1-0, 0-1, 1/2-1/2, or None)."""
        if board is None:
            board = self.board
        
        if not board.is_game_over():
            return None
            
        result = board.result()
        return result
    
    def clone_board(self, board: Optional[chess.Board] = None) -> chess.Board:
        """Create a copy of the board."""
        if board is None:
            board = self.board
        return board.copy()


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
