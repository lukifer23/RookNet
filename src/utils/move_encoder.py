# src/utils/move_encoder.py
"""Flat 4096-index move encoder.

The mapping is simply ``index = from_square * 64 + to_square``.  Promotions are
collapsed to the underlying *to* square (queen promotion implicit).  This keeps
the policy head compact and fully compatible with AlphaZero-style training.
"""

import chess

POLICY_VECTOR_SIZE = 4096  # 64 × 64


def get_policy_vector_size() -> int:  # noqa: D401
    return POLICY_VECTOR_SIZE


def move_to_index(move: chess.Move) -> int:
    """Map *move* to flat 0-4095 index.

    Any promotion piece information is ignored; promotions map to the same
    index as the non-promotion move (queen by convention).
    """
    return move.from_square * 64 + move.to_square


def index_to_move(index: int, board: chess.Board) -> chess.Move:  # noqa: D401
    """Inverse of :pyfunc:`move_to_index`.

    For pawn moves that reach the last rank we default to *queen promotion*.
    """
    from_square = index // 64
    to_square = index % 64

    piece = board.piece_at(from_square)
    if piece and piece.piece_type == chess.PAWN:
        to_rank = to_square // 8
        if (piece.color == chess.WHITE and to_rank == 7) or (
            piece.color == chess.BLACK and to_rank == 0
        ):
            return chess.Move(from_square, to_square, promotion=chess.QUEEN)

    return chess.Move(from_square, to_square)


def encode_move(move: chess.Move) -> int:  # alias for compatibility
    idx = move_to_index(move)
    return idx if idx < POLICY_VECTOR_SIZE else None


def decode_move(index: int) -> chess.Move:  # alias for compatibility
    return index_to_move(index, chess.Board())

def get_legal_move_mask(board: chess.Board) -> list[int]:
    """
    Returns a list of indices corresponding to the legal moves in the current position.
    """
    legal_move_indices = []
    for move in board.legal_moves:
        idx = encode_move(move)
        if idx is not None:
            legal_move_indices.append(idx)
    return legal_move_indices 