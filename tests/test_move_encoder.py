import chess
import pytest

from src.utils.move_encoder import encode_move, decode_move, get_policy_vector_size, MOVE_TO_INDEX, INDEX_TO_MOVE


def test_policy_vector_size_consistency():
    """MOVE_TO_INDEX and INDEX_TO_MOVE must be bijective and sized per helper."""
    assert len(MOVE_TO_INDEX) == len(INDEX_TO_MOVE)
    assert len(MOVE_TO_INDEX) == get_policy_vector_size()


def test_encode_decode_bijection_on_initial_board():
    """For all legal moves in the starting position, encode then decode should yield the same UCI."""
    board = chess.Board()
    for move in board.legal_moves:
        idx = encode_move(move)
        assert idx is not None, f"encode_move returned None for legal move {move}"
        decoded = decode_move(idx)
        assert decoded is not None, f"decode_move returned None for index {idx} (move {move})"
        assert decoded.uci() == move.uci(), "Round-trip encode/decode mismatch" 