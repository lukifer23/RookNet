import numpy as np
import chess
import pytest

from utils.chess_env import ChessEnvironment
from utils.move_encoder import get_policy_vector_size, move_to_index


def test_board_to_tensor_shape():
    env = ChessEnvironment()
    tensor = env.board_to_tensor(env.board)
    assert tensor.shape == (12, 8, 8)
    assert tensor.dtype == np.float32


def test_select_move_with_temperature_zero():
    env = ChessEnvironment()
    board = env.board
    size = get_policy_vector_size()
    policy = np.zeros(size, dtype=np.float64)
    first_move = list(board.legal_moves)[0]
    policy[move_to_index(first_move)] = 1.0
    move = env.select_move_with_temperature(board, policy, temperature=0)
    assert move == first_move


def test_get_result_variants():
    assert ChessEnvironment.get_result(chess.Board()) == 0
    assert ChessEnvironment.get_result(
        chess.Board(
            "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
        )
    ) == 1
    assert ChessEnvironment.get_result(
        chess.Board(
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        )
    ) == -1
    assert ChessEnvironment.get_result(
        chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    ) == 0
