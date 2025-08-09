import chess
import torch


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Converts a chess.Board object to a tensor representation.
    The tensor has shape (12, 8, 8), where 12 channels represent
    the 6 piece types for both white and black.
    """
    piece_to_channel = {
        "p": 0,
        "n": 1,
        "b": 2,
        "r": 3,
        "q": 4,
        "k": 5,
        "P": 6,
        "N": 7,
        "B": 8,
        "R": 9,
        "Q": 10,
        "K": 11,
    }

    tensor = torch.zeros(12, 8, 8, dtype=torch.float32)

    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            channel = piece_to_channel[piece.symbol()]
            # Chess squares are mapped from a1 (0) to h8 (63)
            # We map this to a standard (row, col) format
            rank = i // 8
            file = i % 8
            tensor[channel, rank, file] = 1

    return tensor
