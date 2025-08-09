import chess
from utils.move_encoder import encode_move, decode_move, get_policy_vector_size


def test_move_encoder_round_trip():
    size = get_policy_vector_size()
    for idx in range(size):
        move = decode_move(idx)
        assert encode_move(move) == idx
