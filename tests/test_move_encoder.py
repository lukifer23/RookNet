import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from utils.move_encoder import (  # noqa: E402
    decode_move,
    encode_move,
    get_policy_vector_size,
)


def test_move_encoder_round_trip():
    size = get_policy_vector_size()
    for idx in range(size):
        move = decode_move(idx)
        assert encode_move(move) == idx
