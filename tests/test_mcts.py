import chess
import pytest

# Skip entire module if torch is unavailable
torch = pytest.importorskip("torch")

from search.mcts import MCTS
from utils.move_encoder import get_policy_vector_size
from models.base_model import BaseModel


class DummyModel(BaseModel):
    def forward(self, x):
        batch = x.shape[0]
        value = torch.zeros(batch)
        policy = torch.zeros(batch, get_policy_vector_size())
        return value, policy


def test_mcts_root_visit_count():
    config = {
        "training": {
            "mcts": {
                "simulations": 5,
                "c_puct": 1.0,
                "dirichlet_alpha": 0.0,
                "dirichlet_epsilon": 0.0,
            }
        }
    }
    mcts = MCTS(DummyModel(), config, device="cpu")
    board = chess.Board()
    mcts.search(board, simulations=5)
    assert mcts.root.visit_count == 6


def _mcts(config_overrides=None):
    base_config = {
        "training": {
            "mcts": {
                "simulations": 1,
                "c_puct": 1.0,
                "dirichlet_alpha": 0.0,
                "dirichlet_epsilon": 0.0,
            }
        }
    }
    if config_overrides:
        base_config["training"]["mcts"].update(config_overrides)
    return MCTS(DummyModel(), base_config, device="cpu")


@pytest.mark.parametrize(
    "fen",
    [
        "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",  # checkmate
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",  # stalemate
    ],
)
def test_mcts_search_game_over_positions(fen):
    mcts = _mcts()
    board = chess.Board(fen)
    with pytest.raises(ValueError):
        mcts.search(board, simulations=1)


def test_mcts_zero_simulations_returns_move():
    mcts = _mcts()
    board = chess.Board()
    move = mcts.search(board, simulations=0)
    assert isinstance(move, chess.Move)
