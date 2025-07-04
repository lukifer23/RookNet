import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import chess
import pytest

# Skip entire module if torch is unavailable
torch = pytest.importorskip("torch")

from search.mcts import MCTS
from utils.move_encoder import get_policy_vector_size
from utils.board_utils import board_to_tensor
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
