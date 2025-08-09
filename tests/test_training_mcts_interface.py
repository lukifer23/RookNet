import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import chess
import pytest

# Skip entire module if torch is unavailable
torch = pytest.importorskip("torch")

from training.mcts_interface import MCTSRunner, MCTSConfig
from utils.move_encoder import get_policy_vector_size
from models.base_model import BaseModel


class DummyModel(BaseModel):
    def forward(self, x):
        batch = x.shape[0]
        value = torch.zeros(batch)
        policy = torch.zeros(batch, get_policy_vector_size())
        return value, policy


def test_mcts_runner_selects_legal_move():
    board = chess.Board()
    config = MCTSConfig(simulations=1, c_puct=1.0)
    runner = MCTSRunner(DummyModel(), config, device="cpu")
    move = runner.select_move(board)
    assert move in board.legal_moves
