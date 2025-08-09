import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import chess

from training.training_loop import TrainingLoop, TrainingConfig
from training.worker import SelfPlayWorker, WorkerState


class DummyWorker(SelfPlayWorker):
    def __init__(self):
        # initialise with stub mcts; it won't be used
        self.moves_played = 0
        super().__init__(mcts=self)
        self.state = WorkerState(board=chess.Board())

    def select_move(self, board: chess.Board):  # type: ignore[override]
        move = next(iter(board.legal_moves))
        return move

    def play_move(self):  # override to count
        move = self.select_move(self.state.board)
        self.state.board.push(move)
        self.moves_played += 1
        return move


def test_training_loop_runs_moves():
    worker = DummyWorker()
    loop = TrainingLoop(worker, TrainingConfig(num_moves=2))
    state = loop.run()
    assert len(state.moves) == 2
    assert worker.moves_played == 2
