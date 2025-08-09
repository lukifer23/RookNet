import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import chess

from training.worker import SelfPlayWorker, WorkerState


class StubMCTS:
    def __init__(self, move: chess.Move):
        self.move = move
        self.called = 0

    def select_move(self, board: chess.Board) -> chess.Move:
        self.called += 1
        return self.move


def test_worker_plays_selected_move():
    board = chess.Board()
    move = next(iter(board.legal_moves))
    worker = SelfPlayWorker(StubMCTS(move))
    worker.state = WorkerState(board=board)
    played = worker.play_move()
    assert played == move
    assert worker.mcts.called == 1
    assert worker.state.board.move_stack[-1] == move
