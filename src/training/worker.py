from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import chess

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from .mcts_interface import MCTSRunner


@dataclass
class WorkerState:
    """Mutable state for a self-play worker."""

    board: chess.Board


class SelfPlayWorker:
    """Worker that uses an MCTS-like object to play moves."""

    def __init__(self, mcts: Any):
        self.mcts = mcts
        self.state = WorkerState(board=chess.Board())

    def play_move(self) -> chess.Move:
        """Select and play a move on the internal board."""
        move = self.mcts.select_move(self.state.board)
        self.state.board.push(move)
        return move
