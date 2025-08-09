from dataclasses import dataclass, field
from typing import List

from .worker import SelfPlayWorker


@dataclass
class TrainingConfig:
    """Configuration for the simple training loop."""

    num_moves: int = 10


@dataclass
class TrainingState:
    """State accumulated during training."""

    moves: List[str] = field(default_factory=list)


class TrainingLoop:
    """Minimal training loop coordinating a worker."""

    def __init__(self, worker: SelfPlayWorker, config: TrainingConfig):
        self.worker = worker
        self.config = config
        self.state = TrainingState()

    def run(self) -> TrainingState:
        for _ in range(self.config.num_moves):
            if self.worker.state.board.is_game_over():
                break
            move = self.worker.play_move()
            self.state.moves.append(move.uci())
        return self.state
