from dataclasses import dataclass
from typing import Optional, Any

import chess

from search.mcts import MCTS


@dataclass
class MCTSConfig:
    """Configuration for MCTS interaction."""

    simulations: int = 32
    c_puct: float = 1.0

    def to_dict(self) -> dict:
        """Convert to a minimal config dict expected by :class:`search.mcts.MCTS`."""
        return {
            "training": {
                "mcts": {
                    "simulations": self.simulations,
                    "c_puct": self.c_puct,
                    "dirichlet_alpha": 0.0,
                    "dirichlet_epsilon": 0.0,
                }
            }
        }


class MCTSRunner:
    """Thin wrapper around :class:`search.mcts.MCTS` with structured config."""

    def __init__(self, model: Any, config: Optional[MCTSConfig] = None, device: str = "cpu"):
        self.config = config or MCTSConfig()
        self.mcts = MCTS(model, self.config.to_dict(), device=device)

    def select_move(self, board: chess.Board) -> chess.Move:
        """Run MCTS search and return the selected move."""
        return self.mcts.search(board, simulations=self.config.simulations)
