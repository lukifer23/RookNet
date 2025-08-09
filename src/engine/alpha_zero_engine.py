from __future__ import annotations

import asyncio

import chess
import numpy as np
import torch

from engine.base_engine import BaseEngine
from models.base_model import BaseModel
from search.mcts import MCTS


class AlphaZeroEngine(BaseEngine):
    """
    Neural engine that wraps the production model and new MCTS search.
    Provides an async interface for responsive GUI integration.
    """

    def __init__(
        self,
        model: BaseModel,
        config: dict,
        device: torch.device,
        *,
        temperature: float = 0.1,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.mcts = MCTS(self.model, self.config, self.device)
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Public API – BaseEngine implementation
    # ------------------------------------------------------------------
    async def select_move(self, board: chess.Board) -> chess.Move:
        """Off-loads the expensive MCTS search to a worker thread."""
        best_move = await asyncio.to_thread(self.mcts.search, board)

        # The new MCTS search directly returns the best move based on visit counts.
        # If temperature is > 0, we could sample from the visit counts distribution.
        # For now, we will use the most visited move as determined by MCTS.

        if self.temperature > 0 and self.mcts.root and self.mcts.root.children:
            return self._select_move_with_temp(board)

        return best_move

    def _select_move_with_temp(self, board: chess.Board) -> chess.Move:
        """Selects a move from the root children based on visit counts and temperature."""
        root = self.mcts.root
        if not root or not root.children:
            return next(iter(board.legal_moves))

        moves = list(root.children.keys())
        visit_counts = np.array(
            [child.visit_count for child in root.children.values()], dtype=np.float32
        )

        # Apply temperature to visit counts
        powered_visits = np.power(visit_counts, 1.0 / self.temperature)
        probs = powered_visits / np.sum(powered_visits)

        # Sample a move based on the new probabilities
        selected_move_index = np.random.choice(len(moves), p=probs)
        return moves[selected_move_index]

    # No external resources to close, but keep interface symmetrical.
    async def aclose(self):  # noqa: D401
        pass


# ---------------------------------------------------------------------------
# Convenience synchronous wrapper
# ---------------------------------------------------------------------------


async def _async_select(engine: "AlphaZeroEngine", board: chess.Board):
    return await engine.select_move(board)


def select_move_sync(engine: "AlphaZeroEngine", board: chess.Board) -> chess.Move:
    """Convenience synchronous wrapper."""
    return asyncio.run(engine.select_move(board))
