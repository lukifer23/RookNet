from __future__ import annotations

import asyncio
from typing import Optional, Union

import chess
import chess.engine

from .base_engine import BaseEngine


class StockfishEngine(BaseEngine):
    """Asynchronous wrapper around the classic Stockfish UCI engine.

    The interface conforms to :class:`BaseEngine`, exposing a single
    :pyasync:`select_move` coroutine.  This allows callers to await it
    concurrently with other async engines (e.g. LLM-based ones) and keeps
    the API consistent across all engine types.
    """

    def __init__(
        self,
        path: str = "stockfish",
        *,
        depth: Optional[int] = 15,
        time_limit: Optional[float] = None,
    ) -> None:
        self.path = path
        self.depth = depth
        self.time_limit = time_limit
        # Spawn the UCI process immediately; failures surface at construction.
        # We deliberately allow the exception to propagate – better fail-fast.
        self._engine = chess.engine.SimpleEngine.popen_uci(self.path)

    async def select_move(self, board: chess.Board) -> chess.Move:  # noqa: D401
        """Return Stockfish's best move for *board* with the current limits."""
        limit = chess.engine.Limit(depth=self.depth, time=self.time_limit)
        # The underlying `play` call is blocking, so we off-load to a thread-pool
        # to avoid blocking the event-loop.
        result = await asyncio.to_thread(self._engine.play, board, limit)
        return result.move

    async def aclose(self) -> None:  # noqa: D401
        """Gracefully terminate the underlying UCI process."""
        await asyncio.to_thread(self._engine.quit)

    def configure(self, options: "dict[str, Union[int, str]]"):
        """Forward *options* to the underlying UCI engine."""
        self._engine.configure(options)

    async def analyse(self, board: chess.Board, limit: chess.engine.Limit):
        """Async wrapper around Stockfish's analyse call (blocking)."""
        return await asyncio.to_thread(self._engine.analyse, board, limit)

    # Convenience sync wrapper for legacy code
    def analyse_sync(self, board: chess.Board, limit: chess.engine.Limit):
        return asyncio.run(self.analyse(board, limit))


# ---------------------------------------------------------------------------
# Convenience synchronous wrapper – mirrors :pyfunc:`gemini_engine.select_move_sync`.
# ---------------------------------------------------------------------------

async def _async_select(engine: "StockfishEngine", board: chess.Board):
    return await engine.select_move(board)


def select_move_sync(engine: "StockfishEngine", board: chess.Board) -> chess.Move:
    """Blocking helper for legacy sync call-sites."""
    return asyncio.run(_async_select(engine, board)) 