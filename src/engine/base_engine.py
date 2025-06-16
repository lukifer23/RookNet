from typing import Protocol
import chess

class BaseEngine(Protocol):
    """Minimal interface any engine (neural-net, Stockfish, LLM, etc.) must provide."""

    async def select_move(self, board: chess.Board) -> chess.Move:  # noqa: D401
        """Return a legal move for the given position.

        Implementations must guarantee the move is legal; caller will not re-check.
        Block-and-wait or perform asynchronous HTTP calls as desired, but expose
        an *async* API so callers can await concurrently.
        """
        ... 