import os
import asyncio
from typing import Optional
import chess
import httpx
from .base_engine import BaseEngine

# Endpoint template – v1beta matches current public API (June-2025)
def _build_endpoint(model: str, api_version: str) -> str:
    return f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:generateContent"

# Prefer GA v1 for 2.x and later; keep v1beta for preview/experimental models.
def _preferred_api_version(model: str) -> str:
    if "preview" in model or "exp" in model:
        return "v1beta"
    # 1.x models are still fine on v1beta; use v1 for 2.x+ GA
    if model.startswith("gemini-2"):
        return "v1"
    return "v1beta"

_SYSTEM_PROMPT = (
    "You are a chess grandmaster. Respond with the single best move for the side to move "
    "in Universal Chess Interface (UCI) four-character notation.  Do not add any text, "
    "no commentary, no punctuation – only the move, e.g.: e2e4"
)

_USER_TEMPLATE = "FEN: {fen}\nMove history: {history}\nBest move?"


class GeminiEngine(BaseEngine):
    """LLM-backed chess engine using Google Gemini REST API."""

    def __init__(
        self,
        model: str = "gemini-1.5-pro-latest",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GeminiEngine requires an API key. Provide api_key arg or set GEMINI_API_KEY env var.")
        self.temperature = temperature
        self.timeout = timeout

    async def select_move(self, board: chess.Board) -> chess.Move:  # noqa: D401
        fen = board.fen()
        history = " ".join(move.uci() for move in board.move_stack)
        prompt = _USER_TEMPLATE.format(fen=fen, history=history if history else "—")

        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": _SYSTEM_PROMPT}]},
                {"role": "user", "parts": [{"text": prompt}]},
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": 20,
                "topP": 1,
                "topK": 1,
            },
        }

        api_version = _preferred_api_version(self.model)
        url = _build_endpoint(self.model, api_version)
        headers = {"x-goog-api-key": self.api_key}

        # Open a fresh HTTPX client tied to *this* event loop.
        async with httpx.AsyncClient(http2=True, timeout=self.timeout) as client:
            
            async def _post(u):
                r = await client.post(u, headers=headers, json=payload)
                r.raise_for_status()
                return r

            fallback_error = ""
            try:
                r = await _post(url)
            except httpx.HTTPStatusError as e:
                # If first attempt fails and we weren't using both versions, try the other.
                alt_version = "v1beta" if api_version == "v1" else "v1"
                alt_url = _build_endpoint(self.model, alt_version)
                try:
                    r = await _post(alt_url)
                except Exception:
                    # Log detailed error once and re-raise to outer handler.
                    print(f"[GeminiEngine] Request failed. Status {e.response.status_code}. Body: {e.response.text[:500]}")
                    raise

            # At this point request succeeded
            def _extract_move(resp: dict):
                """Return first token of model reply or None."""
                cand = resp.get("candidates", [{}])[0]
                content = cand.get("content")
                if content is None:
                    return None
                # Shape A: {'content': {'parts': [{'text': 'e2e4'}]}}
                if isinstance(content, dict):
                    if "parts" in content and content["parts"]:
                        txt = content["parts"][0].get("text", "")
                    else:
                        txt = content.get("text", "")  # Shape B: {'content': {'text': 'e2e4'}}
                else:
                    # Shape C: 'content' itself is the string
                    txt = str(content)
                txt = str(txt).strip()
                return txt.split()[0] if txt else None

            try:
                data = r.json()
                text = _extract_move(data)
            except Exception as e:
                text = None
                fallback_error = str(e)

            move = None
            if text:
                try:
                    move = chess.Move.from_uci(text.lower())
                    if move not in board.legal_moves:
                        move = None
                except Exception:
                    move = None

            if move is None:
                # Fallback: choose random legal move so the game continues
                import random

                move = random.choice(list(board.legal_moves))
                # Log the error but do not raise
                print(f"[GeminiEngine] Invalid LLM move '{text}'. Using random legal move instead. Error: {fallback_error}")

            return move

    async def aclose(self):
        # Nothing to close – client is per-call.
        pass


# Convenience synchronous wrapper (e.g. for tests)
async def _async_select(engine: GeminiEngine, board: chess.Board):
    return await engine.select_move(board)

def select_move_sync(engine: GeminiEngine, board: chess.Board) -> chess.Move:
    return asyncio.run(_async_select(engine, board)) 