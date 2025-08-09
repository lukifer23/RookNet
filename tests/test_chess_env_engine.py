import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import pytest
from unittest.mock import patch, MagicMock
import types
import sys as _sys

# Provide a stub for torch if it's not installed to allow importing ChessEnvironment
_sys.modules.setdefault("torch", types.SimpleNamespace())

from utils.chess_env import ChessEnvironment
_sys.modules.pop("torch", None)


def test_start_engine_absolute_path(tmp_path):
    dummy_engine = tmp_path / "stockfish"
    dummy_engine.write_text("")
    dummy_engine.chmod(0o755)
    env = ChessEnvironment(config={"evaluation": {"stockfish": {"path": str(dummy_engine)}}})
    with patch("chess.engine.SimpleEngine.popen_uci", return_value=MagicMock() ) as popen:
        env.start_engine()
        popen.assert_called_once_with(str(dummy_engine))


def test_start_engine_relative_path(tmp_path):
    dummy_engine = tmp_path / "stockfish"
    dummy_engine.write_text("")
    dummy_engine.chmod(0o755)
    env = ChessEnvironment(config={"evaluation": {"stockfish": {"path": "stockfish"}}})
    with patch("shutil.which", return_value=str(dummy_engine)) as which, \
         patch("chess.engine.SimpleEngine.popen_uci", return_value=MagicMock()) as popen:
        env.start_engine()
        which.assert_called_once_with("stockfish")
        popen.assert_called_once_with(str(dummy_engine))


def test_start_engine_missing_exec():
    env = ChessEnvironment(config={"evaluation": {"stockfish": {"path": "missing"}}})
    with patch("shutil.which", return_value=None):
        with pytest.raises(FileNotFoundError):
            env.start_engine()
