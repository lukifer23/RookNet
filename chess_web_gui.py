#!/usr/bin/env python3
"""
Chess AI Web GUI - Flask Backend
Lightweight server providing API endpoints for chess AI interaction
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import chess
import chess.engine
import torch
import numpy as np
import sys
import os
import logging
from pathlib import Path
import time
import subprocess
import threading

# Add src to path
# sys.path.append('src')
from models.chess_transformer import ChessTransformer
from utils.chess_env import ChessEnvironment
from utils.config_loader import load_config
from utils.move_encoder import get_policy_vector_size

# Import MCTS for move generation
# We need to import the classes, and the file itself to access the static methods
from search.mcts import MCTS
from src.training.alphazero_trainer import select_move

# --- Configuration & Global Setup ---
try:
    CONFIG = load_config("configs/config.v2.yaml")
    POLICY_SIZE = get_policy_vector_size()
except Exception as e:
    print(f"FATAL: Could not load config or setup policy size. Error: {e}")
    sys.exit(1)

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Strength level definitions for Stockfish
STOCKFISH_STRENGTHS = {
    "very_easy": {"depth": 1,  "time": 0.1,  "skill": 0},
    "easy":      {"depth": 3,  "time": 0.3,  "skill": 5},
    "medium":    {"depth": 5,  "time": 1.0, "skill": 10},
    "hard":      {"depth": 10, "time": 3.0, "skill": 15},
    "expert":    {"depth": 16, "time": 8.0, "skill": 18},
    "master":    {"depth": 20, "time": 15.0, "skill": 20},
}

# HopeChess root relative to project
HOPECHESS_DIR = Path(__file__).resolve().parent / 'HopeChess'
HOPECHESS_BIN = HOPECHESS_DIR / 'zig-out' / 'bin' / 'HopeChess'

# Strength profiles for HopeChess (uses depth only – time optional)
HOPECHESS_STRENGTHS = {
    'very_easy': {'depth': 1, 'time': 0.1},
    'easy': {'depth': 3, 'time': 0.3},
    'medium': {'depth': 5, 'time': 1.0},
    'hard': {'depth': 8, 'time': 3.0},
    'expert': {'depth': 12, 'time': 8.0},
    'master': {'depth': 16, 'time': 15.0},
}

class ChessAIServer:
    def __init__(self):
        # System and Model Config
        self.sys_config = CONFIG['system']
        self.model_config = CONFIG['model']['chess_transformer']
        self.checkpoint_config = CONFIG['training']['checkpoints']

        # Device
        self.device = torch.device(self.sys_config['device'] if self.sys_config['device'] != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
        logger.info(f"GUI using device: {self.device}")
        
        # Core Components
        self.model = self._load_production_model()
        # Wrap the model with AlphaZeroEngine abstraction for async/sync compatibility
        self.az_engine = AlphaZeroEngine(self.model, config=CONFIG, device=self.device) if self.model else None
        self.chess_env = ChessEnvironment()

        # Stockfish Engine (for comparison)
        self.stockfish_engine = None
        self._setup_stockfish()

        # HopeChess Engine
        self.hopechess_lock = threading.Lock()
        self.hopechess_engine = None
        self._setup_hopechess()

    def _load_production_model(self) -> ChessTransformer:
        """Loads the best-performing model from the AlphaZero pipeline."""
        logger.info("Attempting to load the best production model...")
        ckpt_dir = self.checkpoint_config['dir']
        # 1️⃣ Preferred: best opponent (promoted) model
        candidate_paths = [os.path.join(ckpt_dir, self.checkpoint_config['best_opponent_model'])]
        # 2️⃣ Fallback: latest player model (exact name or any *_latest_player.pt)
        candidate_paths.append(os.path.join(ckpt_dir, self.checkpoint_config['latest_player_model']))
        candidate_paths.extend([str(p) for p in Path(ckpt_dir).glob('*_latest_player.pt')])
        # 3️⃣ Fallback: newest iteration-specific checkpoint
        iter_ckpts = sorted(Path(ckpt_dir).glob('checkpoint_iter_*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
        if iter_ckpts:
            candidate_paths.append(str(iter_ckpts[0]))

        # Additionally consider any *_best_opponent.pt files in case naming prefixed
        candidate_paths.extend([str(p) for p in Path(ckpt_dir).glob('*_best_opponent.pt')])

        model_path = None
        for path in candidate_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            logger.error("FATAL: No suitable model checkpoint found in models/alpha_zero_checkpoints. The GUI will still run but neural engine options will be disabled.")
            return None

        try:
            model = ChessTransformer(
                input_channels=self.model_config['input_channels'],
                cnn_channels=self.model_config['cnn_channels'],
                cnn_blocks=self.model_config['cnn_blocks'],
                transformer_layers=self.model_config['transformer_layers'],
                attention_heads=self.model_config['attention_heads'],
                policy_head_output_size=POLICY_SIZE
            ).to(self.device)

            checkpoint = torch.load(model_path, map_location=self.device)
            
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Handle models saved with torch.compile()
            # The actual model is stored under _orig_mod.
            if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
                 state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict)
            model.eval()
            
            if self.sys_config.get('compile_model', False):
                if self.device.type != 'cpu':
                    logger.info("Compiling the model for the GUI...")
                    # Use a mode that is safe for dynamic shapes often found in GUIs
                    model = torch.compile(model, mode="reduce-overhead")

            logger.info(f"✅ Successfully loaded and prepared production model from {model_path}")
            return model

        except Exception as e:
            logger.error(f"CRITICAL ERROR loading production model: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _setup_stockfish(self):
        """Initializes the Stockfish engine."""
        stockfish_path = CONFIG['evaluation']['stockfish']['path']
        if not os.path.exists(stockfish_path):
            # Attempt to find stockfish in PATH if the direct path fails or isn't absolute
            if not os.path.isabs(stockfish_path):
                import shutil
                found_path = shutil.which(stockfish_path)
                if found_path:
                    stockfish_path = found_path
                else:
                    logger.warning(f"Stockfish executable '{stockfish_path}' not found in system PATH or as a direct path. Please check config.v2.yaml.")
                    self.stockfish_engine = None
                    return
            else:
                logger.warning(f"Stockfish not found at path: {stockfish_path}. Please check config.v2.yaml.")
                self.stockfish_engine = None
                return
        try:
            self.stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            logger.info(f"Stockfish engine initialized from: {stockfish_path}")
        except Exception as e:
            logger.warning(f"Could not initialize Stockfish engine: {e}")
            self.stockfish_engine = None

    def _setup_hopechess(self):
        """Builds (if needed) and initializes HopeChess UCI engine."""
        try:
            if not HOPECHESS_BIN.exists():
                logger.info("Building HopeChess engine via zig ... this may take a minute")
                try:
                    subprocess.run(['zig', 'build', '-Drelease-safe'], cwd=str(HOPECHESS_DIR), check=True)
                except subprocess.CalledProcessError:
                    # Older scripts may not have release-safe; fall back to release-small
                    subprocess.run(['zig', 'build', '-Drelease-small'], cwd=str(HOPECHESS_DIR), check=True)

            proxy_path = Path(__file__).resolve().parent / 'tools' / 'hopechess_proxy.py'
            self.hopechess_engine = chess.engine.SimpleEngine.popen_uci(
                [sys.executable, str(proxy_path), str(HOPECHESS_BIN), '--uci'],
                setpgrp=True,
                timeout=60.0,  # seconds
                stderr=subprocess.DEVNULL,
            )
            logger.info("HopeChess engine initialized from: %s", HOPECHESS_BIN)
        except Exception as e:
            logger.warning(f"Could not initialize HopeChess engine: {e}")
            self.hopechess_engine = None

    def get_ai_move(self, fen: str):
        """
        Determines the AI's next move using MCTS.
        """
        if not self.az_engine:
            return None, "AI model not loaded.", 0.0

        try:
            board = chess.Board(fen)
            policy, value = self.az_engine.search(board, add_noise=False)
            
            move = select_move(policy, temperature=0, board=board)
            
            if move:
                return move.uci(), None, float(value)
            else:
                return None, "No legal moves found by AI.", 0.0

        except Exception as e:
            logger.error(f"Error during AI move generation: {e}")
            import traceback
            traceback.print_exc()
            return None, str(e), 0.0

    def get_stockfish_move(self, fen: str, strength: str = "medium"):
        """Return a move from Stockfish and log detailed search metadata."""
        if not self.stockfish_engine:
            return None, "Stockfish engine not available."

        try:
            board = chess.Board(fen)
            if board.is_game_over():
                return None, "Game is already over."

            params = STOCKFISH_STRENGTHS.get(strength, STOCKFISH_STRENGTHS["medium"])

            # Configure skill level if available
            if "skill" in params:
                try:
                    self.stockfish_engine.configure({"Skill Level": params["skill"]})
                except Exception:
                    pass  # Some builds might not expose Skill Level

            # ------------------------------------------------------------------
            # Send play command with combined depth + time limit and time the call
            # ------------------------------------------------------------------
            start_ts = time.time()
            limit = chess.engine.Limit(depth=params["depth"], time=params["time"])
            result = self.stockfish_engine.play(board, limit, info=chess.engine.INFO_ALL)
            elapsed = time.time() - start_ts

            # Engine might report reached depth in result.info
            reached_depth = None
            if hasattr(result, 'info') and isinstance(result.info, dict):
                reached_depth = result.info.get('depth')
                nodes = result.info.get('nodes')
                nps = result.info.get('nps')
            else:
                nodes = nps = None

            logger.info(
                "Stockfish move | strength=%s | skill=%s | depth_limit=%s | time_limit=%.1fs | elapsed=%.3fs | reached_depth=%s | nodes=%s | nps=%s | move=%s",
                strength,
                params.get("skill"),
                params["depth"],
                params["time"],
                elapsed,
                reached_depth if reached_depth is not None else "?",
                nodes if nodes is not None else "?",
                nps if nps is not None else "?",
                result.move
            )

            if result.move:
                return result.move.uci(), None
            else:
                return None, "Stockfish did not return a move."
        except Exception as e:
            logger.error(f"Stockfish error: {e}")
            return None, str(e)
    
    def evaluate_position(self, fen: str, stockfish_depth: int = 10):
        """
        Provides both AI and Stockfish evaluations for a position.
        Handles cases where one or both models are not available.
        """
        ai_eval = None
        stockfish_eval = None
        error_msg = None

        # 1. Get AI Evaluation if model is loaded
        if self.model:
            try:
                board_tensor = torch.from_numpy(self.chess_env.board_to_tensor(chess.Board(fen))).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    value, _ = self.model(board_tensor)
                # The model's value is in [-1, 1]. We can scale it to a pawn equivalent for display if needed, but raw value is fine.
                ai_eval = value.item()
            except Exception as e:
                logger.warning(f"AI evaluation failed: {e}")
        
        # 2. Get Stockfish Evaluation if engine is available
        if self.stockfish_engine:
            try:
                board = chess.Board(fen)
                if not board.is_game_over():
                    # Use .analyse() for evaluation, not .play()
                    info = self.stockfish_engine.analyse(board, chess.engine.Limit(depth=stockfish_depth))
                    # Safely get score, check if 'score' key exists
                    if "score" in info and info["score"] is not None:
                        # Score is from white's perspective. Convert to centipawns.
                        score = info["score"].white().score(mate_score=10000)
                        stockfish_eval = score
            except Exception as e:
                logger.error(f"Stockfish evaluation error: {e}")
                error_msg = str(e) # This is a more critical error if it fails
        
        return ai_eval, stockfish_eval, error_msg

    def _get_position_status(self, board: chess.Board):
        """Get position status (checkmate, draw, etc.)"""
        if board.is_checkmate():
            return "Checkmate"
        if board.is_stalemate():
            return "Stalemate"
        if board.is_insufficient_material():
            return "Insufficient Material"
        if board.can_claim_draw():
            return "Can Claim Draw"
        if board.is_seventyfive_moves():
            return "75-move Rule"
        if board.is_fivefold_repetition():
            return "Fivefold Repetition"
        if board.is_check():
            return "Check"
        return "In Progress"

    # ---------------------------------------------------------------------
    # HopeChess Move
    # ---------------------------------------------------------------------
    def get_hopechess_move(self, fen: str, strength: str = 'medium'):
        if not self.hopechess_engine:
            return None, 'HopeChess engine not available.'

        try:
            board = chess.Board(fen)
            if board.is_game_over():
                return None, 'Game is already over.'

            params = HOPECHESS_STRENGTHS.get(strength, HOPECHESS_STRENGTHS['medium'])
            # Use a short absolute time limit (1s) to bound search duration regardless of position complexity.
            limit = chess.engine.Limit(time=1.0)

            start_ts = time.time()
            # Omit INFO_ALL – some UCI fields are not yet implemented by HopeChess and cause parse errors.
            try:
                with self.hopechess_lock:
                    result = self.hopechess_engine.play(board, limit)
            except (chess.engine.EngineTerminatedError, chess.engine.EngineError) as inner_err:
                logger.error("HopeChess crashed or became unresponsive: %s. Restarting engine and retrying once.", inner_err)
                self._setup_hopechess()
                if not self.hopechess_engine:
                    return None, 'HopeChess engine restart failed.'
                with self.hopechess_lock:
                    result = self.hopechess_engine.play(board, limit)
            elapsed = time.time() - start_ts

            nodes = nps = None  # HopeChess does not yet return extended search info
            if hasattr(result, 'info') and isinstance(result.info, dict):
                reached_depth = result.info.get('depth')

            logger.info(
                'HopeChess move | strength=%s | time_limit=1.0s | elapsed=%.3fs | move=%s',
                strength, elapsed,
                reached_depth if reached_depth else '?', result.move)

            if result.move:
                return result.move.uci(), None
            return None, 'HopeChess did not return a move.'
        except Exception as e:
            logger.error(f"HopeChess error: {e.__class__.__name__}: {e}")
            return None, str(e)

# --- Flask API Endpoints ---
@app.route('/')
def index():
    return render_template('chess.html')

@app.route('/api/move/ai', methods=['POST'])
def api_get_ai_move():
    data = request.get_json()
    fen = data.get('fen', chess.STARTING_FEN)
    
    move, error, evaluation = chess_ai.get_ai_move(fen)
    
    if error:
        return jsonify({'error': error}), 500
        
    return jsonify({
        'move': move,
        'evaluation': evaluation,
        'engine': 'Native Transformer'
    })

@app.route('/api/move/stockfish', methods=['POST'])
def api_get_stockfish_move():
    """Gets a move from the Stockfish engine."""
    data = request.json
    fen = data.get('fen')
    strength = data.get('strength', 'medium') # Default to medium if not provided
    if not fen:
        return jsonify({"error": "FEN string is required."}), 400
    
    move, error = chess_ai.get_stockfish_move(fen, strength)
    
    if error:
        return jsonify({"error": error}), 500
    return jsonify({"move": move})

@app.route('/api/evaluate', methods=['POST'])
def api_evaluate_position():
    """Provides a unified evaluation of the current board position."""
    data = request.json
    fen = data.get('fen')
    if not fen:
        return jsonify({"error": "FEN string is required."}), 400

    # We only need one comprehensive evaluation from Stockfish for the GUI.
    # The NN evaluation can be intensive and is better suited for training/analysis loops.
    _, stockfish_eval, error = chess_ai.evaluate_position(fen, stockfish_depth=12) # Use a decent depth

    if error:
        return jsonify({"error": f"Evaluation failed: {error}"}), 500

    # Return a single score from White's perspective.
    # The frontend will interpret this for both Black and White.
    return jsonify({"evaluation": stockfish_eval})

@app.route('/api/legal_moves', methods=['POST'])
def get_legal_moves():
    data = request.get_json()
    fen = data.get('fen', chess.STARTING_FEN)
    try:
        board = chess.Board(fen)
        moves = [move.uci() for move in board.legal_moves]
        return jsonify({'legal_moves': moves})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/status')
def get_status():
    return jsonify({
        'status': 'running',
        'model_loaded': chess_ai.model is not None,
        'stockfish_available': chess_ai.stockfish_engine is not None,
        'device': str(chess_ai.device)
    })

@app.route('/img/chesspieces/wikipedia/<piece>')
def serve_piece_image_fallback(piece):
    # This is a fallback for older board implementations if needed.
    return redirect(url_for('static', filename=f'img/chesspieces/wikipedia/{piece}'))

@app.route('/api/move/hopechess', methods=['POST'])
def api_get_hopechess_move():
    data = request.json
    fen = data.get('fen')
    strength = data.get('strength', 'medium')
    if not fen:
        return jsonify({'error': 'FEN string is required.'}), 400

    move, error = chess_ai.get_hopechess_move(fen, strength)
    if error:
        return jsonify({'error': error}), 500
    return jsonify({'move': move})

if __name__ == '__main__':
    # Initialize the server class
    chess_ai = ChessAIServer()
    
    print("--- Native Chess Transformer GUI ---")
    print("🌐 Open http://localhost:8080 in your browser")
    
    # Run Flask server
    app.run(host='0.0.0.0', port=8080, debug=False) 