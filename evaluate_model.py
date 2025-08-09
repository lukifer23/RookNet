#!/usr/bin/env python3
"""
Comprehensive Model Evaluation System
Replaces all scattered evaluation scripts with a single unified tool.
"""

import argparse
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import chess
import torch

from evaluation.stats import sprt_decide, sprt_llr
from models.chess_transformer import ChessTransformer
from search.mcts import MCTS
from utils.chess_env import ChessEnvironment
from utils.config_loader import load_config
from utils.move_encoder import get_policy_vector_size

# Load unified configuration
CONFIG = load_config("configs/config.v2.yaml")


class ModelEvaluator:
    def __init__(self, model_paths: List[str], device: str = "auto"):

        # System and evaluation configs
        sys_config = CONFIG["system"]

        self.device = torch.device(sys_config["device"] if device == "auto" else device)
        self.log_dir = Path(CONFIG["logging"]["log_dir"]) / "evaluation_results"
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # Load models using the new self-describing checkpoint format
        self.models = {}
        for path in model_paths:
            name = Path(path).stem
            try:
                checkpoint = torch.load(path, map_location=self.device)
                model_config = checkpoint["model_config"]

                # Add policy vector size to config for model creation
                model_config["policy_head_output_size"] = get_policy_vector_size()

                model = ChessTransformer(**model_config)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(self.device)
                model.eval()
                self.models[name] = model
                logger.info(f"Loaded model '{name}' from {path}")
            except Exception as e:
                logger.error(f"Failed to load model from {path}: {e}")

        # Initialize chess environment for engine management
        self.env = ChessEnvironment(CONFIG)
        self.stockfish_available = False
        try:
            self.env.start_engine()
            self.stockfish_available = True
            logger.info(
                f"Stockfish engine loaded via python-chess: {self.env.engine.id['name']}"
            )
        except Exception as e:
            logger.warning(f"Could not start chess engine: {e}")

    def __del__(self):
        """Ensure engine is cleanly shut down."""
        if self.stockfish_available:
            self.env.stop_engine()

    def evaluate_vs_stockfish(
        self,
        model_name: str,
        num_games: int = 10,
        stockfish_depth: int = 1,
        mcts_sims: int = 20,
        time_per_move: float = 1.0,
    ) -> Dict:
        """Evaluate model against Stockfish"""

        if not self.stockfish_available:
            return {"error": "Stockfish not available"}

        model = self.models[model_name]
        mcts = MCTS(model, device=self.device)

        results = {"wins": 0, "losses": 0, "draws": 0, "games": []}

        for game_num in range(num_games):
            logger.info(
                f"Starting Stockfish game {game_num+1}/{num_games} for model '{model_name}'"
            )

            self.env.board.reset()
            game_moves = []
            model_plays_white = game_num % 2 == 0

            while not self.env.board.is_game_over():
                if (self.env.board.turn == chess.WHITE) == model_plays_white:
                    # Model's turn
                    start_time = time.time()
                    move = mcts.search(self.env.board, mcts_sims)
                    think_time = time.time() - start_time

                    if move is None:
                        break

                    self.env.board.push(move)
                    game_moves.append(
                        {"move": str(move), "player": "model", "time": think_time}
                    )

                else:
                    # Stockfish's turn using python-chess
                    move, _ = self.env.get_engine_move(time_limit=time_per_move)
                    if move is None:
                        break

                    self.env.board.push(move)
                    game_moves.append(
                        {
                            "move": str(move),
                            "player": "stockfish",
                            "time": time_per_move,
                        }
                    )

            # Determine result
            result_str = self.env.board.result()
            if result_str == "1/2-1/2":
                results["draws"] += 1
                outcome = "draw"
            elif (result_str == "1-0" and model_plays_white) or (
                result_str == "0-1" and not model_plays_white
            ):
                results["wins"] += 1
                outcome = "win"
            else:
                results["losses"] += 1
                outcome = "loss"

            results["games"].append(
                {
                    "game_num": game_num + 1,
                    "model_color": "white" if model_plays_white else "black",
                    "outcome": outcome,
                    "moves": game_moves,
                    "final_fen": self.env.board.fen(),
                }
            )

        # Calculate statistics
        total_games = results["wins"] + results["losses"] + results["draws"]
        results["win_rate"] = results["wins"] / total_games if total_games > 0 else 0
        results["loss_rate"] = results["losses"] / total_games if total_games > 0 else 0
        results["draw_rate"] = results["draws"] / total_games if total_games > 0 else 0

        # --- Extra statistics -------------------------------------------------
        score = (
            (results["wins"] + 0.5 * results["draws"]) / total_games
            if total_games
            else 0.0
        )

        # Elo diff estimate via logistic model (Glicko-style but aggregated)
        if 0 < score < 1:
            results["elo_diff"] = 400 * math.log10(score / (1 - score))
        else:
            results["elo_diff"] = None

        # SPRT log-likelihood ratio and decision against H0: p=0.5, H1: p=0.55
        llr = sprt_llr(
            results["wins"], results["draws"], results["losses"], epsilon=0.05
        )
        results["sprt_llr"] = llr
        results["sprt_decision"] = sprt_decide(llr)

        return results

    def evaluate_model_vs_model(
        self,
        model1_name: str,
        model2_name: str,
        num_games: int = 10,
        mcts_sims: int = 20,
    ) -> Dict:
        """Evaluate two models against each other"""

        model1 = self.models[model1_name]
        model2 = self.models[model2_name]
        # Use a local environment for model vs model games
        env = ChessEnvironment(CONFIG)
        mcts1 = MCTS(model1, device=self.device)
        mcts2 = MCTS(model2, device=self.device)

        results = {
            f"{model1_name}_wins": 0,
            f"{model2_name}_wins": 0,
            "draws": 0,
            "games": [],
        }

        for game_num in range(num_games):
            logger.info(
                f"Game {game_num+1}/{num_games}: {model1_name} vs {model2_name}"
            )

            env.board.reset()
            game_moves = []
            model1_plays_white = game_num % 2 == 0

            while not env.board.is_game_over():
                if (env.board.turn == chess.WHITE) == model1_plays_white:
                    # Model 1's turn
                    move = mcts1.search(env.board, mcts_sims)
                    player = model1_name
                else:
                    # Model 2's turn
                    move = mcts2.search(env.board, mcts_sims)
                    player = model2_name

                if move is None:
                    break

                env.board.push(move)
                game_moves.append({"move": str(move), "player": player})

            # Determine result
            result_str = env.board.result()
            if result_str == "1/2-1/2":
                results["draws"] += 1
                outcome = "draw"
            elif (result_str == "1-0" and model1_plays_white) or (
                result_str == "0-1" and not model1_plays_white
            ):
                results[f"{model1_name}_wins"] += 1
                outcome = f"{model1_name}_win"
            else:
                results[f"{model2_name}_wins"] += 1
                outcome = f"{model2_name}_win"

            results["games"].append(
                {
                    "game_num": game_num + 1,
                    f"{model1_name}_color": "white" if model1_plays_white else "black",
                    f"{model2_name}_color": "black" if model1_plays_white else "white",
                    "outcome": outcome,
                    "moves": game_moves,
                    "final_fen": env.board.fen(),
                }
            )

        # After aggregation compute statistics identical to Stockfish eval for consistency
        total_games = (
            results[f"{model1_name}_wins"]
            + results[f"{model2_name}_wins"]
            + results["draws"]
        )
        wins = results[f"{model1_name}_wins"]
        losses = results[f"{model2_name}_wins"]
        draws = results["draws"]

        score = (wins + 0.5 * draws) / total_games if total_games else 0.0
        results["win_rate"] = score

        if 0 < score < 1:
            results["elo_diff"] = 400 * math.log10(score / (1 - score))
        else:
            results["elo_diff"] = None

        llr = sprt_llr(wins, draws, losses, epsilon=0.05)
        results["sprt_llr"] = llr
        results["sprt_decision"] = sprt_decide(llr)

        return results

    def save_results(self, results: Dict, filename: str):
        """Save evaluation results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.log_dir / f"{filename}_{timestamp}.json"

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logging.info(f"Results saved to {filepath}")
        return filepath


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Unified Model Evaluation System")
    parser.add_argument(
        "models", nargs="+", help="Paths to model checkpoint files (.pt)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="stockfish",
        choices=["stockfish", "model"],
        help="Evaluation mode: 'stockfish' or 'model' vs model.",
    )
    parser.add_argument(
        "--games", type=int, default=10, help="Number of games to play."
    )
    parser.add_argument(
        "--stockfish_depth", type=int, default=5, help="Stockfish search depth."
    )
    parser.add_argument(
        "--mcts_sims", type=int, default=50, help="MCTS simulations per move."
    )
    parser.add_argument(
        "--time_limit", type=float, default=1.0, help="Time per move for Stockfish."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (e.g., 'cuda', 'cpu').",
    )

    args = parser.parse_args()

    # Setup logging
    log_file = Path(CONFIG["logging"]["log_dir"]) / "evaluation.log"
    log_file.parent.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    global logger
    logger = logging.getLogger(__name__)

    evaluator = ModelEvaluator(model_paths=args.models, device=args.device)

    if args.mode == "stockfish":
        if len(args.models) != 1:
            logger.error("Stockfish evaluation requires exactly one model.")
            return

        logger.info(f"Starting Stockfish evaluation for {args.models[0]}")
        results = evaluator.evaluate_vs_stockfish(
            model_name=Path(args.models[0]).stem,
            num_games=args.games,
            stockfish_depth=args.stockfish_depth,
            mcts_sims=args.mcts_sims,
            time_per_move=args.time_limit,
        )
        evaluator.save_results(results, f"stockfish_eval_{Path(args.models[0]).stem}")

    elif args.mode == "model":
        if len(args.models) != 2:
            logger.error("Model vs. model evaluation requires exactly two models.")
            return

        logger.info(
            f"Starting model vs. model evaluation: {args.models[0]} vs {args.models[1]}"
        )
        results = evaluator.evaluate_model_vs_model(
            model1_name=Path(args.models[0]).stem,
            model2_name=Path(args.models[1]).stem,
            num_games=args.games,
            mcts_sims=args.mcts_sims,
        )
        model1_stem = Path(args.models[0]).stem
        model2_stem = Path(args.models[1]).stem
        evaluator.save_results(results, f"model_eval_{model1_stem}_vs_{model2_stem}")


if __name__ == "__main__":
    main()
