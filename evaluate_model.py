#!/usr/bin/env python3
"""
Comprehensive Model Evaluation System
Replaces all scattered evaluation scripts with a single unified tool.
"""

import argparse
import logging
import torch
import chess
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import math

from src.models.chess_transformer import ChessTransformer
from src.utils.chess_env import ChessEnvironment
from src.search.mcts import MCTS
from src.utils.config_loader import load_config
from src.utils.move_encoder import get_policy_vector_size
from stockfish import Stockfish
from src.evaluation.stats import sprt_llr, sprt_decide

# Load unified configuration
CONFIG = load_config("configs/config.v2.yaml")

class ModelEvaluator:
    def __init__(self, 
                 model_paths: List[str],
                 device: str = 'auto'):
        
        # System and evaluation configs
        sys_config = CONFIG['system']
        eval_config = CONFIG['evaluation']
        
        self.device = torch.device(sys_config['device'] if device == 'auto' else device)
        self.log_dir = Path(CONFIG['logging']['log_dir']) / 'evaluation_results'
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Load models using the new self-describing checkpoint format
        self.models = {}
        for path in model_paths:
            name = Path(path).stem
            try:
                checkpoint = torch.load(path, map_location=self.device)
                model_config = checkpoint['model_config']
                
                # Add policy vector size to config for model creation
                model_config['policy_head_output_size'] = get_policy_vector_size()

                model = ChessTransformer(**model_config)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                model.eval()
                self.models[name] = model
                logger.info(f"Loaded model '{name}' from {path}")
            except Exception as e:
                logger.error(f"Failed to load model from {path}: {e}")
                
        # Initialize Stockfish from config
        try:
            stockfish_path = eval_config['stockfish']['path']
            self.stockfish = Stockfish(path=stockfish_path)
            self.stockfish_available = True
            logger.info(f"Stockfish engine loaded from {stockfish_path}")
        except Exception as e:
            logger.warning(f"Stockfish not available at path '{stockfish_path}': {e}")
            self.stockfish_available = False
    
    def evaluate_vs_stockfish(self, 
                            model_name: str,
                            num_games: int = 10,
                            stockfish_depth: int = 1,
                            mcts_sims: int = 20,
                            time_per_move: float = 1.0) -> Dict:
        """Evaluate model against Stockfish"""
        
        if not self.stockfish_available:
            return {"error": "Stockfish not available"}
            
        model = self.models[model_name]
        env = ChessEnvironment()
        mcts = MCTS(model, device=self.device)
        
        results = {"wins": 0, "losses": 0, "draws": 0, "games": []}
        
        for game_num in range(num_games):
            logging.info(f"Game {game_num+1}/{num_games}")
            
            env.reset()
            game_moves = []
            model_plays_white = game_num % 2 == 0
            
            while not env.is_terminal():
                if (env.board.turn == chess.WHITE) == model_plays_white:
                    # Model's turn
                    start_time = time.time()
                    move = mcts.search(env.board, mcts_sims)
                    think_time = time.time() - start_time
                    
                    if move is None:
                        break  # No legal moves
                        
                    env.make_move(move)
                    game_moves.append({"move": str(move), "player": "model", "time": think_time})
                    
                else:
                    # Stockfish's turn
                    self.stockfish.set_fen_position(env.board.fen())
                    self.stockfish.set_depth(stockfish_depth)
                    
                    move_str = self.stockfish.get_best_move_time(int(time_per_move * 1000))
                    if move_str is None:
                        break
                        
                    move = chess.Move.from_uci(move_str)
                    env.make_move(move)
                    game_moves.append({"move": str(move), "player": "stockfish", "time": time_per_move})
            
            # Determine result
            result = env.get_result()
            if result == 0.5:  # Draw
                results["draws"] += 1
                outcome = "draw"
            elif (result == 1.0 and model_plays_white) or (result == 0.0 and not model_plays_white):
                results["wins"] += 1
                outcome = "win"
            else:
                results["losses"] += 1
                outcome = "loss"
                
            results["games"].append({
                "game_num": game_num + 1,
                "model_color": "white" if model_plays_white else "black",
                "outcome": outcome,
                "moves": game_moves,
                "final_fen": env.board.fen()
            })
        
        # Calculate statistics
        total_games = results["wins"] + results["losses"] + results["draws"]
        results["win_rate"] = results["wins"] / total_games if total_games > 0 else 0
        results["loss_rate"] = results["losses"] / total_games if total_games > 0 else 0
        results["draw_rate"] = results["draws"] / total_games if total_games > 0 else 0
        
        # --- Extra statistics -------------------------------------------------
        score = (results["wins"] + 0.5 * results["draws"]) / total_games if total_games else 0.0

        # Elo diff estimate via logistic model (Glicko-style but aggregated)
        if 0 < score < 1:
            results["elo_diff"] = 400 * math.log10(score / (1 - score))
        else:
            results["elo_diff"] = None

        # SPRT log-likelihood ratio and decision against H0: p=0.5, H1: p=0.55
        llr = sprt_llr(results["wins"], results["draws"], results["losses"], epsilon=0.05)
        results["sprt_llr"] = llr
        results["sprt_decision"] = sprt_decide(llr)
        
        return results
    
    def evaluate_model_vs_model(self, 
                               model1_name: str,
                               model2_name: str,
                               num_games: int = 10,
                               mcts_sims: int = 20) -> Dict:
        """Evaluate two models against each other"""
        
        model1 = self.models[model1_name]
        model2 = self.models[model2_name]
        env = ChessEnvironment()
        mcts1 = MCTS(model1, device=self.device)
        mcts2 = MCTS(model2, device=self.device)
        
        results = {f"{model1_name}_wins": 0, f"{model2_name}_wins": 0, "draws": 0, "games": []}
        
        for game_num in range(num_games):
            logging.info(f"Game {game_num+1}/{num_games}: {model1_name} vs {model2_name}")
            
            env.reset()
            game_moves = []
            model1_plays_white = game_num % 2 == 0
            
            while not env.is_terminal():
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
                    
                env.make_move(move)
                game_moves.append({"move": str(move), "player": player})
            
            # Determine result
            result = env.get_result()
            if result == 0.5:  # Draw
                results["draws"] += 1
                outcome = "draw"
            elif (result == 1.0 and model1_plays_white) or (result == 0.0 and not model1_plays_white):
                results[f"{model1_name}_wins"] += 1
                outcome = f"{model1_name}_win"
            else:
                results[f"{model2_name}_wins"] += 1
                outcome = f"{model2_name}_win"
                
            results["games"].append({
                "game_num": game_num + 1,
                f"{model1_name}_color": "white" if model1_plays_white else "black",
                f"{model2_name}_color": "black" if model1_plays_white else "white",
                "outcome": outcome,
                "moves": game_moves,
                "final_fen": env.board.fen()
            })
        
        # After aggregation compute statistics identical to Stockfish eval for consistency
        total_games = results[f"{model1_name}_wins"] + results[f"{model2_name}_wins"] + results["draws"]
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
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logging.info(f"Results saved to {filepath}")
        return filepath

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Chess Model Evaluation")
    parser.add_argument("--models", nargs="+", help="Model checkpoint paths. Overrides config default.")
    parser.add_argument("--vs-stockfish", action="store_true", help="Evaluate against Stockfish")
    parser.add_argument("--vs-models", action="store_true", help="Evaluate models against each other")
    parser.add_argument("--games", type=int, help="Number of games per evaluation. Overrides config.")
    parser.add_argument("--stockfish-depth", type=int, help="Stockfish search depth. Overrides config.")
    parser.add_argument("--mcts-sims", type=int, help="MCTS simulations per move. Overrides config.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device to use")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()

    # Get configs
    eval_config = CONFIG['evaluation']
    mcts_config = CONFIG['training']['mcts']

    # Determine parameters, preferring command line args over config file
    model_paths = args.models or [eval_config['default_model_to_evaluate']]
    num_games = args.games or eval_config.get('num_games', 10)
    stockfish_depth = args.stockfish_depth or eval_config['stockfish']['default_depth']
    mcts_sims = args.mcts_sims or mcts_config['simulations']

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_paths=model_paths,
        device=args.device
    )
    
    if not evaluator.models:
        logging.error("No models were loaded successfully. Aborting evaluation.")
        return

    if args.vs_stockfish:
        for model_path in model_paths:
            model_name = Path(model_path).stem
            if model_name not in evaluator.models:
                continue
            logging.info(f"Evaluating {model_name} vs Stockfish (Depth: {stockfish_depth})")
            
            results = evaluator.evaluate_vs_stockfish(
                model_name=model_name,
                num_games=num_games,
                stockfish_depth=stockfish_depth,
                mcts_sims=mcts_sims
            )
            
            evaluator.save_results(results, f"{model_name}_vs_stockfish")
            
            logging.info(f"Results: {results['wins']}W-{results['losses']}L-{results['draws']}D "
                        f"(Win Rate: {results['win_rate']:.1%})")
    
    if args.vs_models and len(model_paths) >= 2:
        # Evaluate all pairs of models
        for i, model1_path in enumerate(model_paths):
            for model2_path in model_paths[i+1:]:
                model1_name = Path(model1_path).stem
                model2_name = Path(model2_path).stem
                
                logging.info(f"Evaluating {model1_name} vs {model2_name}")
                
                results = evaluator.evaluate_model_vs_model(
                    model1_name=model1_name,
                    model2_name=model2_name,
                    num_games=num_games,
                    mcts_sims=mcts_sims
                )
                
                evaluator.save_results(results, f"{model1_name}_vs_{model2_name}")
                
                logging.info(f"Results: {model1_name} {results[f'{model1_name}_wins']}W vs "
                           f"{model2_name} {results[f'{model2_name}_wins']}W, {results['draws']}D")

if __name__ == "__main__":
    main() 