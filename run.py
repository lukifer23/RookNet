#!/usr/bin/env python3
"""
Command-Line Interface for Native Chess Transformer
---------------------------------------------------
This script provides a CLI for managing the chess AI project, including
training, evaluation, data generation, and environment checks.
"""

import argparse
import sys
import os
import subprocess

# (Removed obsolete sys.path manipulation after package installation)

from utils.config_loader import load_config

def test_environment(config):
    """
    Tests the chess environment, including Stockfish integration.
    """
    print("Testing Chess Environment...")
    try:
        from utils.chess_env import ChessEnvironment
        
        print("✓ Chess environment imported successfully")
        
        env = ChessEnvironment(config)
        print("✓ Chess environment initialized")
        
        # Test Stockfish
        env.start_engine()
        print(f"✓ Stockfish engine started: {env.engine.id['name']}")
        
        # Test basic functionality
        eval_score = env.get_engine_evaluation()
        print(f"✓ Position evaluation: {eval_score} centipawns")
        
        best_move, info = env.get_engine_move(time_limit=0.5)
        if best_move:
            print(f"✓ Best move: {best_move}")
        
        env.stop_engine()
        print("✓ Environment test completed successfully!")
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        print(f"Error details: {traceback.format_exc()}")
        sys.exit(1)


def train_model(args):
    """Entry point for the 'train' subcommand."""
    from src.training.alphazero_trainer import AlphaZeroTrainer

    trainer = AlphaZeroTrainer(config_path=args.config)
    trainer.run()


def evaluate_models(args):
    """Run the evaluation script with provided arguments."""
    cmd = [
        sys.executable,
        "evaluate_model.py",
        *args.models,
        "--mode",
        args.mode,
        "--games",
        str(args.games),
        "--stockfish_depth",
        str(args.stockfish_depth),
        "--mcts_sims",
        str(args.mcts_sims),
        "--time_limit",
        str(args.time_limit),
        "--device",
        args.device,
    ]
    subprocess.run(cmd, check=True)


def launch_web_gui(args):
    """Launch the Flask-based chess web GUI."""
    cmd = [sys.executable, "chess_web_gui.py"]
    subprocess.run(cmd, check=True)

def main():
    """
    Main function to parse arguments and run commands.
    """
    parser = argparse.ArgumentParser(description="Native Chess Transformer CLI")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.v2.yaml",
        help="Path to the configuration file.",
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Test environment command
    test_parser = subparsers.add_parser(
        "test_env", help="Test the chess environment and dependencies."
    )
    test_parser.set_defaults(func=test_environment)

    # Training command
    train_parser = subparsers.add_parser(
        "train", help="Run AlphaZero training using alphazero_trainer."
    )
    train_parser.set_defaults(func=train_model)

    # Evaluation command
    eval_parser = subparsers.add_parser(
        "eval", help="Evaluate models using evaluate_model.py."
    )
    eval_parser.add_argument(
        "models",
        nargs="+",
        help="Paths to model checkpoint files (.pt).",
    )
    eval_parser.add_argument(
        "--mode",
        type=str,
        default="stockfish",
        choices=["stockfish", "model"],
        help="Evaluation mode: 'stockfish' or 'model' vs model.",
    )
    eval_parser.add_argument(
        "--games", type=int, default=10, help="Number of games to play."
    )
    eval_parser.add_argument(
        "--stockfish-depth",
        type=int,
        default=5,
        help="Stockfish search depth.",
    )
    eval_parser.add_argument(
        "--mcts-sims",
        type=int,
        default=50,
        help="MCTS simulations per move.",
    )
    eval_parser.add_argument(
        "--time-limit",
        type=float,
        default=1.0,
        help="Time per move for Stockfish.",
    )
    eval_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (e.g., 'cuda', 'cpu').",
    )
    eval_parser.set_defaults(func=evaluate_models)

    # Web GUI command
    web_parser = subparsers.add_parser(
        "web", help="Launch the chess web GUI."
    )
    web_parser.set_defaults(func=launch_web_gui)

    args = parser.parse_args()

    if args.command == "test_env":
        config = load_config(args.config)
        args.func(config)
    else:
        args.func(args)


if __name__ == "__main__":
    main()