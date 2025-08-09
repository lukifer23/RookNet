#!/usr/bin/env python3
"""
Command-Line Interface for Native Chess Transformer
---------------------------------------------------
This script provides a CLI for managing the chess AI project, including
training, evaluation, data generation, and environment checks.
"""

import argparse
import sys

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

    args = parser.parse_args()

    config = load_config(args.config)

    args.func(config)


if __name__ == "__main__":
    main()
