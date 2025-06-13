#!/usr/bin/env python3
"""
Setup and Quick Start Script for Chess AI Project

This script helps set up the environment and provides quick commands
to get started with the chess neural network project.
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(command: str, description: str = "") -> bool:
    """Run a shell command and return success status."""
    if description:
        logger.info(f"{description}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.info(result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {command}")
        logger.error(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        logger.error("Python 3.9+ is required. Current version: {}.{}.{}".format(
            version.major, version.minor, version.micro))
        return False
    
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro} ✓")
    return True


def setup_virtual_environment():
    """Create and activate virtual environment."""
    venv_path = Path("chess_ai_env")
    
    if venv_path.exists():
        logger.info("Virtual environment already exists ✓")
        return True
    
    logger.info("Creating virtual environment...")
    if not run_command("python3 -m venv chess_ai_env", "Creating virtual environment"):
        return False
    
    logger.info("Virtual environment created ✓")
    logger.info("To activate: source chess_ai_env/bin/activate")
    return True


def install_dependencies():
    """Install Python dependencies."""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        logger.error("requirements.txt not found")
        return False
    
    # Check if we're in a virtual environment
    if not os.environ.get('VIRTUAL_ENV'):
        logger.warning("Not in a virtual environment. Activating chess_ai_env...")
        activate_cmd = "source chess_ai_env/bin/activate && "
    else:
        activate_cmd = ""
    
    logger.info("Installing Python dependencies...")
    cmd = f"{activate_cmd}pip install -r requirements.txt"
    return run_command(cmd, "Installing dependencies")


def check_stockfish():
    """Check if Stockfish is installed."""
    try:
        result = subprocess.run(["which", "stockfish"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Stockfish found: {result.stdout.strip()} ✓")
            return True
        else:
            logger.warning("Stockfish not found in PATH")
            return False
    except Exception as e:
        logger.error(f"Error checking Stockfish: {e}")
        return False


def install_stockfish():
    """Install Stockfish using Homebrew."""
    logger.info("Installing Stockfish...")
    
    # Check if Homebrew is installed
    try:
        subprocess.run(["brew", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Homebrew not found. Please install Homebrew first:")
        logger.error("Visit: https://brew.sh/")
        return False
    
    return run_command("brew install stockfish", "Installing Stockfish via Homebrew")


def test_chess_environment():
    """Test the chess environment setup."""
    logger.info("Testing chess environment...")
    
    try:
        # Import and test chess environment
        from src.utils.chess_env import ChessEnvironment
        
        env = ChessEnvironment()
        
        # Test basic functionality
        tensor = env.board_to_tensor()
        legal_moves = env.get_legal_moves()
        
        logger.info(f"Board tensor shape: {tensor.shape} ✓")
        logger.info(f"Legal moves: {len(legal_moves)} ✓")
        
        # Test engine if available
        try:
            env.start_engine()
            best_move, _ = env.get_engine_move(time_limit=0.1)
            eval_score = env.get_engine_evaluation(time_limit=0.1)
            env.stop_engine()
            
            logger.info(f"Engine test: Move={best_move}, Eval={eval_score} ✓")
            
        except Exception as e:
            logger.warning(f"Engine test failed: {e}")
            logger.warning("Install Stockfish to enable engine features")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"Chess environment test failed: {e}")
        return False


def generate_sample_data():
    """Generate a small sample of training data."""
    logger.info("Generating sample training data...")
    
    try:
        from src.data.stockfish_generator import TrainingDataGenerator
        
        config = {
            "data_dir": "data",
            "stockfish": {
                "depth": 8,
                "time_limit": 0.5,
                "threads": 2
            }
        }
        
        generator = TrainingDataGenerator(config)
        
        # Generate a few quick games for testing
        data = generator.generate_game_data(
            num_games=5,
            white_depth=6,
            black_depth=6,
            time_per_move=0.5,
            save_pgn=True
        )
        
        logger.info(f"Generated {len(data)} training positions ✓")
        return True
        
    except Exception as e:
        logger.error(f"Data generation failed: {e}")
        return False


def test_model():
    """Test model creation and basic functionality."""
    logger.info("Testing model creation...")
    
    try:
        from src.models.baseline import create_model
        import torch
        
        # Test different model types
        models_to_test = [
            ("mlp", {"input_size": 768, "hidden_size": 512, "output_size": 4096}),
            ("cnn", {"input_channels": 12, "hidden_channels": 64}),
        ]
        
        for model_type, config in models_to_test:
            model = create_model(model_type, config)
            
            # Test forward pass
            if model_type == "mlp":
                test_input = torch.randn(2, 12, 8, 8)
            else:
                test_input = torch.randn(2, 12, 8, 8)
            
            output = model(test_input)
            logger.info(f"{model_type.upper()} model test passed ✓")
        
        return True
        
    except Exception as e:
        logger.error(f"Model test failed: {e}")
        return False


def create_run_scripts():
    """Create convenience scripts for common tasks."""
    
    scripts = {
        "generate_data.py": """#!/usr/bin/env python3
import sys
sys.path.append('.')
from src.data.stockfish_generator import main
if __name__ == "__main__":
    main()
""",
        
        "train_model.py": """#!/usr/bin/env python3
import sys
sys.path.append('.')
from src.training.trainer import main
if __name__ == "__main__":
    main()
""",
        
        "test_environment.py": """#!/usr/bin/env python3
import sys
sys.path.append('.')
from src.utils.chess_env import ChessEnvironment

def main():
    print("Testing Chess Environment...")
    env = ChessEnvironment()
    
    # Basic tests
    tensor = env.board_to_tensor()
    print(f"Board tensor shape: {tensor.shape}")
    
    moves = env.get_legal_moves()
    print(f"Legal moves: {len(moves)}")
    
    # Engine test
    try:
        env.start_engine()
        move, info = env.get_engine_move(time_limit=0.1)
        print(f"Best move: {move}")
        env.stop_engine()
        print("Engine test passed!")
    except Exception as e:
        print(f"Engine test failed: {e}")

if __name__ == "__main__":
    main()
"""
    }
    
    for script_name, content in scripts.items():
        script_path = Path(script_name)
        with open(script_path, 'w') as f:
            f.write(content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        logger.info(f"Created script: {script_name} ✓")


def main():
    """Main setup function."""
    logger.info("🚀 Chess AI Project Setup")
    logger.info("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        return
    
    # Setup virtual environment
    if not setup_virtual_environment():
        logger.error("Failed to setup virtual environment")
        return
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Failed to install dependencies")
        return
    
    # Check/install Stockfish
    if not check_stockfish():
        logger.info("Stockfish not found. Attempting to install...")
        if not install_stockfish():
            logger.warning("Stockfish installation failed. Some features may not work.")
        else:
            check_stockfish()
    
    # Test components
    logger.info("\n📋 Running Component Tests")
    logger.info("-" * 30)
    
    if test_chess_environment():
        logger.info("Chess environment test passed ✓")
    
    if test_model():
        logger.info("Model creation test passed ✓")
    
    # Create utility scripts
    logger.info("\n🔧 Creating Utility Scripts")
    logger.info("-" * 30)
    create_run_scripts()
    
    # Final instructions
    logger.info("\n🎯 Setup Complete!")
    logger.info("=" * 50)
    logger.info("Next steps:")
    logger.info("1. Activate virtual environment: source chess_ai_env/bin/activate")
    logger.info("2. Generate training data: python generate_data.py")
    logger.info("3. Train a model: python train_model.py")
    logger.info("4. Test environment: python test_environment.py")
    logger.info("\nProject structure:")
    logger.info("- src/: Source code")
    logger.info("- configs/: Configuration files")
    logger.info("- data/: Training data")
    logger.info("- models/: Model checkpoints")
    logger.info("- logs/: Training logs")
    logger.info("\nReady to build the future of chess AI! 🚀")


if __name__ == "__main__":
    main()
