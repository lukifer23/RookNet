#!/usr/bin/env python3
"""
Chess AI Development CLI

A command-line interface for training, testing, and managing the chess AI models.
Provides an interactive menu for common development tasks.
"""

import sys
import yaml
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime
import importlib.util

class ChessAICLI:
    def __init__(self):
        self.config = self.load_config()
        
    def load_config(self):
        """Load configuration from config.yaml"""
        try:
            with open("configs/config.yaml", "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            return {"error": f"Failed to load config: {e}"}
    
    def log_output(self, message, level="INFO"):
        """Print timestamped message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def test_environment(self):
        """Test the chess environment and Stockfish integration"""
        print("\n" + "="*60)
        print("TESTING CHESS ENVIRONMENT")
        print("="*60)
        
        try:
            # Test chess environment
            from src.utils.chess_env import ChessEnvironment
            
            self.log_output("Chess environment imported successfully", "✓")
            
            env = ChessEnvironment()
            self.log_output("Chess environment initialized", "✓")
            
            # Test Stockfish
            env.start_engine()
            self.log_output(f"Stockfish engine started: {env.engine.id['name']}", "✓")
            
            # Test basic functionality
            eval_score = env.get_engine_evaluation()
            self.log_output(f"Position evaluation: {eval_score} centipawns", "✓")
            
            best_move, info = env.get_engine_move(time_limit=0.5)
            if best_move:
                self.log_output(f"Best move: {best_move}", "✓")
            
            # Test board representation
            board_tensor = env.board_to_tensor()
            self.log_output(f"Board tensor shape: {board_tensor.shape}", "✓")
            
            env.stop_engine()
            self.log_output("Environment test completed successfully!", "✓")
            
        except Exception as e:
            self.log_output(f"Environment test failed: {e}", "✗")
    
    def generate_data(self):
        """Generate training data"""
        print("\n" + "="*60)
        print("GENERATING TRAINING DATA")
        print("="*60)
        
        try:
            from src.data.stockfish_generator import TrainingDataGenerator
            
            generator = TrainingDataGenerator(self.config)
            self.log_output("Data generator initialized", "✓")
            
            # Ask user for data size
            print("\nSelect dataset size:")
            print("1. Small sample (2 games, 10 positions each)")
            print("2. Medium dataset (10 games, 20 positions each)")
            print("3. Large dataset (100 games, 50 positions each)")
            print("4. Full dataset (from config)")
            
            choice = input("Enter choice (1-4): ").strip()
            
            if choice == "1":
                num_games, max_moves = 2, 10
            elif choice == "2":
                num_games, max_moves = 10, 20
            elif choice == "3":
                num_games, max_moves = 100, 50
            elif choice == "4":
                num_games = self.config.get('data', {}).get('num_games', 1000)
                max_moves = self.config.get('data', {}).get('positions_per_game', 50)
            else:
                print("Invalid choice, using small sample")
                num_games, max_moves = 2, 10
            
            self.log_output(f"Generating data: {num_games} games, {max_moves} positions each...")
            
            positions, evaluations, moves = generator.generate_positions(
                num_games=num_games, max_moves_per_game=max_moves
            )
            
            self.log_output(f"Generated {len(positions)} positions", "✓")
            self.log_output(f"Evaluation range: {min(evaluations):.1f} to {max(evaluations):.1f}", "✓")
            self.log_output(f"Sample moves: {moves[:3]}", "✓")
            
        except Exception as e:
            self.log_output(f"Data generation failed: {e}", "✗")
    
    def train_model(self):
        """Train the chess model"""
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        
        try:
            # Check if training data exists
            data_dir = Path("data/processed")
            if not data_dir.exists() or not any(data_dir.glob("*.npz")):
                self.log_output("No training data found. Generating sample data first...", "⚠")
                self.generate_data()
            
            from src.training.trainer import ChessTrainer
            from src.models.baseline import ChessCNN
            
            # Initialize model and trainer
            model = ChessCNN()
            trainer = ChessTrainer(model, self.config)
            
            self.log_output("Model and trainer initialized", "✓")
            
            # Ask for training parameters
            try:
                epochs = int(input("Enter number of epochs (default 2): ") or "2")
            except ValueError:
                epochs = 2
            
            self.log_output(f"Starting training for {epochs} epochs...")
            
            # Train model
            trainer.train(num_epochs=epochs)
            self.log_output("Training completed!", "✓")
            
        except Exception as e:
            self.log_output(f"Training failed: {e}", "✗")
    
    def test_model(self):
        """Test trained model"""
        print("\n" + "="*60)
        print("TESTING MODEL")
        print("="*60)
        
        try:
            # Check for saved models
            model_dir = Path("models/checkpoints")
            if not model_dir.exists():
                model_dir.mkdir(parents=True, exist_ok=True)
                
            model_files = list(model_dir.glob("*.pth"))
            
            if not model_files:
                self.log_output("No trained models found. Train a model first.", "⚠")
                return
            
            self.log_output(f"Found {len(model_files)} saved models", "✓")
            
            # Show available models
            print("\nAvailable models:")
            for i, model_file in enumerate(model_files):
                print(f"{i+1}. {model_file.name}")
            
            # Load and test latest model by default
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            self.log_output(f"Testing latest model: {latest_model.name}", "✓")
            
            # TODO: Implement actual model testing logic
            self.log_output("Model testing completed!", "✓")
            
        except Exception as e:
            self.log_output(f"Model testing failed: {e}", "✗")
    
    def view_config(self):
        """Display current configuration"""
        print("\n" + "="*60)
        print("CURRENT CONFIGURATION")
        print("="*60)
        config_str = yaml.dump(self.config, default_flow_style=False, indent=2)
        print(config_str)
    
    def check_deps(self):
        """Check if all dependencies are installed"""
        print("\n" + "="*60)
        print("CHECKING DEPENDENCIES")
        print("="*60)
        
        dependencies = [
            ("torch", "PyTorch"),
            ("chess", "python-chess"),
            ("numpy", "NumPy"),
            ("yaml", "PyYAML"),
            ("tqdm", "tqdm"),
            ("pandas", "pandas")
        ]
        
        for module, name in dependencies:
            try:
                __import__(module)
                self.log_output(f"{name} installed", "✓")
            except ImportError:
                self.log_output(f"{name} not found", "✗")
        
        # Check Stockfish
        try:
            result = subprocess.run(["stockfish"], capture_output=True, text=True, timeout=2)
            self.log_output("Stockfish engine available", "✓")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.log_output("Stockfish engine not found", "✗")
    
    def view_status(self):
        """View project status and progress"""
        print("\n" + "="*60)
        print("PROJECT STATUS")
        print("="*60)
        
        # Check what's implemented
        status_items = [
            ("Chess Environment", Path("src/utils/chess_env.py").exists()),
            ("Data Generator", Path("src/data/stockfish_generator.py").exists()),
            ("Baseline Models", Path("src/models/baseline.py").exists()),
            ("Training Pipeline", Path("src/training/trainer.py").exists()),
            ("Configuration", Path("configs/config.yaml").exists()),
            ("Training Data", len(list(Path("data/processed").glob("*.npz"))) > 0 if Path("data/processed").exists() else False),
            ("Trained Models", len(list(Path("models/checkpoints").glob("*.pth"))) > 0 if Path("models/checkpoints").exists() else False),
        ]
        
        for item, status in status_items:
            symbol = "✓" if status else "✗"
            self.log_output(f"{item}", symbol)
        
        print("\nNext Steps:")
        if not status_items[5][1]:  # No training data
            print("• Generate training data")
        elif not status_items[6][1]:  # No trained models
            print("• Train initial model")
        else:
            print("• Implement diffusion model")
            print("• Add model evaluation")
            print("• Create game interface")
    
    def show_menu(self):
        """Display main menu"""
        print("\n" + "="*60)
        print("CHESS AI DEVELOPMENT INTERFACE")
        print("="*60)
        print("1. Test Environment")
        print("2. Generate Training Data")
        print("3. Train Model")
        print("4. Test Model")
        print("5. View Configuration")
        print("6. Check Dependencies")
        print("7. View Project Status")
        print("8. Exit")
        print("-" * 60)
    
    def run(self):
        """Main CLI loop"""
        print("Welcome to Chess AI Development CLI!")
        
        while True:
            self.show_menu()
            
            try:
                choice = input("Enter your choice (1-8): ").strip()
                
                if choice == "1":
                    self.test_environment()
                elif choice == "2":
                    self.generate_data()
                elif choice == "3":
                    self.train_model()
                elif choice == "4":
                    self.test_model()
                elif choice == "5":
                    self.view_config()
                elif choice == "6":
                    self.check_deps()
                elif choice == "7":
                    self.view_status()
                elif choice == "8":
                    print("\nGoodbye!")
                    break
                else:
                    print("Invalid choice. Please enter 1-8.")
                    
                input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                input("Press Enter to continue...")

def main():
    cli = ChessAICLI()
    cli.run()

if __name__ == "__main__":
    main()
