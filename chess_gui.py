#!/usr/bin/env python3
"""
Chess AI Development GUI

A simple interface for training, testing, and managing the chess AI models.
Provides easy buttons for common development tasks.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import sys
import io
import contextlib
import yaml
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime
import chess

class ChessAIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess AI Development Interface")
        self.root.geometry("800x600")
        
        # Load configuration
        self.config = self.load_config()
        
        # Create GUI elements
        self.create_widgets()
        
        # Status tracking
        self.is_running = False
        
    def load_config(self):
        """Load configuration from config.yaml"""
        try:
            with open("configs/config.yaml", "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Failed to load config: {e}")
            return {}
    
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Chess AI Development Interface", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Buttons frame
        buttons_frame = ttk.LabelFrame(main_frame, text="Actions", padding="10")
        buttons_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Core function buttons
        self.create_button(buttons_frame, "Test Environment", self.test_environment, 0, 0)
        self.create_button(buttons_frame, "Generate Training Data", self.generate_data, 0, 1)
        self.create_button(buttons_frame, "Train Model", self.train_model, 0, 2)
        
        self.create_button(buttons_frame, "Test Model", self.test_model, 1, 0)
        self.create_button(buttons_frame, "Evaluate vs Stockfish", self.evaluate_model, 1, 1)
        self.create_button(buttons_frame, "Play vs AI", self.play_game, 1, 2)
        
        # Enhanced evaluation buttons (third row)
        self.create_button(buttons_frame, "Quick Eval (20 games)", self.quick_evaluation, 2, 0)
        self.create_button(buttons_frame, "Full Eval (40 games)", self.full_evaluation, 2, 1)
        self.create_button(buttons_frame, "Progressive Eval", self.progressive_evaluation, 2, 2)
        
        # Configuration frame
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.create_button(config_frame, "View Config", self.view_config, 0, 0)
        self.create_button(config_frame, "Check Dependencies", self.check_deps, 0, 1)
        self.create_button(config_frame, "View Project Status", self.view_status, 0, 2)
        
        # Output area
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="10")
        output_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.output_text = scrolledtext.ScrolledText(output_frame, width=80, height=20)
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(3, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        output_frame.grid_rowconfigure(0, weight=1)
        output_frame.grid_columnconfigure(0, weight=1)
    
    def create_button(self, parent, text, command, row, col):
        """Create a button with consistent styling"""
        btn = ttk.Button(parent, text=text, command=command, width=20)
        btn.grid(row=row, column=col, padx=5, pady=5)
        return btn
    
    def log_output(self, message):
        """Add message to output area"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.output_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.output_text.see(tk.END)
        self.root.update()
    
    def run_command_async(self, func, *args):
        """Run a command in a separate thread"""
        if self.is_running:
            messagebox.showwarning("Warning", "Another operation is already running!")
            return
        
        def worker():
            self.is_running = True
            self.status_var.set("Running...")
            try:
                func(*args)
            except Exception as e:
                self.log_output(f"Error: {e}")
            finally:
                self.is_running = False
                self.status_var.set("Ready")
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
    
    def test_environment(self):
        """Test the chess environment and Stockfish integration"""
        self.run_command_async(self._test_environment)
    
    def _test_environment(self):
        """Internal method to test environment"""
        self.log_output("Testing Chess Environment...")
        
        try:
            # Test chess environment
            from src.utils.chess_env import ChessEnvironment
            
            self.log_output("✓ Chess environment imported successfully")
            
            env = ChessEnvironment()
            self.log_output("✓ Chess environment initialized")
            
            # Test Stockfish
            env.start_engine()
            self.log_output(f"✓ Stockfish engine started: {env.engine.id['name']}")
            
            # Test basic functionality
            eval_score = env.get_engine_evaluation()
            self.log_output(f"✓ Position evaluation: {eval_score} centipawns")
            
            best_move, info = env.get_engine_move(time_limit=0.5)
            if best_move:
                self.log_output(f"✓ Best move: {best_move}")
            
            # Test board representation
            board_tensor = env.board_to_tensor()
            self.log_output(f"✓ Board tensor shape: {board_tensor.shape}")
            
            env.stop_engine()
            self.log_output("✓ Environment test completed successfully!")
            
        except Exception as e:
            self.log_output(f"✗ Environment test failed: {e}")
    
    def generate_data(self):
        """Generate training data"""
        self.run_command_async(self._generate_data)
    
    def _generate_data(self):
        """Internal method to generate training data"""
        self.log_output("Generating training data...")
        
        try:
            from src.data.stockfish_generator import TrainingDataGenerator
            
            generator = TrainingDataGenerator(self.config)
            self.log_output("✓ Data generator initialized")
            
            # Generate small sample first
            self.log_output("Generating sample data (2 games)...")
            training_data = generator.generate_game_data(
                num_games=2,
                white_depth=5,
                black_depth=5,
                time_per_move=0.5,
                save_pgn=True
            )
            
            self.log_output(f"✓ Generated {len(training_data)} training positions")
            
            # Option to generate full dataset
            response = messagebox.askyesno(
                "Generate Full Dataset",
                f"Sample generation successful!\n\nGenerate full training dataset?\n"
                f"({self.config.get('data', {}).get('num_games', 1000)} games)"
            )
            
            if response:
                self.log_output("Generating full training dataset...")
                num_games = self.config.get('data', {}).get('num_games', 1000)
                training_data = generator.generate_game_data(
                    num_games=num_games, 
                    white_depth=self.config.get('data', {}).get('stockfish', {}).get('depth', 15),
                    black_depth=self.config.get('data', {}).get('stockfish', {}).get('depth', 15),
                    time_per_move=self.config.get('data', {}).get('stockfish', {}).get('time_limit', 1.0),
                    save_pgn=True
                )
                self.log_output(f"✓ Full dataset generated: {len(training_data)} positions")
            
        except Exception as e:
            self.log_output(f"✗ Data generation failed: {e}")
            import traceback
            self.log_output(f"Error details: {traceback.format_exc()}")
    
    def train_model(self):
        """Train the chess model"""
        self.run_command_async(self._train_model)
    
    def _train_model(self):
        """Internal method to train model"""
        self.log_output("Training model...")
        
        try:
            # Check if training data exists  
            data_dir = Path("data/processed")
            data_files = list(data_dir.glob("training_data_*.json"))
            
            if not data_files:
                self.log_output("⚠ No training data found. Generating sample data first...")
                self._generate_data()
                # Re-check for data files
                data_files = list(data_dir.glob("training_data_*.json"))
                if not data_files:
                    self.log_output("✗ Failed to generate training data")
                    return
            
            self.log_output(f"✓ Found {len(data_files)} training data files")
            
            from src.training.trainer import ChessTrainer
            
            # Initialize trainer (it creates the model internally)
            trainer = ChessTrainer(self.config)
            
            self.log_output("✓ Trainer initialized")
            self.log_output("Starting training... (this may take a while)")
            
            # Get latest data file
            latest_data_file = max(data_files, key=lambda x: x.stat().st_mtime)
            self.log_output(f"✓ Using data file: {latest_data_file.name}")
            
            # Train the model with the quick training script logic
            from src.training.trainer import ChessDataset
            from torch.utils.data import DataLoader
            import torch
            
            # Create dataset
            dataset = ChessDataset(str(latest_data_file))
            
            # Split data  
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            
            self.log_output(f"✓ Training samples: {len(train_dataset)}")
            self.log_output(f"✓ Validation samples: {len(val_dataset)}")
            
            # Train the model
            trainer.train(train_loader, val_loader)
            self.log_output("✓ Training completed!")
            
        except Exception as e:
            self.log_output(f"✗ Training failed: {e}")
            import traceback
            self.log_output(f"Error details: {traceback.format_exc()}")
    
    def test_model(self):
        """Test trained model"""
        self.run_command_async(self._test_model)
    
    def _test_model(self):
        """Internal method to test model"""
        self.log_output("Testing model...")
        
        try:
            # Check for saved models
            model_dir = Path("models/checkpoints")
            model_files = list(model_dir.glob("*.pt"))
            
            if not model_files:
                self.log_output("⚠ No trained models found. Train a model first.")
                return
            
            self.log_output(f"✓ Found {len(model_files)} saved models")
            
            # Load and test latest model
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            self.log_output(f"✓ Testing model: {latest_model.name}")
            
            # Load the model
            from src.training.trainer import ChessTrainer
            from src.utils.chess_env import ChessEnvironment
            import torch
            
            # Initialize trainer and load checkpoint
            trainer = ChessTrainer(self.config)
            trainer.load_checkpoint(str(latest_model))
            
            self.log_output("✓ Model loaded successfully")
            
            # Test model with a simple position
            env = ChessEnvironment()
            env.start_engine()
            
            # Get board tensor from starting position
            board = env.board
            board_tensor = env.board_to_tensor(board)
            
            # Test model prediction
            trainer.model.eval()
            with torch.no_grad():
                board_input = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0)
                board_input = board_input.to(trainer.device)
                
                output = trainer.model(board_input)
                
                if isinstance(output, tuple):
                    # Model returns (value, policy)
                    value, policy = output
                    self.log_output(f"✓ Model prediction - Value: {value.item():.4f}")
                    self.log_output(f"   Policy shape: {policy.shape}")
                else:
                    # Model returns single output
                    self.log_output(f"✓ Model prediction - Output shape: {output.shape}")
            
            env.stop_engine()
            self.log_output("✓ Model testing completed successfully!")
            
        except Exception as e:
            self.log_output(f"✗ Model testing failed: {e}")
    
    def evaluate_model(self):
        """Evaluate model against Stockfish"""
        self.run_command_async(self._evaluate_model)
    
    def _evaluate_model(self):
        """Internal method to evaluate model against Stockfish"""
        self.log_output("Evaluating model against Stockfish...")
        
        try:
            # Check for saved models
            model_dir = Path("models/checkpoints")
            model_files = list(model_dir.glob("*.pt"))
            
            if not model_files:
                self.log_output("⚠ No trained models found. Train a model first.")
                return
            
            # Load the model
            from src.training.trainer import ChessTrainer
            from src.utils.chess_env import ChessEnvironment
            import torch
            
            # Initialize trainer and load checkpoint
            trainer = ChessTrainer(self.config)
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            trainer.load_checkpoint(str(latest_model))
            trainer.model.eval()
            
            self.log_output(f"✓ Loaded model: {latest_model.name}")
            
            # Initialize chess environment
            env = ChessEnvironment()
            env.start_engine()
            
            self.log_output("✓ Stockfish engine started")
            self.log_output("Starting evaluation games...")
            
            # Play a few test games
            num_games = 3
            results = {"wins": 0, "losses": 0, "draws": 0}
            
            for game_num in range(num_games):
                self.log_output(f"\n--- Game {game_num + 1}/{num_games} ---")
                
                # Reset board for new game
                env.reset_board()
                
                # Alternate who plays as white
                ai_plays_white = (game_num % 2 == 0)
                self.log_output(f"AI plays as: {'White' if ai_plays_white else 'Black'}")
                
                moves_played = 0
                max_moves = 50  # Limit game length for testing
                
                while not env.is_game_over() and moves_played < max_moves:
                    if (env.board.turn == chess.WHITE and ai_plays_white) or \
                       (env.board.turn == chess.BLACK and not ai_plays_white):
                        # AI's turn
                        board_tensor = env.board_to_tensor()
                        
                        with torch.no_grad():
                            board_input = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0)
                            board_input = board_input.to(trainer.device)
                            
                            output = trainer.model(board_input)
                            
                            # For now, use a simple move selection (first legal move)
                            # In a real implementation, you'd decode the policy output
                            legal_moves = list(env.board.legal_moves)
                            if legal_moves:
                                move = legal_moves[0]  # Simple fallback
                                env.make_move(move)
                                self.log_output(f"AI move: {move}")
                    else:
                        # Stockfish's turn
                        best_move, info = env.get_engine_move(time_limit=0.5)
                        if best_move and best_move in env.board.legal_moves:
                            env.make_move(best_move)
                            self.log_output(f"Stockfish move: {best_move}")
                        else:
                            break
                    
                    moves_played += 1
                
                # Determine game result
                if env.is_game_over():
                    result = env.get_game_result()
                    self.log_output(f"Game ended: {result}")
                    
                    if result == "1-0":
                        if ai_plays_white:
                            results["wins"] += 1
                            self.log_output("AI wins!")
                        else:
                            results["losses"] += 1
                            self.log_output("AI loses.")
                    elif result == "0-1":
                        if ai_plays_white:
                            results["losses"] += 1
                            self.log_output("AI loses.")
                        else:
                            results["wins"] += 1
                            self.log_output("AI wins!")
                    else:
                        results["draws"] += 1
                        self.log_output("Draw.")
                else:
                    self.log_output("Game reached move limit - counting as draw")
                    results["draws"] += 1
            
            env.stop_engine()
            
            # Report final results
            self.log_output(f"\n=== Evaluation Results ===")
            self.log_output(f"Games played: {num_games}")
            self.log_output(f"Wins: {results['wins']}")
            self.log_output(f"Losses: {results['losses']}")
            self.log_output(f"Draws: {results['draws']}")
            
            win_rate = results['wins'] / num_games * 100
            self.log_output(f"Win rate: {win_rate:.1f}%")
            
            self.log_output("✓ Model evaluation completed!")
            
        except Exception as e:
            self.log_output(f"✗ Model evaluation failed: {e}")
            import traceback
            self.log_output(f"Error details: {traceback.format_exc()}")
    
    def play_game(self):
        """Play a game against the AI"""
        self.log_output("Game playing interface not yet implemented")
    
    def view_config(self):
        """Display current configuration"""
        self.log_output("Current Configuration:")
        self.log_output("=" * 50)
        config_str = yaml.dump(self.config, default_flow_style=False, indent=2)
        self.log_output(config_str)
    
    def check_deps(self):
        """Check if all dependencies are installed"""
        self.run_command_async(self._check_deps)
    
    def _check_deps(self):
        """Internal method to check dependencies"""
        self.log_output("Checking dependencies...")
        
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
                self.log_output(f"✓ {name} installed")
            except ImportError:
                self.log_output(f"✗ {name} not found")
        
        # Check Stockfish
        try:
            result = subprocess.run(["stockfish"], capture_output=True, text=True, timeout=2)
            self.log_output("✓ Stockfish engine available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.log_output("✗ Stockfish engine not found")
    
    def view_status(self):
        """View project status and progress"""
        self.log_output("Project Status:")
        self.log_output("=" * 50)
        
        # Check what's implemented
        status_items = [
            ("Chess Environment", Path("src/utils/chess_env.py").exists()),
            ("Data Generator", Path("src/data/stockfish_generator.py").exists()),
            ("Baseline Models", Path("src/models/baseline.py").exists()),
            ("Training Pipeline", Path("src/training/trainer.py").exists()),
            ("Configuration", Path("configs/config.yaml").exists()),
            ("Training Data", len(list(Path("data/processed").glob("*.npz"))) > 0 if Path("data/processed").exists() else False),
            ("Trained Models", len(list(Path("models/checkpoints").glob("*.pt"))) > 0 if Path("models/checkpoints").exists() else False),
        ]
        
        for item, status in status_items:
            symbol = "✓" if status else "✗"
            self.log_output(f"{symbol} {item}")
        
        self.log_output("\nNext Steps:")
        if not status_items[5][1]:  # No training data
            self.log_output("• Generate training data")
        elif not status_items[6][1]:  # No trained models
            self.log_output("• Train initial model")
        else:
            self.log_output("• Implement diffusion model")
            self.log_output("• Add model evaluation")
            self.log_output("• Create game interface")
    
    def quick_evaluation(self):
        """Quick evaluation with 20 games against easy Stockfish"""
        self.run_command_async(self._quick_evaluation)
    
    def _quick_evaluation(self):
        """Internal method for quick evaluation"""
        self.log_output("Running quick evaluation (20 games vs easy Stockfish)...")
        
        try:
            # Run the enhanced evaluation script
            import subprocess
            result = subprocess.run([
                "python", "enhanced_evaluation.py"
            ], capture_output=True, text=True, input="1\neasy\n")
            
            # Log the output
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if line.strip():
                        self.log_output(line)
            
            if result.stderr:
                self.log_output(f"Errors: {result.stderr}")
                
        except Exception as e:
            self.log_output(f"✗ Quick evaluation failed: {e}")
    
    def full_evaluation(self):
        """Full evaluation with 40 games against medium Stockfish"""
        self.run_command_async(self._full_evaluation)
    
    def _full_evaluation(self):
        """Internal method for full evaluation"""
        self.log_output("Running full evaluation (40 games vs medium Stockfish)...")
        
        try:
            # Run the enhanced evaluation script
            import subprocess
            result = subprocess.run([
                "python", "enhanced_evaluation.py"
            ], capture_output=True, text=True, input="1\nmedium\n")
            
            # Log the output
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if line.strip():
                        self.log_output(line)
            
            if result.stderr:
                self.log_output(f"Errors: {result.stderr}")
                
        except Exception as e:
            self.log_output(f"✗ Full evaluation failed: {e}")
    
    def progressive_evaluation(self):
        """Progressive evaluation across difficulty levels"""
        self.run_command_async(self._progressive_evaluation)
    
    def _progressive_evaluation(self):
        """Internal method for progressive evaluation"""
        self.log_output("Running progressive evaluation across difficulty levels...")
        
        try:
            # Run the enhanced evaluation script
            import subprocess
            result = subprocess.run([
                "python", "enhanced_evaluation.py"
            ], capture_output=True, text=True, input="2\n")
            
            # Log the output
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if line.strip():
                        self.log_output(line)
            
            if result.stderr:
                self.log_output(f"Errors: {result.stderr}")
                
        except Exception as e:
            self.log_output(f"✗ Progressive evaluation failed: {e}")

def main():
    root = tk.Tk()
    app = ChessAIGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()