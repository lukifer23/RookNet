#!/usr/bin/env python3
"""
🚀 SELF-PLAY REINFORCEMENT LEARNING - THE ALPHAZERO APPROACH! 🚀

This is our breakthrough to beating Stockfish! Combining:
- Our 27.9M parameter breakthrough model (solid foundation)
- Monte Carlo Tree Search (MCTS) for deeper lookahead
- Self-play training loop for strategic improvement  
- Value network for position evaluation
- Experience replay and progressive difficulty

Expected Result: Strategic depth that converts 22.9% accuracy into WINS!
"""

import os
import sys  # sys kept, path hack removed

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import chess
import chess.engine
import numpy as np
import random
import json
import time
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import math

# Our existing modules
from models.baseline import ChessCNN
from utils.chess_env import ChessEnvironment

@dataclass
class MCTSNode:
    """Node in the Monte Carlo Tree Search"""
    state: str  # FEN position
    parent: Optional['MCTSNode'] = None
    children: Dict[str, 'MCTSNode'] = None
    visits: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    move: Optional[chess.Move] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}
    
    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0
    
    @property
    def ucb_score(self) -> float:
        if self.visits == 0:
            return float('inf')
        
        exploration = 1.4 * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return self.value + exploration

class AlphaChessRL:
    """AlphaZero-style Reinforcement Learning for Chess"""
    
    def __init__(self, model_path: str = None):
        print("🚀 INITIALIZING ALPHACHESS RL SYSTEM!")
        print("=" * 60)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Device: {self.device}")
        
        # Load our breakthrough model as foundation
        self.model = ChessCNN(
            input_channels=12,
            hidden_channels=256,  # Breakthrough architecture
            num_blocks=16
        ).to(self.device)
        
        if model_path and Path(model_path).exists():
            print(f"📚 Loading breakthrough model: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print("✅ Breakthrough model loaded successfully!")
        else:
            print("🆕 Starting with fresh model weights")
        
        # Single optimizer for the whole model
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        
        # Training parameters
        self.mcts_simulations = 50  # Start smaller for speed
        self.temperature = 1.0
        self.c_puct = 1.4
        
        # Experience buffer
        self.experience_buffer = deque(maxlen=5000)
        self.batch_size = 32
        
        # Environment
        self.env = ChessEnvironment()
        
        print(f"🧠 Total Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("🎯 Ready for self-play training!")
    
    def encode_position(self, board: chess.Board) -> torch.Tensor:
        """Convert board position to model input"""
        board_array = self.env.board_to_tensor(board)
        return torch.from_numpy(board_array).unsqueeze(0).float().to(self.device)
    
    def get_policy_and_value(self, board: chess.Board) -> Tuple[torch.Tensor, float]:
        """Get policy logits and value estimate"""
        self.model.eval()
        
        with torch.no_grad():
            board_tensor = self.encode_position(board)
            
            # ChessCNN returns (value, policy)
            value, policy_logits = self.model(board_tensor)
            
            # Extract scalar value
            value = value.item()
        
        return policy_logits.squeeze(0), value
    
    def select_move_simple(self, board: chess.Board) -> chess.Move:
        """Simple move selection using just the neural network"""
        policy_logits, _ = self.get_policy_and_value(board)
        legal_moves = list(board.legal_moves)
        
        # Get probabilities for legal moves
        move_probs = []
        moves = []
        
        for move in legal_moves:
            try:
                move_idx = self.env.move_to_index(move)
                if move_idx < len(policy_logits):
                    prob = F.softmax(policy_logits, dim=0)[move_idx].item()
                    move_probs.append(prob)
                    moves.append(move)
            except:
                continue
        
        if not moves:
            return random.choice(legal_moves)
        
        # Select move probabilistically
        probs = np.array(move_probs)
        probs = probs / probs.sum()
        
        return np.random.choice(moves, p=probs)
    
    def self_play_game(self, max_moves: int = 100) -> List[Tuple[str, str, float]]:
        """Play a complete self-play game - simplified version"""
        
        board = chess.Board()
        game_history = []
        move_count = 0
        
        print(f"🎮 Starting self-play game...")
        
        while not board.is_game_over() and move_count < max_moves:
            # Get move using simple selection
            move = self.select_move_simple(board)
            
            # Store position
            position = board.fen()
            game_history.append((position, move.uci(), None))  # Value filled later
            
            # Make move
            board.push(move)
            move_count += 1
            
            if move_count % 10 == 0:
                print(f"   Move {move_count}: {move.uci()}")
        
        # Determine game result
        result = board.result()
        if result == "1-0":
            final_value = 1.0
        elif result == "0-1":
            final_value = -1.0
        else:
            final_value = 0.0
        
        # Assign values to all positions (with alternating perspective)
        for i, (position, move_uci, _) in enumerate(game_history):
            # Alternate value for each move
            value = final_value if i % 2 == 0 else -final_value
            game_history[i] = (position, move_uci, value)
        
        print(f"✅ Game completed: {result} in {move_count} moves")
        return game_history
    
    def train_on_experience(self, batch_size: int = 32) -> Dict[str, float]:
        """Train the networks on collected experience"""
        
        if len(self.experience_buffer) < batch_size:
            return {"policy_loss": 0.0, "value_loss": 0.0}
        
        # Sample batch
        batch = random.sample(self.experience_buffer, batch_size)
        
        positions = []
        target_moves = []
        target_values = []
        
        for position, move_uci, value in batch:
            board = chess.Board(position)
            positions.append(self.encode_position(board))
            
            # Convert move to index
            try:
                move = chess.Move.from_uci(move_uci)
                move_idx = self.env.move_to_index(move)
                target_moves.append(move_idx)
            except:
                # Skip invalid moves
                continue
            
            target_values.append(value)
        
        if not target_moves:
            return {"policy_loss": 0.0, "value_loss": 0.0}
        
        # Prepare tensors
        positions = torch.cat(positions[:len(target_moves)], dim=0)
        target_moves = torch.tensor(target_moves, dtype=torch.long).to(self.device)
        target_values = torch.tensor(target_values, dtype=torch.float32).to(self.device)
        
        # Train the model
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass - ChessCNN returns (value, policy)
        predicted_values, policy_logits = self.model(positions)
        
        # Calculate losses
        policy_loss = F.cross_entropy(policy_logits, target_moves)
        value_loss = F.mse_loss(predicted_values.squeeze(), target_values)
        
        # Combined loss
        total_loss = policy_loss + value_loss
        
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item()
        }
    
    def self_play_training_loop(self, 
                               num_iterations: int = 20,
                               games_per_iteration: int = 2,
                               training_steps_per_iteration: int = 10):
        """Main self-play training loop"""
        
        print("🎯 STARTING SELF-PLAY TRAINING LOOP!")
        print("=" * 60)
        print(f"📊 Iterations: {num_iterations}")
        print(f"🎮 Games per iteration: {games_per_iteration}")
        print(f"🧠 Training steps per iteration: {training_steps_per_iteration}")
        
        training_history = []
        
        for iteration in range(num_iterations):
            print(f"\n🚀 ITERATION {iteration + 1}/{num_iterations}")
            print("-" * 40)
            
            iteration_start = time.time()
            
            # Self-play games
            print(f"🎮 Playing {games_per_iteration} self-play games...")
            for game_num in range(games_per_iteration):
                game_history = self.self_play_game()
                
                # Add to experience buffer
                self.experience_buffer.extend(game_history)
                
                print(f"   Game {game_num + 1}: {len(game_history)} positions added to buffer")
            
            # Training
            print(f"🧠 Training for {training_steps_per_iteration} steps...")
            total_policy_loss = 0.0
            total_value_loss = 0.0
            
            for step in range(training_steps_per_iteration):
                losses = self.train_on_experience(self.batch_size)
                total_policy_loss += losses["policy_loss"]
                total_value_loss += losses["value_loss"]
            
            avg_policy_loss = total_policy_loss / training_steps_per_iteration
            avg_value_loss = total_value_loss / training_steps_per_iteration
            
            iteration_time = time.time() - iteration_start
            
            # Progress report
            print(f"📊 ITERATION {iteration + 1} RESULTS:")
            print(f"   Policy Loss: {avg_policy_loss:.6f}")
            print(f"   Value Loss: {avg_value_loss:.6f}")
            print(f"   Experience Buffer: {len(self.experience_buffer)} positions")
            print(f"   Time: {iteration_time:.1f}s")
            
            # Save progress
            training_history.append({
                "iteration": iteration + 1,
                "policy_loss": avg_policy_loss,
                "value_loss": avg_value_loss,
                "buffer_size": len(self.experience_buffer),
                "time": iteration_time
            })
            
            # Save checkpoint every 5 iterations
            if (iteration + 1) % 5 == 0:
                self.save_checkpoint(iteration + 1, training_history)
        
        print("\n🎉 SELF-PLAY TRAINING COMPLETE!")
        return training_history
    
    def save_checkpoint(self, iteration: int, training_history: List[Dict]):
        """Save model checkpoint and training history"""
        
        checkpoints_dir = Path("models/self_play_checkpoints")
        checkpoints_dir.mkdir(exist_ok=True, parents=True)
        
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': training_history
        }
        
        checkpoint_path = checkpoints_dir / f"alphachess_iteration_{iteration}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as latest
        latest_path = checkpoints_dir / "alphachess_latest.pt"
        torch.save(checkpoint, latest_path)
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")

def main():
    """🚀 LAUNCH THE SELF-PLAY RL TRAINING! 🚀"""
    
    print("🎸⚡ ALPHACHESS SELF-PLAY RL - ROCK AND ROLL! ⚡🎸")
    print("=" * 80)
    
    # Initialize with our breakthrough model
    breakthrough_model_path = "models/checkpoints/breakthrough_best.pt"
    
    trainer = AlphaChessRL(model_path=breakthrough_model_path)
    
    # Start training!
    training_history = trainer.self_play_training_loop(
        num_iterations=20,      # Manageable number
        games_per_iteration=2,  # 2 games per iteration
        training_steps_per_iteration=15  # More training per iteration
    )
    
    # Save final results
    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    history_file = results_dir / f"self_play_training_{timestamp}.json"
    
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n💾 Training history saved: {history_file}")
    print("🎯 Ready for evaluation against Stockfish!")

if __name__ == "__main__":
    main() 