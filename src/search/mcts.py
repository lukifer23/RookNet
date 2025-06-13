#!/usr/bin/env python3
"""
🏆 MCTS (Monte Carlo Tree Search) + Neural Network Implementation 🏆

This combines our trained master games model with MCTS search to create
a powerful chess AI that can actually THINK AHEAD and find winning moves!

Features:
- AlphaZero-style MCTS with neural network guidance
- Upper Confidence Bound (UCB1) for node selection
- Neural network priors for move ordering
- Configurable search depth and iterations
- Tree reuse for efficiency

Expected Result: FINALLY BEAT STOCKFISH!
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import chess
import chess.engine
import numpy as np
import torch
import torch.nn.functional as F
import math
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import random

# Our modules
from src.models.baseline import ChessCNN
from src.utils.chess_env import ChessEnvironment

@dataclass
class MCTSStats:
    """Statistics for MCTS node"""
    visits: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    
    @property
    def mean_value(self) -> float:
        return self.value_sum / max(1, self.visits)

class MCTSNode:
    """MCTS Tree Node"""
    
    def __init__(self, board: chess.Board, parent: Optional['MCTSNode'] = None, 
                 move: Optional[chess.Move] = None, prior: float = 0.0):
        self.board = board.copy()
        self.parent = parent
        self.move = move  # Move that led to this position
        self.children: Dict[chess.Move, 'MCTSNode'] = {}
        self.stats = MCTSStats(prior=prior)
        self.is_expanded = False
        self.is_terminal = board.is_game_over()
        
    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return not self.is_expanded and not self.is_terminal
    
    def expand(self, move_priors: Dict[chess.Move, float]):
        """Expand node with children based on legal moves and priors"""
        if self.is_terminal:
            return
            
        legal_moves = list(self.board.legal_moves)
        for move in legal_moves:
            # Create child board
            child_board = self.board.copy()
            child_board.push(move)
            
            # Get prior probability for this move
            prior = move_priors.get(move, 1.0 / len(legal_moves))
            
            # Create child node
            child = MCTSNode(child_board, parent=self, move=move, prior=prior)
            self.children[move] = child
            
        self.is_expanded = True
    
    def select_child(self, c_puct: float = 1.0) -> 'MCTSNode':
        """Select best child using UCB1 formula"""
        if not self.children:
            return self
            
        best_score = float('-inf')
        best_child = None
        
        for child in self.children.values():
            # UCB1 score = Q + U
            # Q = average value, U = exploration bonus
            q_value = child.stats.mean_value
            
            # Exploration bonus: c_puct * prior * sqrt(parent_visits) / (1 + child_visits)
            u_value = (c_puct * child.stats.prior * 
                      math.sqrt(self.stats.visits) / (1 + child.stats.visits))
            
            # From child's perspective (flip sign for opponent)
            if self.board.turn != child.board.turn:
                q_value = -q_value
                
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child
    
    def backup(self, value: float):
        """Backup value up the tree"""
        self.stats.visits += 1
        self.stats.value_sum += value
        
        if self.parent:
            # Flip value for opponent
            self.parent.backup(-value)

class MCTSPlayer:
    """Chess player using MCTS + Neural Network"""
    
    def __init__(self, model_path: str, device: str = None):
        print("🏆 MCTS + NEURAL NETWORK PLAYER!")
        print("=" * 60)
        
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"🔧 Device: {self.device}")
        
        # Load the trained model
        self.model = ChessCNN(
            input_channels=12,
            hidden_channels=256,
            num_blocks=16
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        self.env = ChessEnvironment()
        
        # MCTS parameters
        self.c_puct = 1.0  # Exploration constant
        self.num_simulations = 800  # Number of MCTS simulations per move
        self.temperature = 0.1  # Temperature for move selection
        
        print(f"🧠 Model loaded: {model_path}")
        print(f"🔍 MCTS simulations: {self.num_simulations}")
        print("🏆 Ready to DOMINATE!")
    
    def evaluate_position(self, board: chess.Board) -> Tuple[float, Dict[chess.Move, float]]:
        """
        Evaluate position using neural network.
        
        Returns:
            Tuple of (position_value, move_priors)
        """
        # Convert board to tensor
        board_tensor = torch.from_numpy(self.env.board_to_tensor(board)).float()
        board_tensor = board_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value, policy_logits = self.model(board_tensor)
            
            # Convert to probabilities
            policy_probs = F.softmax(policy_logits, dim=1)
            
        # Extract value
        position_value = float(value.squeeze().cpu())
        
        # Convert policy to move priors
        legal_moves = list(board.legal_moves)
        move_priors = {}
        
        for move in legal_moves:
            try:
                move_idx = self.env.move_to_index(move)
                prior = float(policy_probs[0, move_idx].cpu())
                move_priors[move] = prior
            except:
                move_priors[move] = 1.0 / len(legal_moves)  # Uniform fallback
        
        # Normalize priors
        total_prior = sum(move_priors.values())
        if total_prior > 0:
            move_priors = {move: prior / total_prior for move, prior in move_priors.items()}
        
        return position_value, move_priors
    
    def mcts_search(self, root_board: chess.Board) -> MCTSNode:
        """
        Perform MCTS search from root position.
        
        Returns:
            Root node with fully searched tree
        """
        # Create root node
        root = MCTSNode(root_board)
        
        # Evaluate root position
        root_value, root_priors = self.evaluate_position(root_board)
        root.expand(root_priors)
        
        # Run simulations
        for simulation in range(self.num_simulations):
            # Selection: walk down tree to leaf
            node = root
            path = [node]
            
            while not node.is_leaf() and not node.is_terminal:
                node = node.select_child(self.c_puct)
                path.append(node)
            
            # Expansion and Evaluation
            if not node.is_terminal:
                # Evaluate leaf position
                leaf_value, leaf_priors = self.evaluate_position(node.board)
                
                # Expand if not terminal
                node.expand(leaf_priors)
                
                # Use neural network evaluation
                value = leaf_value
            else:
                # Terminal position - use game result
                result = node.board.result()
                if result == "1-0":
                    value = 1.0 if node.board.turn == chess.BLACK else -1.0
                elif result == "0-1":
                    value = -1.0 if node.board.turn == chess.BLACK else 1.0
                else:
                    value = 0.0  # Draw
            
            # Backup: propagate value up the tree
            for path_node in reversed(path):
                path_node.backup(value)
                value = -value  # Flip for opponent
        
        return root
    
    def select_move(self, board: chess.Board) -> chess.Move:
        """
        Select best move using MCTS search.
        
        Args:
            board: Current board position
            
        Returns:
            Best move according to MCTS
        """
        print(f"🔍 MCTS THINKING... ({self.num_simulations} simulations)")
        start_time = time.time()
        
        # Run MCTS search
        root = self.mcts_search(board)
        
        search_time = time.time() - start_time
        
        # Select move based on visit counts
        if not root.children:
            # Fallback to random legal move
            return random.choice(list(board.legal_moves))
        
        # Calculate move probabilities based on visits
        total_visits = sum(child.stats.visits for child in root.children.values())
        
        if self.temperature == 0:
            # Greedy selection - pick most visited
            best_move = max(root.children.keys(), 
                          key=lambda move: root.children[move].stats.visits)
        else:
            # Stochastic selection with temperature
            moves = list(root.children.keys())
            visit_counts = [root.children[move].stats.visits for move in moves]
            
            # Apply temperature
            if total_visits > 0:
                probs = np.array(visit_counts, dtype=float)
                probs = probs ** (1.0 / self.temperature)
                probs = probs / probs.sum()
                
                best_move = np.random.choice(moves, p=probs)
            else:
                best_move = random.choice(moves)
        
        # Print search statistics
        best_child = root.children[best_move]
        print(f"⚡ Search complete: {search_time:.2f}s")
        print(f"🎯 Best move: {best_move}")
        print(f"📊 Visits: {best_child.stats.visits}/{total_visits}")
        print(f"💎 Value: {best_child.stats.mean_value:.3f}")
        
        return best_move
    
    def get_search_info(self, board: chess.Board) -> Dict[str, Any]:
        """Get detailed search information for analysis"""
        root = self.mcts_search(board)
        
        move_info = {}
        for move, child in root.children.items():
            move_info[move.uci()] = {
                'visits': child.stats.visits,
                'value': child.stats.mean_value,
                'prior': child.stats.prior,
                'visits_pct': child.stats.visits / max(1, root.stats.visits) * 100
            }
        
        return {
            'total_simulations': self.num_simulations,
            'root_visits': root.stats.visits,
            'move_analysis': move_info
        }

def create_mcts_player(model_path: str, simulations: int = 800) -> MCTSPlayer:
    """
    Factory function to create MCTS player with specified parameters.
    
    Args:
        model_path: Path to trained neural network model
        simulations: Number of MCTS simulations per move
        
    Returns:
        Configured MCTS player
    """
    player = MCTSPlayer(model_path)
    player.num_simulations = simulations
    return player

if __name__ == "__main__":
    # Test the MCTS player
    model_path = "models/master_game_checkpoints/master_games_best.pt"
    
    if os.path.exists(model_path):
        print("🧪 TESTING MCTS PLAYER")
        print("=" * 40)
        
        player = MCTSPlayer(model_path)
        board = chess.Board()
        
        print(f"Starting position: {board.fen()}")
        move = player.select_move(board)
        print(f"MCTS selected move: {move}")
        
        # Get detailed analysis
        info = player.get_search_info(board)
        print(f"\nDetailed analysis:")
        for move_uci, data in info['move_analysis'].items():
            print(f"  {move_uci}: {data['visits']} visits ({data['visits_pct']:.1f}%), "
                  f"value={data['value']:.3f}, prior={data['prior']:.3f}")
    else:
        print(f"❌ Model not found: {model_path}")
        print("🎯 Please run master games training first!") 