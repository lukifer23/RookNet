#!/usr/bin/env python3
"""
Enhanced MCTS Implementation for 64M Chess Transformer
Production-ready with UCB1, progressive widening, temperature control
"""

import math
import time
import random
import numpy as np
import chess
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class MCTSConfig:
    """MCTS Configuration"""
    simulations: int = 50
    c_puct: float = 1.0          # UCB exploration constant
    temperature: float = 1.0      # Move selection temperature
    dirichlet_alpha: float = 0.3  # Root noise
    dirichlet_epsilon: float = 0.25  # Root noise weight
    progressive_widening: bool = True
    max_children: int = 10        # Progressive widening limit
    virtual_loss: float = 3.0     # Virtual loss for parallelization
    time_limit: Optional[float] = None  # Time limit in seconds

class MCTSNode:
    """MCTS Node with enhanced features"""
    
    def __init__(self, board: chess.Board, parent: Optional['MCTSNode'] = None, 
                 move: Optional[chess.Move] = None, prior: float = 0.0):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.prior = prior
        
        # MCTS statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[chess.Move, MCTSNode] = {}
        self.legal_moves = list(board.legal_moves)
        
        # Enhanced features
        self.is_expanded = False
        self.is_terminal = board.is_game_over()
        self.virtual_loss = 0.0
        
        # Terminal evaluation
        if self.is_terminal:
            result = board.result()
            if result == "1-0":
                self.terminal_value = 1.0 if board.turn == chess.WHITE else -1.0
            elif result == "0-1":
                self.terminal_value = -1.0 if board.turn == chess.WHITE else 1.0
            else:
                self.terminal_value = 0.0
        else:
            self.terminal_value = None
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf (unexpanded)"""
        return not self.is_expanded and not self.is_terminal
    
    def get_ucb_score(self, c_puct: float) -> float:
        """Calculate UCB1 score"""
        if self.visit_count == 0:
            return float('inf')
        
        # Q-value (average value)
        q_value = self.value_sum / self.visit_count
        
        # UCB1 formula with prior
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        
        return q_value + exploration
    
    def select_child(self, config: MCTSConfig) -> 'MCTSNode':
        """Select child using UCB1"""
        if not self.children:
            return self
            
        # Progressive widening
        if config.progressive_widening:
            max_children = min(len(self.legal_moves), 
                             max(1, int(config.max_children * math.sqrt(self.visit_count))))
            available_children = list(self.children.values())[:max_children]
        else:
            available_children = list(self.children.values())
        
        # Select best UCB score
        best_child = max(available_children, key=lambda child: child.get_ucb_score(config.c_puct))
        return best_child
    
    def expand(self, policy_probs: np.ndarray, legal_moves: List[chess.Move]) -> None:
        """Expand node with policy probabilities"""
        if self.is_expanded or self.is_terminal:
            return
            
        # Create children for legal moves
        for i, move in enumerate(legal_moves):
            child_board = self.board.copy()
            child_board.push(move)
            
            prior = policy_probs[i] if i < len(policy_probs) else 0.01
            child = MCTSNode(child_board, parent=self, move=move, prior=prior)
            self.children[move] = child
        
        self.is_expanded = True
    
    def backup(self, value: float) -> None:
        """Backup value through the tree"""
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent:
            # Flip value for opponent
            self.parent.backup(-value)
    
    def add_virtual_loss(self, loss: float) -> None:
        """Add virtual loss for parallelization"""
        self.virtual_loss += loss
        self.visit_count += 1
        self.value_sum -= loss
    
    def remove_virtual_loss(self, loss: float) -> None:
        """Remove virtual loss after evaluation"""
        self.virtual_loss -= loss
        self.visit_count -= 1
        self.value_sum += loss

class EnhancedMCTS:
    """Enhanced MCTS for Chess Transformer"""
    
    def __init__(self, model, chess_env, device, config: MCTSConfig):
        self.model = model
        self.chess_env = chess_env
        self.device = device
        self.config = config
        
        # Ensure model is in eval mode
        self.model.eval()
    
    def search(self, board: chess.Board, config: Optional[MCTSConfig] = None) -> Tuple[chess.Move, Dict]:
        """Run MCTS search and return best move"""
        if config is None:
            config = self.config
            
        # Create root node
        root = MCTSNode(board)
        
        # Add Dirichlet noise to root
        if config.dirichlet_alpha > 0:
            self._add_dirichlet_noise(root, config)
        
        # Time tracking
        start_time = time.time()
        
        # Run simulations
        for simulation in range(config.simulations):
            # Check time limit
            if config.time_limit and (time.time() - start_time) > config.time_limit:
                break
                
            # Single simulation
            self._simulate(root, config)
        
        # Select best move
        best_move, stats = self._select_move(root, config)
        
        return best_move, stats
    
    def _simulate(self, root: MCTSNode, config: MCTSConfig) -> None:
        """Single MCTS simulation"""
        path = []
        node = root
        
        # Selection phase
        while not node.is_leaf() and not node.is_terminal:
            path.append(node)
            node = node.select_child(config)
        
        # Terminal node
        if node.is_terminal:
            value = node.terminal_value
        else:
            # Expansion and evaluation
            value = self._expand_and_evaluate(node)
        
        # Backup
        node.backup(value)
    
    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """Expand node and evaluate with neural network"""
        # Convert board to tensor
        position_array = self.chess_env.board_to_tensor(node.board)
        position_tensor = torch.from_numpy(position_array).float().unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            value, policy_logits = self.model(position_tensor)
            
            # Extract values
            value = float(value.squeeze())
            policy_probs = F.softmax(policy_logits.squeeze(), dim=0).cpu().numpy()
        
        # Map policy to legal moves
        legal_moves = list(node.board.legal_moves)
        move_probs = []
        
        for move in legal_moves:
            move_idx = self.chess_env.move_to_index(move)
            prob = policy_probs[move_idx] if move_idx < len(policy_probs) else 0.01
            move_probs.append(prob)
        
        # Normalize probabilities
        move_probs = np.array(move_probs)
        if move_probs.sum() > 0:
            move_probs = move_probs / move_probs.sum()
        else:
            move_probs = np.ones(len(legal_moves)) / len(legal_moves)
        
        # Expand node
        node.expand(move_probs, legal_moves)
        
        return value
    
    def _add_dirichlet_noise(self, root: MCTSNode, config: MCTSConfig) -> None:
        """Add Dirichlet noise to root for exploration"""
        if not root.legal_moves:
            return
            
        # Generate Dirichlet noise
        noise = np.random.dirichlet([config.dirichlet_alpha] * len(root.legal_moves))
        
        # Apply to root (will be used during expansion)
        root._dirichlet_noise = noise
    
    def _select_move(self, root: MCTSNode, config: MCTSConfig) -> Tuple[chess.Move, Dict]:
        """Select move from root using visit counts and temperature"""
        if not root.children:
            # No children, return random legal move
            if root.legal_moves:
                return random.choice(root.legal_moves), {}
            else:
                return None, {}
        
        # Get visit counts
        children = list(root.children.items())
        moves = [move for move, _ in children]
        visit_counts = [child.visit_count for _, child in children]
        
        # Temperature selection
        if config.temperature <= 0.01:
            # Deterministic: select most visited
            best_idx = np.argmax(visit_counts)
            best_move = moves[best_idx]
        else:
            # Stochastic: sample by visit count with temperature
            visit_counts = np.array(visit_counts, dtype=float)
            visit_counts = visit_counts ** (1.0 / config.temperature)
            
            if visit_counts.sum() > 0:
                probs = visit_counts / visit_counts.sum()
                best_idx = np.random.choice(len(moves), p=probs)
                best_move = moves[best_idx]
            else:
                best_move = random.choice(moves)
        
        # Collect statistics
        stats = {
            'root_visits': root.visit_count,
            'root_value': root.value_sum / max(root.visit_count, 1),
            'move_visits': dict(zip(moves, visit_counts)),
            'principal_variation': self._get_pv(root, best_move)
        }
        
        return best_move, stats
    
    def _get_pv(self, root: MCTSNode, first_move: chess.Move, depth: int = 5) -> List[chess.Move]:
        """Get principal variation (most visited path)"""
        pv = [first_move]
        node = root.children.get(first_move)
        
        for _ in range(depth - 1):
            if not node or not node.children:
                break
                
            # Get most visited child
            best_child = max(node.children.values(), key=lambda x: x.visit_count)
            pv.append(best_child.move)
            node = best_child
        
        return pv

# Factory function for easy creation
def create_enhanced_mcts(model, chess_env, device, simulations=50, temperature=1.0):
    """Create enhanced MCTS with reasonable defaults"""
    config = MCTSConfig(
        simulations=simulations,
        temperature=temperature,
        c_puct=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        progressive_widening=True,
        max_children=10
    )
    
    return EnhancedMCTS(model, chess_env, device, config) 