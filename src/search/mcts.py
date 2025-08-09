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

import os

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
from models.base_model import BaseModel
from utils.move_encoder import move_to_index, index_to_move, get_policy_vector_size

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
    """A node in the Monte Carlo Tree Search tree."""

    def __init__(self, parent: Optional['MCTSNode'], prior: float):
        self.parent = parent
        self.children: Dict[chess.Move, MCTSNode] = {}
        self.visit_count = 0
        self.total_action_value = 0.0
        self.prior = prior

    @property
    def mean_action_value(self) -> float:
        """Calculates the mean action value (Q-value) of the node."""
        if self.visit_count == 0:
            return 0.0
        return self.total_action_value / self.visit_count

    def select_child(self, c_puct: float) -> chess.Move:
        """Selects the best child node to explore using the PUCT formula."""
        best_score = -float('inf')
        best_move = None
        
        for move, child in self.children.items():
            puct_score = child.mean_action_value + c_puct * child.prior * \
                         math.sqrt(self.visit_count) / (1 + child.visit_count)
            
            if puct_score > best_score:
                best_score = puct_score
                best_move = move
                
        return best_move

    def expand(self, policy: Dict[chess.Move, float]):
        """Expands the node by creating children for all legal moves."""
        for move, prob in policy.items():
            self.children[move] = MCTSNode(parent=self, prior=prob)

    def backup(self, value: float):
        """Backpropagates the evaluation value up the tree."""
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_action_value += value
            value = -value
            node = node.parent

class MCTS:
    """
    A reusable MCTS implementation for chess, guided by a neural network.
    This class is model-agnostic and configured via the project's config file.
    """

    def __init__(
        self,
        model: BaseModel,
        config: Dict,
        device: str,
        random_seed: Optional[int] = None,
    ):
        """Create a new MCTS instance.

        Args:
            model: Neural network used for policy and value evaluation.
            config: Project configuration dictionary.
            device: Torch device string.
            random_seed: Optional seed for ``numpy`` and Python ``random``.
                When provided, MCTS will behave deterministically which is
                useful for testing.
        """
        self.model = model
        self.config = config['training']['mcts']
        self.device = device
        self.policy_size = get_policy_vector_size()
        self.root = None

        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

    @torch.no_grad()
    def search(
        self,
        board: chess.Board,
        simulations: Optional[int] = None,
        verbose: bool = False,
    ) -> chess.Move:
        """Run MCTS to find the best move.

        Args:
            board: The current chess board state.
            simulations: Number of simulations to run. Defaults to the value
                in the configuration.
            verbose: If ``True`` print debugging information including selected
                moves, visit counts and evaluation scores.
        """
        sim_count = simulations or self.config['simulations']
        self.root = MCTSNode(parent=None, prior=0.0)

        # Evaluate the root node
        value, policy_logits = self._evaluate(board)

        legal_moves = list(board.legal_moves)
        policy = self._get_policy_dict(policy_logits, legal_moves)

        # Add Dirichlet noise for exploration at the root
        if self.config.get('dirichlet_alpha'):
            dirichlet_noise = np.random.dirichlet(
                [self.config['dirichlet_alpha']] * len(legal_moves)
            )
            eps = self.config['dirichlet_epsilon']
            for i, move in enumerate(legal_moves):
                policy[move] = (1 - eps) * policy.get(move, 0.0) + eps * dirichlet_noise[i]

        self.root.expand(policy)
        self.root.backup(value)

        if verbose:
            print(f"Root evaluation value: {value:.3f}")

        # Run simulations
        for sim in range(sim_count):
            node = self.root
            search_path = [node]
            current_board = board.copy()
            first_move = None

            # 1. Selection: Traverse the tree
            while node.children:
                move = node.select_child(self.config['c_puct'])
                if first_move is None:
                    first_move = move
                node = node.children[move]
                current_board.push(move)
                search_path.append(node)

            # 2. Expansion & Evaluation
            if not current_board.is_game_over():
                value, policy_logits = self._evaluate(current_board)
                policy = self._get_policy_dict(
                    policy_logits, list(current_board.legal_moves)
                )
                node.expand(policy)
            else:
                # Terminal node: get game result
                result = current_board.result()
                if result == "1-0":
                    value = 1.0 if current_board.turn == chess.BLACK else -1.0
                elif result == "0-1":
                    value = -1.0 if current_board.turn == chess.BLACK else 1.0
                else:
                    value = 0.0

            if verbose and first_move is not None:
                print(
                    f"Simulation {sim + 1}: move {first_move}, evaluation {value:.3f}"
                )

            # 3. Backup
            node.backup(value)

            if verbose and first_move is not None:
                child = self.root.children[first_move]
                print(
                    f"\tVisits {child.visit_count}, mean value {child.mean_action_value:.3f}"
                )

        if verbose:
            print("Final visit counts:")
            for move, child in self.root.children.items():
                print(
                    f"\tMove {move}: Visits {child.visit_count}, Mean Value {child.mean_action_value:.3f}"
                )

        # Choose the best move based on visit counts
        return max(self.root.children.items(), key=lambda item: item[1].visit_count)[0]

    def _evaluate(self, board: chess.Board) -> (float, torch.Tensor):
        """
        Evaluates a board state with the neural network.
        Returns the value and policy logits.
        """
        from utils.board_utils import board_to_tensor
        
        board_tensor = board_to_tensor(board).unsqueeze(0).to(self.device)
        value_tensor, policy_logits = self.model(board_tensor)
        return value_tensor.item(), policy_logits.squeeze(0)

    def _get_policy_dict(self, policy_logits: torch.Tensor, legal_moves: list) -> Dict[chess.Move, float]:
        """
        Creates a dictionary mapping legal moves to their policy probabilities.
        """
        policy = torch.softmax(policy_logits, dim=0)
        policy_dict = {}
        for move in legal_moves:
            try:
                move_idx = move_to_index(move)
                if move_idx < self.policy_size:
                    policy_dict[move] = policy[move_idx].item()
            except Exception:
                # Fallback for moves not in the policy vector
                pass

        # Normalize probabilities for legal moves
        total_prob = sum(policy_dict.values())
        if total_prob > 0:
            for move in policy_dict:
                policy_dict[move] /= total_prob
        else: # If all legal moves have 0 prob, use uniform
             for move in legal_moves:
                policy_dict[move] = 1.0 / len(legal_moves)
        
        return policy_dict

def create_mcts_player(model_path: str, simulations: int = 800) -> MCTS:
    """
    Factory function to create MCTS player with specified parameters.
    
    Args:
        model_path: Path to trained neural network model
        simulations: Number of MCTS simulations per move
        
    Returns:
        Configured MCTS player
    """
    player = MCTS(model_path)
    player.num_simulations = simulations
    return player

if __name__ == "__main__":
    # Test the MCTS player
    model_path = "models/master_game_checkpoints/master_games_best.pt"
    
    if os.path.exists(model_path):
        print("🧪 TESTING MCTS PLAYER")
        print("=" * 40)
        
        player = MCTS(model_path)
        board = chess.Board()
        
        print(f"Starting position: {board.fen()}")
        move = player.search(board)
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