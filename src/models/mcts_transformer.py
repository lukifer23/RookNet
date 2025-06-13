"""
🏆 MCTS + Transformer Chess AI
Ultimate hybrid system combining 141M parameter transformer with Monte Carlo Tree Search

Architecture: ChessTransformer (strategic evaluation) + MCTS (tactical search) = Superhuman play
Target: Beat Stockfish depth 15+, achieve 2600+ ELO
"""

import torch
import chess
import chess.engine
import numpy as np
import math
import random
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

from src.models.chess_transformer import ChessTransformer, create_chess_transformer
from src.utils.chess_utils import board_to_tensor

@dataclass
class MCTSConfig:
    """Configuration for MCTS + Transformer system"""
    simulations: int = 800  # High-quality search
    c_puct: float = 1.5     # Exploration constant
    dirichlet_alpha: float = 0.3  # Root noise
    dirichlet_epsilon: float = 0.25  # Noise weight
    max_depth: int = 100    # Search depth limit
    time_limit: float = 5.0  # Time per move (seconds)
    temperature: float = 1.0  # Move selection temperature
    virtual_loss: int = 3   # Virtual loss for parallel search
    threads: int = 4        # Parallel simulations

class MCTSNode:
    """Enhanced MCTS node with transformer integration"""
    
    def __init__(self, board: chess.Board, parent: Optional['MCTSNode'] = None, 
                 move: Optional[chess.Move] = None, prior: float = 0.0):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.prior = prior
        
        # MCTS statistics
        self.visits = 0
        self.value_sum = 0.0
        self.virtual_losses = 0
        
        # Children and expansion
        self.children: Dict[chess.Move, 'MCTSNode'] = {}
        self.expanded = False
        
        # Transformer evaluation cache
        self.transformer_value: Optional[float] = None
        self.policy_priors: Optional[Dict[chess.Move, float]] = None
        
        # Threading
        self.lock = threading.Lock()
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf (unexpanded)"""
        return not self.expanded
    
    def is_terminal(self) -> bool:
        """Check if position is terminal"""
        return self.board.is_game_over()
    
    def get_value(self) -> float:
        """Get average value of node"""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    def get_ucb_score(self, c_puct: float) -> float:
        """Calculate UCB score for node selection"""
        if self.visits == 0:
            return float('inf')
        
        # UCB1 with prior policy
        exploitation = self.get_value()
        exploration = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        
        return exploitation + exploration
    
    def add_virtual_loss(self):
        """Add virtual loss for parallel search"""
        with self.lock:
            self.virtual_losses += 1
            self.visits += 1
            self.value_sum -= 1.0  # Pessimistic virtual loss
    
    def revert_virtual_loss(self):
        """Revert virtual loss after backup"""
        with self.lock:
            self.virtual_losses -= 1
            self.visits -= 1
            self.value_sum += 1.0
    
    def backup(self, value: float):
        """Backup value through the tree"""
        with self.lock:
            self.visits += 1
            self.value_sum += value

class MCTSTransformerPlayer:
    """🏆 Ultimate Chess AI: MCTS + Transformer Hybrid"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 model_config: str = "superhuman",
                 mcts_config: MCTSConfig = None,
                 device: str = "mps"):
        
        self.device = torch.device(device)
        self.mcts_config = mcts_config or MCTSConfig()
        
        # Load transformer model
        if model_path and model_path != "create_new":
            self.model = self._load_model(model_path)
        else:
            self.model = create_chess_transformer(model_config)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Performance tracking
        self.nodes_searched = 0
        self.cache_hits = 0
        self.evaluation_cache = {}
        
        print(f"🏆 MCTS+Transformer Player initialized")
        print(f"🧠 Model: {self.model.get_model_size()}")
        print(f"🔍 MCTS: {self.mcts_config.simulations} simulations")
        print(f"⚡ Device: {self.device}")
    
    def _load_model(self, model_path: str) -> ChessTransformer:
        """Load pre-trained transformer model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Create model and load weights
        model = create_chess_transformer("superhuman")
        model.load_state_dict(state_dict)
        
        print(f"✅ Loaded model from: {model_path}")
        return model
    
    def evaluate_position(self, board: chess.Board) -> Tuple[float, Dict[chess.Move, float]]:
        """Evaluate position with transformer and get move priors"""
        # Check cache first
        board_fen = board.fen()
        if board_fen in self.evaluation_cache:
            self.cache_hits += 1
            return self.evaluation_cache[board_fen]
        
        # Transformer evaluation - fix tensor conversion
        position_array = board_to_tensor(board)  # Returns numpy array
        position_tensor = torch.FloatTensor(position_array).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value, policy_logits = self.model(position_tensor)
        
        # Convert to scalars
        position_value = float(value.item())
        policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        
        # Map policy to legal moves
        legal_moves = list(board.legal_moves)
        move_priors = {}
        
        for move in legal_moves:
            # Simplified move encoding (can be enhanced)
            move_idx = move.from_square * 64 + move.to_square
            if move_idx < len(policy_probs):
                move_priors[move] = float(policy_probs[move_idx])
            else:
                move_priors[move] = 1e-6  # Small default value
        
        # Normalize priors
        total_prior = sum(move_priors.values())
        if total_prior > 0:
            move_priors = {move: prior / total_prior for move, prior in move_priors.items()}
        else:
            # Uniform priors if normalization fails
            uniform_prior = 1.0 / len(legal_moves)
            move_priors = {move: uniform_prior for move in legal_moves}
        
        # Cache result
        result = (position_value, move_priors)
        self.evaluation_cache[board_fen] = result
        
        return result
    
    def expand_node(self, node: MCTSNode):
        """Expand MCTS node using transformer evaluation"""
        if node.is_terminal():
            return
        
        # Get transformer evaluation
        value, move_priors = self.evaluate_position(node.board)
        
        # Store transformer evaluation
        node.transformer_value = value
        node.policy_priors = move_priors
        
        # Create children for all legal moves
        for move, prior in move_priors.items():
            if move in node.board.legal_moves:
                child_board = node.board.copy()
                child_board.push(move)
                node.children[move] = MCTSNode(child_board, node, move, prior)
        
        node.expanded = True
    
    def select_child(self, node: MCTSNode) -> MCTSNode:
        """Select best child using UCB + transformer priors"""
        best_score = -float('inf')
        best_child = None
        
        for child in node.children.values():
            score = child.get_ucb_score(self.mcts_config.c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def simulate(self, node: MCTSNode) -> float:
        """Simulation using transformer evaluation"""
        if node.is_terminal():
            # Terminal position evaluation
            result = node.board.result()
            if result == "1-0":
                return 1.0 if node.board.turn == chess.BLACK else -1.0
            elif result == "0-1":
                return -1.0 if node.board.turn == chess.BLACK else 1.0
            else:
                return 0.0
        
        # Use transformer value directly (no random rollout)
        if node.transformer_value is not None:
            return node.transformer_value
        
        # Emergency fallback
        value, _ = self.evaluate_position(node.board)
        return value
    
    def mcts_search(self, root_board: chess.Board) -> MCTSNode:
        """Enhanced MCTS search with transformer guidance"""
        root = MCTSNode(root_board)
        
        # Add Dirichlet noise to root for exploration
        self.expand_node(root)
        if root.policy_priors:
            noise = np.random.dirichlet([self.mcts_config.dirichlet_alpha] * len(root.policy_priors))
            for i, move in enumerate(root.policy_priors.keys()):
                if move in root.children:
                    original_prior = root.children[move].prior
                    root.children[move].prior = (
                        (1 - self.mcts_config.dirichlet_epsilon) * original_prior +
                        self.mcts_config.dirichlet_epsilon * noise[i]
                    )
        
        # MCTS iterations
        for simulation in range(self.mcts_config.simulations):
            node = root
            path = [node]
            
            # Selection phase
            while not node.is_leaf() and not node.is_terminal():
                node = self.select_child(node)
                if node is None:
                    break
                path.append(node)
            
            # Expansion and evaluation
            if node and not node.is_terminal():
                if not node.expanded:
                    self.expand_node(node)
                    if node.children:
                        node = next(iter(node.children.values()))
                        path.append(node)
            
            # Simulation
            if node:
                value = self.simulate(node)
                
                # Backup
                for path_node in reversed(path):
                    path_node.backup(value)
                    value = -value  # Flip perspective
            
            self.nodes_searched += 1
        
        return root
    
    def get_best_move(self, board: chess.Board, temperature: float = None) -> chess.Move:
        """Get best move using MCTS + Transformer"""
        if temperature is None:
            temperature = self.mcts_config.temperature
        
        start_time = time.time()
        
        # MCTS search
        root = self.mcts_search(board)
        
        if not root.children:
            # Fallback to random legal move
            legal_moves = list(board.legal_moves)
            return random.choice(legal_moves) if legal_moves else None
        
        # Select move based on visit counts
        if temperature == 0:
            # Deterministic selection
            best_move = max(root.children.keys(), 
                          key=lambda move: root.children[move].visits)
        else:
            # Stochastic selection with temperature
            visits = np.array([root.children[move].visits for move in root.children.keys()])
            if temperature == float('inf'):
                probs = np.ones_like(visits) / len(visits)
            else:
                probs = visits ** (1 / temperature)
                probs = probs / probs.sum()
            
            move_list = list(root.children.keys())
            best_move = np.random.choice(move_list, p=probs)
        
        search_time = time.time() - start_time
        
        # Performance info
        if root.children[best_move].visits > 0:
            win_rate = (root.children[best_move].get_value() + 1) / 2 * 100
            print(f"🔍 Move: {best_move} | Visits: {root.children[best_move].visits} | "
                  f"Win Rate: {win_rate:.1f}% | Time: {search_time:.2f}s | "
                  f"Nodes: {self.nodes_searched}")
        
        return best_move
    
    def get_move_analysis(self, board: chess.Board) -> Dict:
        """Get detailed move analysis"""
        root = self.mcts_search(board)
        
        analysis = {
            'position_value': root.get_value(),
            'nodes_searched': self.nodes_searched,
            'cache_hits': self.cache_hits,
            'top_moves': []
        }
        
        # Sort moves by visits
        sorted_moves = sorted(root.children.items(), 
                            key=lambda x: x[1].visits, reverse=True)
        
        for move, child in sorted_moves[:5]:  # Top 5 moves
            win_rate = (child.get_value() + 1) / 2 * 100
            analysis['top_moves'].append({
                'move': str(move),
                'visits': child.visits,
                'win_rate': win_rate,
                'prior': child.prior * 100
            })
        
        return analysis

def create_ultimate_player(model_path: str = None, 
                         simulations: int = 800,
                         device: str = "mps") -> MCTSTransformerPlayer:
    """Create ultimate chess player with optimal settings"""
    
    config = MCTSConfig(
        simulations=simulations,
        c_puct=1.5,
        time_limit=5.0,
        temperature=0.1  # Slightly stochastic for creativity
    )
    
    return MCTSTransformerPlayer(
        model_path=model_path,
        model_config="superhuman",
        mcts_config=config,
        device=device
    )

if __name__ == "__main__":
    # Test the ultimate player
    print("🏆 Testing MCTS + Transformer Ultimate Player")
    
    player = create_ultimate_player(simulations=100)  # Quick test
    
    board = chess.Board()
    move = player.get_best_move(board)
    analysis = player.get_move_analysis(board)
    
    print(f"✅ Best move: {move}")
    print(f"📊 Analysis: {analysis}")
    print("🚀 Ultimate Chess AI ready for superhuman performance!") 