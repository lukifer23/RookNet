#!/usr/bin/env python3
"""
Dynamic Self-Play Trainer (AlphaZero-style)
Implements a true reinforcement learning loop for chess model training.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import chess
import numpy as np
import random
import os
import time
import math
import logging
import copy
import sys
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict
import json
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add the src directory to Python path
sys.path.append('src')
from models.chess_transformer import ChessTransformer
from utils.chess_env import ChessEnvironment
from utils.config_loader import load_config
from utils.move_encoder import encode_move, decode_move, get_policy_vector_size

# --- Pre-computation and Setup ---
# Wrap initial setup in a try-catch to ensure any errors are visible
try:
    # Load unified configuration
    CONFIG = load_config("configs/config.v2.yaml")

    # Setup logging
    log_config = CONFIG['logging']
    log_dir = os.path.join(log_config['log_dir'], 'alpha_zero_training')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_run_{int(time.time())}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    # Pre-calculate policy size
    POLICY_SIZE = get_policy_vector_size()

except Exception as e:
    print(f"FATAL: A critical error occurred during initial setup: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- MCTS Node and Search ---

@dataclass
class MCTSNode:
    """A node in the Monte Carlo Tree Search."""
    parent: 'MCTSNode' = None
    move: chess.Move = None
    prior: float = 0.0
    children: Dict[chess.Move, 'MCTSNode'] = None
    visits: int = 0
    value_sum: float = 0.0
    _is_expanded = False

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def is_expanded(self) -> bool:
        return self._is_expanded

    def mark_as_expanded(self):
        self._is_expanded = True
    
    def uct_value(self, parent_visits: int, c_puct: float) -> float:
        """Calculates the UCT value for this node."""
        q_value = -self.mean_value # From the perspective of the node to be chosen
        u_value = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        return q_value + u_value

class MCTS:
    """Monte Carlo Tree Search implementation."""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.env = ChessEnvironment()
        self.policy_size = POLICY_SIZE
        self.mcts_config = CONFIG['training']['mcts']
        self.c_puct = self.mcts_config['c_puct']

    def search(self, board: chess.Board, add_noise=False) -> Tuple[np.ndarray, float]:
        """
        Performs MCTS from the root node (current board state).
        Returns the improved policy and the value of the root state.
        """
        root = MCTSNode()

        # Expansion of the root node
        value, policy_logits = self._evaluate(board)
        self._expand_node(root, board, policy_logits)

        if add_noise:
            self._add_dirichlet_noise(root, board)

        for _ in range(self.mcts_config['simulations']):
            self._simulate(root, board.copy())

        # Extract policy and value
        policy = np.zeros(self.policy_size, dtype=np.float32)
        if root.is_expanded():
            for move, child in root.children.items():
                idx = encode_move(move)
                if idx is not None:
                    policy[idx] = child.visits
            
            policy_sum = np.sum(policy)
            if policy_sum > 0:
                policy /= policy_sum
            else:
                 # This can happen if all legal moves are illegal in our encoding, which is rare but possible.
                 # Fallback to a uniform policy over legal moves.
                 logger.warning("MCTS search resulted in a zero policy vector. Falling back to uniform.")
                 num_legal_moves = board.legal_moves.count()
                 if num_legal_moves > 0:
                    for move in board.legal_moves:
                        idx = encode_move(move)
                        if idx is not None:
                            policy[idx] = 1.0 / num_legal_moves

        return policy, value # Return the direct value from the root's first evaluation

    def _simulate(self, node: MCTSNode, board: chess.Board):
        # Selection
        while node.is_expanded():
            if not node.children: # Terminal node in tree search
                break
            parent_visits = node.visits
            best_move = max(node.children, key=lambda m: node.children[m].uct_value(parent_visits, self.c_puct))
            node = node.children[best_move]
            board.push(best_move)

        # Expansion and Evaluation
        outcome = None
        if board.is_game_over():
            outcome = AlphaZeroTrainer._get_game_result(board)
            value = outcome
        else:
            value, policy_logits = self._evaluate(board)
            self._expand_node(node, board, policy_logits)

        # Backpropagation
        # If the game ended, we propagate the true outcome. Otherwise, we use the network's value estimate.
        current_value = outcome if outcome is not None else value
        while node is not None:
            node.visits += 1
            node.value_sum += current_value
            current_value = -current_value # The value is from the perspective of the other player
            node = node.parent

    def _expand_node(self, node: MCTSNode, board: chess.Board, policy_logits: torch.Tensor):
        node.children = {}
        policy = F.softmax(policy_logits, dim=0).cpu().numpy()
        
        for move in board.legal_moves:
            idx = encode_move(move)
            if idx is not None and idx < len(policy):
                move_prior = policy[idx]
                node.children[move] = MCTSNode(parent=node, move=move, prior=move_prior)
        node.mark_as_expanded()

    def _evaluate(self, board: chess.Board) -> Tuple[float, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            board_numpy = self.env.board_to_tensor(board)
            board_tensor = torch.from_numpy(board_numpy).float().unsqueeze(0).to(self.device)
            value, policy_logits = self.model(board_tensor)
        return value.item(), policy_logits.squeeze(0)
    
    def _add_dirichlet_noise(self, node: MCTSNode, board: chess.Board):
        """Adds Dirichlet noise to the root node's priors for exploration."""
        if not node.is_expanded():
             return

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return
            
        noise = np.random.dirichlet([self.mcts_config['dirichlet_alpha']] * len(legal_moves))
        
        for i, move in enumerate(legal_moves):
            if move in node.children:
                child_prior = node.children[move].prior
                node.children[move].prior = (1 - self.mcts_config['dirichlet_epsilon']) * child_prior + self.mcts_config['dirichlet_epsilon'] * noise[i]

# --- Training Loop Orchestrator ---

class AlphaZeroTrainer:
    def __init__(self):
        # Configs
        self.sys_config = CONFIG['system']
        self.train_config = CONFIG['training']['alpha_zero']
        self.model_config = CONFIG['model']['chess_transformer']
        self.checkpoint_config = CONFIG['training']['checkpoints']

        # Device
        self.device = torch.device(self.sys_config['device'] if self.sys_config['device'] != 'auto' else ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'))
        logger.info(f"Using device: {self.device}")

        # Models
        self.player = self._create_model().to(self.device)
        self.opponent = self._create_model().to(self.device)
        self._load_initial_models()
        self.opponent.eval()

        # Compile model for performance
        if self.sys_config.get('compile_model', False):
            logger.info("Compiling the player model with torch.compile()...")
            try:
                self.player = torch.compile(self.player, mode="max-autotune")
                logger.info("Model compiled successfully.")
            except Exception as e:
                logger.warning(f"torch.compile() failed: {e}. Continuing without compilation.")

        # MCTS and Optimizer
        self.mcts = MCTS(self.player, self.device)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Data Buffer
        self.replay_buffer = deque(maxlen=self.train_config['replay_buffer_size'])
        self.progress_data = {
            "iteration": 0, "total_games": 0, "total_samples": 0,
            "player_win_rate": 0.0, "avg_game_length": 0,
            "losses": {"policy": [], "value": []}
        }

    @staticmethod
    def _get_game_result(board: chess.Board) -> float:
        """
        Determines the game result from white's perspective.
        1.0 for white win, -1.0 for black win, 0.0 for draw.
        """
        if board.is_checkmate():
            # If black is to move, white delivered checkmate.
            return 1.0 if board.turn == chess.BLACK else -1.0
        # Any other game-ending condition is a draw
        if board.is_game_over():
            return 0.0
        # Should not be reached, but as a safeguard
        return 0.0

    def _create_model(self) -> ChessTransformer:
        # Pass the policy head output size from the move encoder
        return ChessTransformer(
            input_channels=self.model_config['input_channels'],
            cnn_channels=self.model_config['cnn_channels'],
            cnn_blocks=self.model_config['cnn_blocks'],
            transformer_layers=self.model_config['transformer_layers'],
            attention_heads=self.model_config['attention_heads'],
            policy_head_output_size=POLICY_SIZE
        )

    def _load_initial_models(self):
        self.start_iteration = self._load_checkpoint(self.player, self.optimizer, self.checkpoint_config['latest_player_model'])
        
        opponent_path = self.checkpoint_config.get('initial_opponent_model', self.checkpoint_config['latest_player_model'])
        self._load_checkpoint(self.opponent, model_path=opponent_path)
        logger.info(f"Opponent model loaded from {opponent_path}")
        
        self.opponent.load_state_dict(self.player.state_dict())
        logger.info("Initial opponent model is a copy of the player model.")

    def _create_optimizer(self) -> optim.Optimizer:
        return optim.AdamW(
            self.player.parameters(),
            lr=self.train_config['learning_rate'],
            weight_decay=self.train_config['weight_decay']
        )
    
    def _create_scheduler(self):
        return CosineAnnealingLR(self.optimizer, T_max=self.train_config['lr_decay_steps'], eta_min=self.train_config['min_learning_rate'])

    def run(self):
        logger.info("--- Starting AlphaZero Training Loop ---")
        logger.info(f"Configuration: {json.dumps(CONFIG, indent=2)}")

        for i in range(self.start_iteration, self.train_config['iterations']):
            self.progress_data['iteration'] = i
            logger.info(f"--- Iteration {i+1}/{self.train_config['iterations']} ---")
            
            # 1. Self-Play
            self.run_self_play()
            
            # 2. Training
            if len(self.replay_buffer) >= self.train_config['min_replay_buffer_size']:
                self.run_training()
                
                # 3. Evaluation and Opponent Update
                if (i + 1) % self.train_config['evaluation_interval'] == 0:
                    self.run_evaluation()
            else:
                logger.info(f"Replay buffer size ({len(self.replay_buffer)}) is less than minimum ({self.train_config['min_replay_buffer_size']}). Skipping training and evaluation.")

            # 4. Save Checkpoint
            if (i + 1) % self.checkpoint_config['save_interval'] == 0:
                self._save_checkpoint(i + 1, self.player, self.optimizer, self.checkpoint_config['latest_player_model'])

            self._save_progress()

        logger.info("--- AlphaZero Training Loop Finished ---")

    def run_self_play(self):
        logger.info("--- Generating self-play games... ---")
        self.player.eval()
        game_histories = []

        for game_num in range(self.train_config['games_per_iteration']):
            start_time = time.time()
            board = chess.Board()
            game_data = [] # (board_tensor, policy_vector, outcome)
            
            while not board.is_game_over():
                # Temperature scheduling
                temperature = self._get_temperature(self.progress_data['total_games'])

                # Run MCTS search
                policy, _ = self.mcts.search(board, add_noise=True)

                # Select move
                move = self._select_move(policy, temperature, board)
                
                # Store data
                board_tensor = ChessEnvironment().board_to_tensor(board)
                # Store the board state *before* the move, along with whose turn it was
                game_data.append({'tensor': board_tensor, 'policy': policy, 'turn': board.turn})

                board.push(move)

            # Game finished, determine result and backpropagate
            result = self._get_game_result(board)
            final_game_data = []
            for i, data in enumerate(game_data):
                # The result is from white's perspective.
                # If it was white's turn, the value is the result.
                # If it was black's turn, the value is the inverted result.
                outcome = result if data['turn'] == chess.WHITE else -result
                final_game_data.append((data['tensor'], data['policy'], outcome))
            
            game_histories.extend(final_game_data)
            self.progress_data['total_games'] += 1
            self.progress_data['avg_game_length'] = (self.progress_data['avg_game_length'] * (self.progress_data['total_games'] - 1) + len(board.move_stack)) / self.progress_data['total_games']
            
            duration = time.time() - start_time
            logger.info(f"Self-play game {game_num+1}/{self.train_config['games_per_iteration']} finished in {duration:.2f}s. Result: {result}, Moves: {len(board.move_stack)}")

        self.replay_buffer.extend(game_histories)
        self.progress_data['total_samples'] = len(self.replay_buffer)
        logger.info(f"--- Self-play finished. Replay buffer size: {len(self.replay_buffer)} ---")
    
    def _get_temperature(self, total_games: int) -> float:
        """
        Decay temperature over the course of training.
        Starts high for exploration, lowers for exploitation.
        """
        temp_config = self.train_config['temperature']
        decay_point = temp_config['decay_after_games']
        if total_games < decay_point:
            return temp_config['initial']
        else:
            return temp_config['final']

    def _select_move(self, policy: np.ndarray, temperature: float, board: chess.Board) -> chess.Move:
        """Selects a move based on the policy and temperature."""
        if temperature == 0: # Greedy selection
            move_idx = np.argmax(policy)
        else:
            # Apply temperature to the policy
            policy_temp = np.power(policy, 1.0 / temperature)
            policy_temp /= np.sum(policy_temp)
            move_idx = np.random.choice(len(policy), p=policy_temp)
        
        move = decode_move(move_idx)
        
        # Fallback if the decoded move is not legal
        if move is None or move not in board.legal_moves:
            logger.warning(f"Decoded move {move} (index {move_idx}) is not legal. Choosing a random legal move.")
            return random.choice(list(board.legal_moves))
            
        return move

    def run_training(self):
        logger.info("--- Training model... ---")
        self.player.train()
        
        total_policy_loss = 0
        total_value_loss = 0

        for i in range(self.train_config['epochs_per_iteration']):
            # Sample a batch from the replay buffer
            if len(self.replay_buffer) < self.train_config['batch_size']:
                logger.warning(f"Replay buffer ({len(self.replay_buffer)}) smaller than batch size ({self.train_config['batch_size']}). Skipping epoch.")
                continue

            batch = random.sample(self.replay_buffer, self.train_config['batch_size'])
            
            board_tensors, policy_targets, value_targets = zip(*batch)
            
            board_tensors = torch.from_numpy(np.array(board_tensors)).float().to(self.device)
            policy_targets = torch.from_numpy(np.array(policy_targets)).float().to(self.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float).view(-1, 1).to(self.device)

            self.optimizer.zero_grad()
            
            # Forward pass
            pred_values, pred_policy_logits = self.player(board_tensors)
            
            # Calculate loss
            policy_loss = F.cross_entropy(pred_policy_logits, policy_targets)
            value_loss = F.mse_loss(pred_values, value_targets)
            loss = policy_loss + value_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        self.scheduler.step() # Step scheduler once per training iteration (which contains multiple epochs/steps)
        
        num_epochs = self.train_config['epochs_per_iteration']
        if num_epochs > 0:
            avg_policy_loss = total_policy_loss / num_epochs
            avg_value_loss = total_value_loss / num_epochs
            self.progress_data['losses']['policy'].append(avg_policy_loss)
            self.progress_data['losses']['value'].append(avg_value_loss)
            logger.info(f"--- Training finished. Avg Policy Loss: {avg_policy_loss:.4f}, Avg Value Loss: {avg_value_loss:.4f} ---")
        else:
            logger.info("--- No training epochs were run. ---")

    def run_evaluation(self):
        logger.info("--- Evaluating model against opponent... ---")
        self.player.eval()
        
        wins = 0
        draws = 0
        losses = 0
        
        for i in range(self.train_config['evaluation_games']):
            player_is_white = i % 2 == 0
            result = self._play_eval_game(player_is_white)
            
            if (player_is_white and result == 1.0) or (not player_is_white and result == -1.0):
                wins += 1
            elif result == 0.0:
                draws += 1
            else:
                losses += 1

        win_rate = wins / self.train_config['evaluation_games']
        self.progress_data['player_win_rate'] = win_rate
        logger.info(f"--- Evaluation finished. Win rate: {win_rate:.2f} ({wins}W/{losses}L/{draws}D) ---")
        
        # Update opponent if player is clearly better
        if win_rate >= self.train_config['opponent_update_threshold']:
            logger.info(f"Player win rate ({win_rate:.2f}) exceeds threshold ({self.train_config['opponent_update_threshold']}). Updating opponent model.")
            self.opponent.load_state_dict(self.player.state_dict())
            self._save_checkpoint(self.progress_data['iteration'], self.opponent, name=self.checkpoint_config['best_opponent_model'])
        else:
            logger.info("Player performance did not meet threshold. Opponent model not updated.")

    def _play_eval_game(self, player_is_white: bool) -> float:
        board = chess.Board()
        eval_mcts = MCTS(self.player, self.device)
        opponent_mcts = MCTS(self.opponent, self.device)

        while not board.is_game_over():
            if (board.turn == chess.WHITE and player_is_white) or \
               (board.turn == chess.BLACK and not player_is_white):
                # Player's turn
                policy, _ = eval_mcts.search(board.copy(), add_noise=False)
                move = self._select_move(policy, temperature=0, board=board) # Greedy
            else:
                # Opponent's turn
                policy, _ = opponent_mcts.search(board.copy(), add_noise=False)
                move = self._select_move(policy, temperature=0, board=board) # Greedy
            
            board.push(move)
        
        return self._get_game_result(board)

    def _save_checkpoint(self, iteration: int, model: torch.nn.Module, optimizer: optim.Optimizer = None, name: str = "checkpoint.pt"):
        """Saves a training checkpoint."""
        os.makedirs(self.checkpoint_config['dir'], exist_ok=True)
        path = os.path.join(self.checkpoint_config['dir'], f"{iteration}_{name}")
        
        save_dict = {
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
        }
        if optimizer:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(save_dict, path)
        logger.info(f"Checkpoint saved to {path}")

        # Update latest model symlink/file
        latest_path = os.path.join(self.checkpoint_config['dir'], name)
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(os.path.basename(path), latest_path) # Use relative symlink
        logger.info(f"Updated '{name}' to point to {os.path.basename(path)}")

    def _load_checkpoint(self, model: torch.nn.Module, optimizer: optim.Optimizer = None, model_path: str = None) -> int:
        """Loads a training checkpoint."""
        if not model_path or not os.path.exists(model_path):
            logger.info("No checkpoint found. Starting from scratch.")
            return 0
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handling compiled models
            model_state_dict = checkpoint['model_state_dict']
            if hasattr(model, '_orig_mod'):
                model._orig_mod.load_state_dict(model_state_dict)
            else:
                model.load_state_dict(model_state_dict)

            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            iteration = checkpoint.get('iteration', 0)
            logger.info(f"Successfully loaded checkpoint from {model_path} at iteration {iteration}.")
            return iteration

        except Exception as e:
            logger.error(f"Error loading checkpoint from {model_path}: {e}. Starting from scratch.", exc_info=True)
            return 0

    def _save_progress(self):
        """Saves training progress to a JSON file."""
        progress_file = os.path.join(log_dir, 'training_progress.json')
        with open(progress_file, 'w') as f:
            # Convert numpy floats to native python floats for json serialization
            serializable_progress = copy.deepcopy(self.progress_data)
            if 'losses' in serializable_progress:
                serializable_progress['losses']['policy'] = [float(x) for x in serializable_progress['losses']['policy']]
                serializable_progress['losses']['value'] = [float(x) for x in serializable_progress['losses']['value']]
            json.dump(serializable_progress, f, indent=4)
        logger.info(f"Training progress saved to {progress_file}")


def main():
    trainer = AlphaZeroTrainer()
    trainer.run() 