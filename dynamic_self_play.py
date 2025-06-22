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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import multiprocessing
from tqdm import tqdm

# Add the src directory to Python path
sys.path.append('src')
from models.chess_transformer import ChessTransformer
from utils.chess_env import ChessEnvironment
from utils.config_loader import load_config
from utils.move_encoder import encode_move, decode_move, get_policy_vector_size

# --- Training Tweaks ---
RESIGN_THRESHOLD = 0.95  # Value head confidence for resign
RESIGN_STREAK = 20       # Consecutive plies before resign

# The move-limit (plies) is injected later in play loop.

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

    # Configure logging to show progress bars
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)  # Ensure output goes to stdout
        ]
    )
    logger = logging.getLogger(__name__)

    # Configure tqdm to work with logging
    tqdm.monitor_interval = 0

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

# This function must be at the top level to be pickleable by multiprocessing
def play_game_worker(worker_args: Tuple) -> Tuple:
    """
    A worker function to play a single game of chess.
    Designed to be run in a separate process. Initializes its own model and environment.
    """
    model_state_dict, config, total_games_played = worker_args
    
    # Suppress verbose logging in worker processes
    logging.basicConfig(level=logging.WARNING) 
    
    # Each worker needs its own model instance on the CPU for process safety
    device = torch.device('cpu')
    model_config = config['model']['chess_transformer']
    model = ChessTransformer(
        input_channels=model_config['input_channels'],
        cnn_channels=model_config['cnn_channels'],
        cnn_blocks=model_config['cnn_blocks'],
        transformer_layers=model_config['transformer_layers'],
        attention_heads=model_config['attention_heads'],
        policy_head_output_size=POLICY_SIZE
    )
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    mcts = MCTS(model, device)
    env = ChessEnvironment()
    board = chess.Board()
    
    board_states, policies, values = [], [], []
    
    # Temperature will be high for first 25 plies of every game, then 0.
    temp_initial = 1.0
    temp_final = 0.0

    # Early-resign tracking
    consecutive_high_value = 0
    consecutive_low_value = 0

    MAX_PLIES = 400  # 200 full moves

    while not board.is_game_over():
        policy, root_value = mcts.search(board, add_noise=True)
        
        board_states.append(env.board_to_tensor(board))
        policies.append(policy)
        values.append(0)  # placeholder

        # Early resign / early win detection
        if root_value >= RESIGN_THRESHOLD:
            consecutive_high_value += 1
            consecutive_low_value = 0
        elif root_value <= -RESIGN_THRESHOLD:
            consecutive_low_value += 1
            consecutive_high_value = 0
        else:
            consecutive_high_value = consecutive_low_value = 0

        if consecutive_high_value >= RESIGN_STREAK or consecutive_low_value >= RESIGN_STREAK:
            break  # One side clearly winning, stop the game

        # Temperature schedule by ply
        ply_count = len(board.move_stack)
        temperature = temp_initial if ply_count < 50 else temp_final

        move = AlphaZeroTrainer._select_move(policy, temperature, board)
        
        if move is None:
            break # No legal moves available
            
        board.push(move)

        # Move-limit draw
        if len(board.move_stack) >= MAX_PLIES:
            break

    result = AlphaZeroTrainer._get_game_result(board)
    
    # Backfill the final game result
    for i in range(len(values)):
        values[i] = result if (len(values) - 1 - i) % 2 == 0 else -result

    return (board_states, policies, values, result, len(board.move_stack))

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
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
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
        
        # Data Buffer
        self.replay_buffer = deque(maxlen=self.train_config['replay_buffer_size'])
        self.progress_data = {
            "iteration": 0, "total_games": 0, "total_samples": 0,
            "player_win_rate": 0.0, "avg_game_length": 0,
            "losses": {"policy": [], "value": []}
        }
        self._load_progress() # Load saved progress

    @staticmethod
    def _get_game_result(board: chess.Board) -> float:
        """
        Determines the game result from white's perspective.
        1.0 for white win, -1.0 for black win, 0.0 for draw.
        """
        if board.is_checkmate():
            return 1.0 if board.turn == chess.BLACK else -1.0
        if board.is_game_over():
            return 0.0
        return 0.0

    def _create_model(self) -> ChessTransformer:
        return ChessTransformer(
            input_channels=self.model_config['input_channels'], cnn_channels=self.model_config['cnn_channels'],
            cnn_blocks=self.model_config['cnn_blocks'], transformer_layers=self.model_config['transformer_layers'],
            attention_heads=self.model_config['attention_heads'], policy_head_output_size=POLICY_SIZE
        )

    def _load_initial_models(self):
        start_iter = self._load_checkpoint(self.player, self.optimizer, model_path=self.checkpoint_config['latest_player_model'])
        if start_iter == 0:
            logger.info("No checkpoint found at 'latest_player.pt'. Starting from scratch.")
        
        best_opponent_path = os.path.join(self.checkpoint_config['dir'], self.checkpoint_config['best_opponent_model'])
        if os.path.exists(best_opponent_path):
            self._load_checkpoint(self.opponent, model_path=best_opponent_path)
            logger.info(f"Opponent model loaded from {best_opponent_path}")
        else:
            logger.info("Initial opponent model is a copy of the player model.")
            self.opponent.load_state_dict(self.player.state_dict())
            
        return start_iter

    def _create_optimizer(self) -> optim.Optimizer:
        opt_config = CONFIG['training']['optimizer']
        return optim.AdamW(self.player.parameters(), lr=self.train_config['learning_rate'], weight_decay=opt_config['weight_decay'])

    def _create_scheduler(self):
        sched_config = CONFIG['training']['scheduler']
        return CosineAnnealingWarmRestarts(self.optimizer, T_0=sched_config['T_0'], T_mult=sched_config['T_mult'], eta_min=self.train_config['min_learning_rate'])

    def run(self):
        logger.info("--- Starting AlphaZero Training Loop ---")
        start_iter = self.progress_data.get('iteration', 0)
        logger.info(f"Resuming from iteration {start_iter + 1}")

        for iteration in range(start_iter + 1, self.train_config['iterations'] + 1):
            logger.info(f"--- Iteration {iteration}/{self.train_config['iterations']} ---")
            self.progress_data['iteration'] = iteration
            
            # Dynamic LR warm-up: ×1.5 until iter 20, then baseline
            base_lr = self.train_config['learning_rate']
            warm_factor = 1.5 if iteration <= 20 else 1.0
            for pg in self.optimizer.param_groups:
                pg['lr'] = base_lr * warm_factor

            # 1. Self-play to generate game data
            games_to_play = self.train_config['games_per_iteration']
            if len(self.replay_buffer) < self.train_config['min_replay_buffer_size']:
                fill_games = (self.train_config['min_replay_buffer_size'] - len(self.replay_buffer)) // 150 # Heuristic
                games_to_play = max(games_to_play, fill_games)
                logger.info(f"Replay buffer needs filling. Playing {games_to_play} games this iteration.")

            self.run_self_play(games_to_play)
            
            self.progress_data['total_samples'] = len(self.replay_buffer)
            logger.info(f"--- Self-play finished. Replay buffer size: {len(self.replay_buffer)} ---")
            
            # 2. Training
            self.run_training()
            
            # 3. Evaluation
            if iteration % self.train_config['evaluation_interval'] == 0:
                self.run_evaluation()

            # 4. Checkpointing
            if iteration % self.checkpoint_config['save_interval'] == 0:
                self._save_checkpoint(iteration, self.player, self.optimizer, is_latest=True, is_iter_specific=True)
            else:
                self._save_checkpoint(iteration, self.player, self.optimizer, is_latest=True)
                
            self._save_progress()

        logger.info("--- AlphaZero Training Loop Finished ---")

    def run_self_play(self, num_games: int):
        num_workers = self.sys_config.get('self_play_workers', 1)
        logger.info(f"Starting self-play with {num_workers} parallel workers for {num_games} games.")
        
        self.player.cpu()
        model_state_dict = self.player.state_dict()
        self.player.to(self.device)

        # Prepare arguments for each worker
        worker_args = [(model_state_dict, CONFIG, self.progress_data['total_games'] + i) for i in range(num_games)]

        total_moves = 0
        game_results = {'wins': 0, 'losses': 0, 'draws': 0}
        start_time = time.time()

        with multiprocessing.Pool(processes=num_workers) as pool:
            with tqdm(total=num_games, desc="Self-Play Games", ncols=100, position=0, leave=True) as pbar:
                for (board_states, policies, values, result, game_move_count) in pool.imap_unordered(play_game_worker, worker_args):
                    self.replay_buffer.extend(zip(board_states, policies, values))
                    self.progress_data['total_games'] += 1
                    total_moves += game_move_count
                    
                    if result == 1.0: game_results['wins'] += 1
                    elif result == -1.0: game_results['losses'] += 1
                    else: game_results['draws'] += 1
                    
                    # Update progress bar with current stats
                    elapsed_time = time.time() - start_time
                    games_per_second = self.progress_data['total_games'] / elapsed_time
                    pbar.set_postfix({
                        'W/L/D': f"{game_results['wins']}/{game_results['losses']}/{game_results['draws']}",
                        'Games/s': f"{games_per_second:.2f}",
                        'Avg Moves': f"{total_moves/self.progress_data['total_games']:.1f}"
                    })
                    pbar.update(1)

        avg_game_length = total_moves / num_games if num_games > 0 else 0
        self.progress_data['avg_game_length'] = avg_game_length
        total_time = time.time() - start_time
        logger.info(f"Self-play stats: {game_results['wins']} Wins, {game_results['losses']} Losses, {game_results['draws']} Draws.")
        logger.info(f"Average game length: {avg_game_length:.1f} moves. Total time: {total_time:.1f}s ({num_games/total_time:.2f} games/s)")

    @staticmethod
    def _select_move(policy: np.ndarray, temperature: float, board: chess.Board) -> chess.Move:
        """Selects a move based on the policy and temperature, ensuring it's legal."""
        legal_moves_map = {encode_move(move): move for move in board.legal_moves if encode_move(move) is not None}
        if not legal_moves_map:
            return random.choice(list(board.legal_moves)) if list(board.legal_moves) else None

        legal_indices = list(legal_moves_map.keys())
        legal_policy = policy[legal_indices]
        
        if np.sum(legal_policy) == 0:
            return legal_moves_map.get(random.choice(legal_indices))

        if temperature == 0:
            move_idx = legal_indices[np.argmax(legal_policy)]
        else:
            distribution = np.power(legal_policy, 1.0 / temperature)
            distribution /= np.sum(distribution)
            move_idx = np.random.choice(legal_indices, p=distribution)
        
        return legal_moves_map.get(move_idx)
    
    def run_training(self):
        if len(self.replay_buffer) < self.train_config['min_replay_buffer_size']:
            logger.info(f"Skipping training: replay buffer has {len(self.replay_buffer)} samples, but {self.train_config['min_replay_buffer_size']} are required.")
            return

        logger.info("--- Training model... ---")
        self.player.train()

        total_policy_loss = 0
        total_value_loss = 0
        batches = 0
        start_time = time.time()

        for epoch in range(self.train_config['epochs_per_iteration']):
            epoch_start = time.time()
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_batches = 0

            training_data = random.sample(self.replay_buffer, k=min(len(self.replay_buffer), self.train_config['batch_size'] * 256))
            data_loader = torch.utils.data.DataLoader(training_data, batch_size=self.train_config['batch_size'], shuffle=True)

            with tqdm(total=len(data_loader), desc=f"Epoch {epoch + 1}/{self.train_config['epochs_per_iteration']}", ncols=100, position=0, leave=True) as pbar:
                for batch_idx, (board_states, policies, values) in enumerate(data_loader):
                    board_states = board_states.to(self.device)
                    policies = policies.to(self.device)
                    values = values.to(self.device, dtype=torch.float32)

                    self.optimizer.zero_grad()
                    
                    with torch.cuda.amp.autocast(enabled=self.sys_config.get('use_amp', False)):
                        pred_values, pred_policies = self.player(board_states)
                        policy_loss = F.cross_entropy(pred_policies, policies)
                        value_loss = F.mse_loss(pred_values.squeeze(), values)
                        loss = policy_loss + value_loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.player.parameters(), self.train_config['gradient_clip'])
                    self.optimizer.step()

                    epoch_policy_loss += policy_loss.item()
                    epoch_value_loss += value_loss.item()
                    epoch_batches += 1

                    # Update progress bar with current stats
                    pbar.set_postfix({
                        'Policy Loss': f"{policy_loss.item():.4f}",
                        'Value Loss': f"{value_loss.item():.4f}",
                        'Total Loss': f"{loss.item():.4f}"
                    })
                    pbar.update(1)

            epoch_time = time.time() - epoch_start
            logger.info(f"    Epoch {epoch + 1}/{self.train_config['epochs_per_iteration']}: "
                       f"Policy Loss = {epoch_policy_loss/epoch_batches:.4f}, "
                       f"Value Loss = {epoch_value_loss/epoch_batches:.4f}, "
                       f"Time = {epoch_time:.1f}s")

            total_policy_loss += epoch_policy_loss
            total_value_loss += epoch_value_loss
            batches += epoch_batches

        training_time = time.time() - start_time
        logger.info(f"--- Training finished. Avg Policy Loss: {total_policy_loss/batches:.4f}, "
                   f"Avg Value Loss: {total_value_loss/batches:.4f}, "
                   f"Total Time: {training_time:.1f}s ---")

    def run_evaluation(self):
        logger.info("--- Evaluating model against opponent... ---")
        start_time = time.time()
        
        num_workers = self.sys_config.get('evaluation_workers', 1)
        logger.info(f"Starting evaluation with {num_workers} parallel workers for {self.train_config['evaluation_games']} games.")
        
        self.player.cpu()
        self.opponent.cpu()
        player_state_dict = self.player.state_dict()
        opponent_state_dict = self.opponent.state_dict()
        self.player.to(self.device)
        self.opponent.to(self.device)

        # Prepare arguments for each worker
        worker_args = [(player_state_dict, opponent_state_dict, CONFIG, i) 
                      for i in range(self.train_config['evaluation_games'])]

        wins = losses = draws = 0
        total_moves = 0

        with multiprocessing.Pool(processes=num_workers) as pool:
            with tqdm(total=self.train_config['evaluation_games'], desc="Evaluation Games", ncols=100) as pbar:
                for (result, move_count) in pool.imap_unordered(evaluate_game_worker, worker_args):
                    if result == 1.0: wins += 1
                    elif result == -1.0: losses += 1
                    else: draws += 1
                    total_moves += move_count
                    pbar.update(1)

        win_rate = (wins + 0.5 * draws) / self.train_config['evaluation_games']
        avg_game_length = total_moves / self.train_config['evaluation_games']
        eval_time = time.time() - start_time

        logger.info(f"--- Evaluation finished. Win rate: {win_rate:.2f} ({wins}W/{losses}L/{draws}D) ---")
        logger.info(f"    Average game length: {avg_game_length:.1f} moves")
        logger.info(f"    Evaluation time: {eval_time:.1f}s")

    def _save_checkpoint(self, iteration: int, model: torch.nn.Module, optimizer: optim.Optimizer = None, name: str = None, is_latest: bool = False, is_iter_specific: bool = False):
        checkpoint_dir = self.checkpoint_config['dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        state = {
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        }

        paths_to_save = []
        if name:
             paths_to_save.append(os.path.join(checkpoint_dir, name))
        if is_iter_specific:
            paths_to_save.append(os.path.join(checkpoint_dir, f"checkpoint_iter_{iteration}.pt"))
        
        if not paths_to_save and not is_latest:
            logger.warning("Save checkpoint called but no name or flags provided. Nothing will be saved.")
            return

        for path in paths_to_save:
            try:
                torch.save(state, path)
                logger.info(f"Checkpoint saved to {path}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint to {path}: {e}")
        
        if is_latest:
            latest_path = os.path.join(checkpoint_dir, self.checkpoint_config['latest_player_model'])
            try:
                torch.save(state, latest_path)
                logger.info(f"Updated latest player model checkpoint: {latest_path}")
            except Exception as e:
                logger.error(f"Failed to save latest checkpoint to {latest_path}: {e}")

    def _load_checkpoint(self, model: torch.nn.Module, optimizer: optim.Optimizer = None, model_path: str = None) -> int:
        if not model_path:
            logger.warning("Load checkpoint called with no model_path.")
            return 0
            
        full_path = os.path.join(self.checkpoint_config['dir'], model_path)
        if not os.path.exists(full_path):
            return 0
        
        try:
            checkpoint = torch.load(full_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            start_iter = checkpoint.get('iteration', 0)
            logger.info(f"Checkpoint loaded from {full_path}. Resuming from iteration {start_iter + 1}.")
            return start_iter
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {full_path}: {e}")
            return 0

    def _save_progress(self):
        progress_file = os.path.join(CONFIG['logging']['log_dir'], 'alpha_zero_training', 'training_progress.json')
        try:
            with open(progress_file, 'w') as f:
                json.dump(self.progress_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress to {progress_file}: {e}")

    def _load_progress(self):
        progress_file = os.path.join(CONFIG['logging']['log_dir'], 'alpha_zero_training', 'training_progress.json')
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    self.progress_data = json.load(f)
                    logger.info(f"Resumed progress from {progress_file}")
            except Exception as e:
                logger.error(f"Failed to load progress from {progress_file}: {e}")


def evaluate_game_worker(args):
    player_state_dict, opponent_state_dict, config, game_id = args
    
    # Create models on CPU
    player = ChessTransformer(**config['model']['chess_transformer'])
    opponent = ChessTransformer(**config['model']['chess_transformer'])
    
    # Load state dicts
    player.load_state_dict(player_state_dict)
    opponent.load_state_dict(opponent_state_dict)
    
    # Create MCTS players
    mcts_player = MCTS(player, 'cpu')
    mcts_opponent = MCTS(opponent, 'cpu')
    
    # Play game
    board = chess.Board()
    player_is_white = game_id % 2 == 0
    move_count = 0
    
    while not board.is_game_over():
        if (board.turn == chess.WHITE and player_is_white) or \
           (board.turn == chess.BLACK and not player_is_white):
            policy, _ = mcts_player.search(board, add_noise=False)
            move = select_move(policy, temperature=0, board=board)  # Greedy
        else:
            policy, _ = mcts_opponent.search(board, add_noise=False)
            move = select_move(policy, temperature=0, board=board)
        
        if move is None:
            break
            
        board.push(move)
        move_count += 1
    
    # Get result
    if board.is_checkmate():
        result = 1.0 if (board.turn == chess.BLACK and player_is_white) or \
                        (board.turn == chess.WHITE and not player_is_white) else -1.0
    else:
        result = 0.0  # Draw
    
    return result, move_count

def main():
    """Main entry point."""
    trainer = AlphaZeroTrainer()
    trainer.run()

if __name__ == "__main__":
    # It's good practice to set the start method for multiprocessing, especially on macOS/Windows
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # This will be raised if the start method has already been set.
        pass
    main() 
