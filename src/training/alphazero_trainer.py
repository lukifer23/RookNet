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
import multiprocessing as mp
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import webdataset as wds
import warnings

# GPU inference server
from multiprocessing import Process, Manager
from engine.gpu_inference_server import run_inference_server

# Ensure the 'src' directory is in the Python path
# (Removed obsolete sys.path manipulation after package installation)

from utils.replay_buffer import StreamingReplayBuffer
from models.chess_transformer import ChessTransformer
from utils.chess_env import ChessEnvironment
from utils.config_loader import load_config
from utils.move_encoder import encode_move, decode_move, get_policy_vector_size
from search.mcts import MCTS
from utils.board_utils import board_to_tensor

# --- Game Outcome Constants ---
WIN = 1.0
LOSS = -1.0
DRAW = 0.0

# ------------------------------------------------------------------
# Silence noisy torch compile warnings globally
# ------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
for noisy in ("torch._dynamo", "torch._inductor"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

# --- Worker Management ---
def game_worker_manager(worker_id: int, config_path: str, model_state_dict: Dict, request_q, result_q, game_data_queue: mp.Queue, num_games_to_play: int, opponent_state_dict: Dict = None):
    """A manager function that runs on a worker process and plays multiple games."""
    for _ in range(num_games_to_play):
        if opponent_state_dict:
            # Evaluation game
            play_eval_game_worker(worker_id, config_path, request_q, result_q, game_data_queue, model_state_dict, opponent_state_dict)
        else:
            # Self-play game
            play_game_worker(worker_id, config_path, request_q, result_q, game_data_queue, model_state_dict)

# --- Worker for Evaluation ---
def play_eval_game_worker(worker_id: int, config_path: str, request_q, result_q, game_data_queue: mp.Queue, model_state_dict: Dict, opponent_state_dict: Dict):
    """
    Worker process for playing a single evaluation game.
    It loads the models, plays one game, and puts the result in the queue.
    """
    try:
        # --- Config and Environment ---
        config = load_config(config_path)
        env = ChessEnvironment(config)

        # --- Model Loading ---
        class RemoteModel:
            def __init__(self, rq, rs):
                self.rq, self.rs = rq, rs
            def __call__(self, board_tensor):
                uid = os.urandom(16)
                self.rq.put((uid, board_tensor.squeeze(0).cpu().numpy()))
                while True:
                    try:
                        resp_uid, policy_np, value = self.rs.get(timeout=120)
                    except Exception:
                        raise RuntimeError("Inference server timeout after 120s; check GPU server logs")
                    if resp_uid == uid:
                        policy = torch.from_numpy(policy_np).float()
                        val = torch.tensor(value, dtype=torch.float32).unsqueeze(0)
                        return val, policy

        player_model = RemoteModel(request_q, result_q)
        opponent_model = RemoteModel(request_q, result_q)

        mcts_player = MCTS(model=player_model, config=config, device='cpu')
        mcts_opponent = MCTS(model=opponent_model, config=config, device='cpu')

        board = chess.Board()
        player_is_white = worker_id % 2 == 0

        while not board.is_game_over(claim_draw=True):
            is_player_turn = (board.turn == chess.WHITE and player_is_white) or \
                             (board.turn == chess.BLACK and not player_is_white)

            if is_player_turn:
                move = mcts_player.search(board, simulations=config['training']['mcts']['simulations'])
            else:
                move = mcts_opponent.search(board, simulations=config['training']['mcts']['simulations'])
            
            if move is None: break
            board.push(move)
        
        # Result from the player's perspective
        result_str = board.result()
        if result_str == "1-0":
            player_result = WIN if player_is_white else LOSS
        elif result_str == "0-1":
            player_result = LOSS if player_is_white else WIN
        else:
            player_result = DRAW
        
        game_data_queue.put(player_result)

    except Exception as e:
        logging.error(f"Error in eval worker {worker_id}: {e}", exc_info=True)
        game_data_queue.put(None)

# --- Worker for Self-Play ---

@dataclass
class GamePlayResult:
    """Stores the results and data from a single game."""
    board_states: List[torch.Tensor]
    policies: List[np.ndarray]
    values: List[float]
    result: float
    moves: int

def play_game_worker(worker_id: int, config_path: str, request_q, result_q, game_data_queue: mp.Queue, model_state_dict: Dict):
    """
    A worker process that plays a specified number of games of self-play.
    """
    try:
        # --- Config and Environment ---
        config = load_config(config_path)
        env = ChessEnvironment(config)
        desired_device = config['system']['device']
        if desired_device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        # RemoteModel wraps queue-based inference so MCTS API stays unchanged
        class RemoteModel:
            def __init__(self, rq, rs):
                self.rq, self.rs = rq, rs

            def __call__(self, board_tensor):
                # returns (policy, value) like original model.forward
                uid = os.urandom(16)
                self.rq.put((uid, board_tensor.squeeze(0).cpu().numpy()))
                while True:
                    try:
                        resp_uid, policy_np, value = self.rs.get(timeout=120)
                    except Exception:
                        raise RuntimeError("Inference server timeout after 120s; check GPU server logs")
                    if resp_uid == uid:
                        policy = torch.from_numpy(policy_np).float()
                        val = torch.tensor(value, dtype=torch.float32).unsqueeze(0)
                        return val, policy

        remote_model = RemoteModel(request_q, result_q)

        mcts = MCTS(model=remote_model, config=config, device='cpu')
        
        board = chess.Board()
        
        # --- Game Loop ---
        board_states, policies, values_ph = [], [], [] # values_ph is a placeholder
        
        temp_config = config['training']['alpha_zero']['temperature']
        
        while not board.is_game_over(claim_draw=True):
            # The number of simulations is now read from the config within MCTS
            move = mcts.search(board)

            # Get the training policy from the MCTS root node
            training_policy = np.zeros(get_policy_vector_size(), dtype=np.float32)
            if mcts.root and mcts.root.children:
                total_visits = sum(child.visit_count for child in mcts.root.children.values())
                if total_visits > 0:
                    for m, child in mcts.root.children.items():
                        training_policy[encode_move(m)] = child.visit_count / total_visits

            board_states.append(board_to_tensor(board))
            policies.append(training_policy)
            values_ph.append(0) # Placeholder, will be backfilled

            if move is None: break
            board.push(move)

        # --- Game Finalization ---
        result_str = board.result()
        if result_str == "1-0":
            result = WIN
        elif result_str == "0-1":
            result = LOSS
        else:
            result = DRAW
        
        # Backfill the final game result into the value placeholders
        final_values = []
        # The board stores plies, not full moves. A ply is one player's move.
        num_plies = len(board.move_stack)
        for i in range(len(values_ph)):
            # Determine whose move it was at state `i`.
            # If the total number of moves made (num_plies) minus the index `i`
            # has the same parity as the final board turn, then the player to move at state `i`
            # is the same as the player who made the final move.
            # This is complex. A simpler way: the value is from the perspective of the *current player* at that state.
            # White's perspective: 1 is a win.
            # If black wins (result=-1), states where it was white's turn get -1.
            # If white wins (result=1), states where it was white's turn get 1.
            is_white_turn_at_state_i = (num_plies - i) % 2 == (0 if board.turn == chess.BLACK else 1)

            if is_white_turn_at_state_i:
                 final_values.append(result)
            else:
                 final_values.append(-result)

        # Send the completed game data back to the main process
        game_data_queue.put(GamePlayResult(
            board_states=board_states,
            policies=policies,
            values=final_values,
            result=result,
            moves=len(board.move_stack)
        ))

    except Exception as e:
        # It's critical to catch exceptions in workers to not hang the main process
        logging.error(f"Error in worker {worker_id}: {e}", exc_info=True)
        # Signal failure
        game_data_queue.put(None)

# --- Main Trainer Class ---

class AlphaZeroTrainer:
    """
    Orchestrates the AlphaZero training loop:
    1. Manages a central model on the main GPU.
    2. Spawns worker processes for self-play.
    3. Runs a central evaluation service to batch requests from workers.
    4. Collects game data into a replay buffer.
    5. Trains the model on data from the buffer.
    6. Evaluates the new model against the old one.
    7. Updates the "best" model if the new one is significantly better.
    """
    
    def __init__(self, config_path="configs/config.v2.yaml"):
        """Initializes the trainer, loading configuration and setting up components."""
        self.config = load_config(config_path)
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)

        # --- Hardware Setup ---
        desired_device = self.config['system']['device']
        if desired_device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # --- Model Initialization ---
        self.policy_size = get_policy_vector_size()
        self.model = self._create_model()
        if self.config['system']['compile_model']:
            if self.device.type != 'cpu':
                self.model = torch.compile(self.model)
        self.best_opponent_model = self._create_model()
        
        # --- Training Components ---
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.replay_buffer = StreamingReplayBuffer(
            root_dir=self.config['data']['replay_buffer_dir']
        )
        
        # --- State Tracking ---
        self.iteration = 0
        self.total_games_played = 0
        self.progress_file = os.path.join(self.config['logging']['log_dir'], 'alpha_zero_training', 'training_progress.json')

        self._load_progress()  # Start fresh; skip forced checkpoint loading

        # Queues for interprocess communication. Manager.Queue is slower but
        # works reliably across spawn-based processes on macOS; switchable via config.
        if self.config['system'].get('use_manager_queue', True):
            mgr = mp.Manager()
            self.inference_request_queue = mgr.Queue()
            self.inference_result_queue = mgr.Queue()
        else:
            self.inference_request_queue = mp.Queue()
            self.inference_result_queue = mp.Queue()

        # Launch dedicated GPU inference server
        ckpt_dir = self.config['training']['checkpoints']['dir']
        best_model = os.path.join(ckpt_dir, self.config['training']['checkpoints']['best_opponent_model'])
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.inference_server_proc = Process(
            target=run_inference_server,
            args=(self.inference_request_queue, self.inference_result_queue, best_model, device_str),
            daemon=True,
        )
        self.inference_server_proc.start()
        self.logger.info("Started GPU inference server PID %s", self.inference_server_proc.pid)

    def run(self):
        """Starts and runs the main AlphaZero training loop."""
        self.logger.info("--- Starting AlphaZero Training Loop ---")
        if self.iteration > 0:
            self.logger.info(f"Resuming from iteration {self.iteration}")

        max_iterations = self.config['training']['alpha_zero']['iterations']
        while self.iteration < max_iterations:
            self.iteration += 1
            self.logger.info(f"--- Iteration {self.iteration}/{max_iterations} ---")

            # 1. Self-Play Phase
            self.run_self_play()
            
            # 2. Training Phase
            if len(self.replay_buffer) >= self.config['training']['alpha_zero']['min_replay_buffer_size']:
                self.run_training()
            else:
                self.logger.info("Replay buffer still too small. Skipping training.")

            # 3. Evaluation Phase
            if self.iteration % self.config['training']['alpha_zero']['evaluation_interval'] == 0:
                self.run_evaluation()
                
            # 4. Save progress
            self._save_checkpoint(is_latest=True)
            self._save_progress()

    def run_self_play(self):
        """
        Manages the self-play phase using a pool of worker processes and a central evaluation service.
        """
        num_games = self.config['training']['alpha_zero']['games_per_iteration']
        num_workers = self.config['system']['self_play_workers']
        self.logger.info(f"Starting self-play with {num_workers} workers for {num_games} games.")

        self.model.eval()

        # --- Multiprocessing Setup ---
        ctx = mp.get_context('spawn')
        eval_result_queue = ctx.Queue()
        
        # Detached model state for workers
        self.model.cpu()
        target_model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        model_state_dict = target_model.state_dict()
        self.model.to(self.device)

        # Start workers
        processes = []
        games_per_worker = [num_games // num_workers] * num_workers
        for i in range(num_games % num_workers):
            games_per_worker[i] += 1

        for i in range(num_workers):
            if games_per_worker[i] == 0:
                continue
            # Note: eval_queue and result_queues are None because workers are self-sufficient
            args = (i, self.config_path, model_state_dict, self.inference_request_queue, self.inference_result_queue, eval_result_queue, games_per_worker[i])
            p = ctx.Process(target=game_worker_manager, args=args)
            processes.append(p)
            p.start()
        
        games_completed = 0
        game_stats = {'wins': 0, 'losses': 0, 'draws': 0, 'total_moves': 0}
        pbar = tqdm(total=num_games, desc="Self-Play Games", unit="game")

        # --- Collection Loop ---
        while games_completed < num_games:
            # Block and wait for a completed game from any worker
            result = eval_result_queue.get()
            
            if result:
                # Add each state from the completed game to the replay buffer
                for i in range(len(result.board_states)):
                    board_tensor = result.board_states[i]
                    # The replay buffer expects numpy arrays, not tensors.
                    board_numpy = board_tensor.cpu().numpy()
                    self.replay_buffer.add(
                        board=board_numpy,
                        policy=result.policies[i],
                        value=result.values[i]
                    )

                # Update stats
                # Note: Win/Loss is from white's perspective in the result. Self-play is symmetric.
                if result.result == WIN: game_stats['wins'] += 1
                elif result.result == LOSS: game_stats['losses'] += 1
                else: game_stats['draws'] += 1
                game_stats['total_moves'] += result.moves
                self.total_games_played += 1
            else:
                # A worker might have failed.
                self.logger.warning("A self-play worker returned a null result, indicating a potential error.")

            games_completed += 1
            pbar.update(1)
            if games_completed > 0:
                pbar.set_postfix({
                    "W/L/D": f"{game_stats['wins']}/{game_stats['losses']}/{game_stats['draws']}",
                    "Avg Moves": f"{game_stats['total_moves'] / games_completed:.1f}",
                    "Buffer": f"{len(self.replay_buffer)}"
                })

        pbar.close()
        # Ensure all workers are terminated
        for p in processes:
            p.join()

        self.logger.info(f"Self-play finished. Results: {game_stats['wins']}W/{game_stats['losses']}L/{game_stats['draws']}D")

    def run_training(self):
        """Trains the model on data from the replay buffer."""
        self.logger.info("--- Training model... ---")
        self.model.train()

        epochs = self.config['training']['alpha_zero']['epochs_per_iteration']
        batch_size = self.config['training']['alpha_zero']['batch_size']
        
        # Create a DataLoader from the replay buffer
        use_webdataset = self.config['data'].get('use_webdataset', True)
        if use_webdataset:
            shard_glob = os.path.join(self.config['data']['replay_buffer_dir'], 'shard-*.tar')
            import glob
            if not glob.glob(shard_glob):
                use_webdataset = False

        if use_webdataset:
            try:
                dataset = self.replay_buffer.to_webdataset().shuffle(1000).decode().to_tuple("pth", "pol.npy", "val.npy")
                loader = wds.WebLoader(dataset, batch_size=batch_size, num_workers=self.config['system'].get('dataloader_workers', 0))
                loader_len = max(1, len(self.replay_buffer) // batch_size)
            except Exception as e:
                self.logger.warning(f"Failed to build WebDataset loader ({e}). Falling back to in-memory.")
                use_webdataset = False

        if not use_webdataset:
            self.logger.info("Using in-memory replay buffer for training.")
            states, policies, values = self.replay_buffer.sample(batch_size * 256) # Sample a large chunk
            if states is None:
                self.logger.error("Replay buffer is empty. Cannot train.")
                return
            tensor_dataset = TensorDataset(states, policies, values)
            loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
            loader_len = max(1, len(loader))

        total_policy_loss = 0
        total_value_loss = 0
        
        for epoch in range(epochs):
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", total=loader_len)
            for states, policies, values in pbar:
                states, policies, values = states.to(self.device), policies.to(self.device), values.to(self.device)

                self.optimizer.zero_grad()
                
                pred_values, pred_policies = self.model(states)
                
                # Soft-label KL-divergence against the full MCTS target distro
                log_probs   = F.log_softmax(pred_policies, dim=1)
                policy_loss = F.kl_div(log_probs, policies, reduction='batchmean')
                value_loss = F.mse_loss(pred_values.squeeze(-1), values)
                loss = policy_loss + value_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip'])
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                pbar.set_postfix({"P-Loss": f"{policy_loss.item():.3f}", "V-Loss": f"{value_loss.item():.3f}"})
        
        self.scheduler.step()
        self.logger.info(f"Training complete. Avg Policy Loss: {total_policy_loss/loader_len/epochs:.4f}, Avg Value Loss: {total_value_loss/loader_len/epochs:.4f}")

    def run_evaluation(self):
        """Evaluates the current model against the best opponent."""
        self.logger.info("--- Evaluating model against best opponent ---")
        num_games = self.config['training']['alpha_zero']['evaluation_games']
        num_workers = self.config['system']['evaluation_workers']

        self.model.eval()
        self.best_opponent_model.eval()

        ctx = mp.get_context('spawn')
        eval_result_queue = ctx.Queue()

        self.model.cpu()
        self.best_opponent_model.cpu()

        player_target = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        opponent_target = self.best_opponent_model._orig_mod if hasattr(self.best_opponent_model, '_orig_mod') else self.best_opponent_model
        player_state_dict = player_target.state_dict()
        opponent_state_dict = opponent_target.state_dict()
        
        self.model.to(self.device)
        self.best_opponent_model.to(self.device)

        processes = []
        games_per_worker = [num_games // num_workers] * num_workers
        for i in range(num_games % num_workers):
            games_per_worker[i] += 1

        for i in range(num_workers):
            if games_per_worker[i] == 0:
                continue
            # Note: eval_queue and result_queues are None because workers are self-sufficient
            args = (i, self.config_path, player_state_dict, None, None, eval_result_queue, games_per_worker[i], opponent_state_dict)
            p = ctx.Process(target=game_worker_manager, args=args)
            processes.append(p)
            p.start()

        games_completed = 0
        wins, losses, draws = 0, 0, 0
        pbar = tqdm(total=num_games, desc="Evaluation Games")

        while games_completed < num_games:
            result = eval_result_queue.get() # Block and wait for a result
            if result is not None:
                if result == WIN: wins += 1
                elif result == LOSS: losses += 1
                else: draws += 1
            else:
                self.logger.warning("An evaluation worker returned a null result, indicating a potential error.")
            games_completed += 1
            pbar.update(1)
            pbar.set_postfix({"W/L/D": f"{wins}/{losses}/{draws}"})

        pbar.close()
        for p in processes: p.join()

        win_rate = (wins + 0.5 * draws) / num_games if num_games > 0 else 0
        self.logger.info(f"Evaluation complete. Win rate: {win_rate:.2%} ({wins}W/{losses}L/{draws}D)")

        if win_rate > self.config['training']['alpha_zero']['opponent_update_threshold']:
            self.logger.info(f"New model is superior. Updating best opponent model.")
            self.best_opponent_model.load_state_dict(self.model.state_dict())
            self._save_checkpoint(is_best=True)
        else:
            self.logger.info("New model did not meet threshold. Keeping current opponent.")

    def _create_model(self):
        model_config = self.config['model']['chess_transformer']
        model = ChessTransformer(
            input_channels=model_config['input_channels'],
            cnn_channels=model_config['cnn_channels'],
            cnn_blocks=model_config['cnn_blocks'],
            transformer_layers=model_config['transformer_layers'],
            attention_heads=model_config['attention_heads'],
            policy_head_output_size=self.policy_size
        )
        model.to(self.device)
        return model

    def _create_optimizer(self):
        opt_config = self.config['training']['optimizer']
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['alpha_zero']['learning_rate'],
            weight_decay=opt_config['weight_decay']
        )

    def _create_scheduler(self):
        sched_config = self.config['training']['scheduler']
        return CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=sched_config['T_0'],
            T_mult=sched_config['T_mult'],
            eta_min=self.config['training']['alpha_zero']['min_learning_rate']
        )

    def _save_checkpoint(self, is_latest=False, is_best=False):
        """Saves the current state of the model and optimizer."""
        chk_dir = self.config['training']['checkpoints']['dir']
        os.makedirs(chk_dir, exist_ok=True)

        # Always save the underlying model's state dict
        target_model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        state = {
            'iteration': self.iteration,
            'model_state_dict': target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'total_games_played': self.total_games_played
        }
        
        # Save the latest player model
        latest_path = os.path.join(chk_dir, self.config['training']['checkpoints']['latest_player_model'])
        torch.save(state, latest_path)
        self.logger.info(f"Saved checkpoint to {latest_path}")

        if is_best:
            path = os.path.join(chk_dir, self.config['training']['checkpoints']['best_opponent_model'])
            # For the opponent, we only need the model weights
            torch.save({'model_state_dict': target_model.state_dict()}, path)
            self.logger.info(f"Saved new best opponent to {path}")

    def _load_selective_checkpoint(self, model: torch.nn.Module, checkpoint_path: str):
        """
        Loads weights selectively into the *underlying* model,
        ignoring layers with shape mismatches and handling compiled/uncompiled checkpoints.
        This is a robust method that manually filters weights to avoid RuntimeError.
        """
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint not found at {checkpoint_path}. Using fresh model.")
            return

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            source_state_dict = checkpoint.get('model_state_dict', checkpoint)

            # Sanitize source checkpoint by removing compile prefix if it exists
            if any(key.startswith('_orig_mod.') for key in source_state_dict.keys()):
                from collections import OrderedDict
                unwrapped_state_dict = OrderedDict()
                for k, v in source_state_dict.items():
                    name = k.replace('_orig_mod.', '')
                    unwrapped_state_dict[name] = v
                source_state_dict = unwrapped_state_dict

            # Get the state dict of the actual underlying target model
            target_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            target_state_dict = target_model.state_dict()

            # Create a new state dict that only contains weights that match in name and shape
            new_state_dict = {}
            loaded_keys = []
            mismatched_keys = []

            for name, param in source_state_dict.items():
                if name in target_state_dict:
                    if target_state_dict[name].shape == param.shape:
                        new_state_dict[name] = param
                        loaded_keys.append(name)
                    else:
                        mismatched_keys.append(f"{name} (shape mismatch: {param.shape} vs {target_state_dict[name].shape})")

            # Load the filtered state dictionary
            target_model.load_state_dict(new_state_dict, strict=False)

            if mismatched_keys:
                self.logger.warning(f"Partially loaded model from {checkpoint_path}. Skipped layers with shape mismatches.")
                self.logger.debug(f"Mismatched keys: {mismatched_keys}")
            elif not loaded_keys:
                self.logger.error(f"Failed to load any weights from {checkpoint_path}. All keys mismatched or missing.")
            else:
                self.logger.info(f"Successfully loaded {len(loaded_keys)} weight tensors from {checkpoint_path}")

        except Exception as e:
            self.logger.error(f"Could not load checkpoint from {checkpoint_path}. Error: {e}", exc_info=True)

    def _load_checkpoint(self):
        # Per user instruction, force-load a specific checkpoint for both player and opponent.
        # This overrides the standard logic of loading 'latest' and 'best'.
        checkpoint_path = "/Users/admin/Downloads/VSCode/Native_Chess_Transformer/models/alpha_zero_checkpoints/checkpoint_iter_30.pt"
        self.logger.info(f"--- OVERRIDE: Forcing load from {checkpoint_path} for both player and opponent. ---")

        if not os.path.exists(checkpoint_path):
            self.logger.error(f"FATAL: Specified override checkpoint not found at {checkpoint_path}. Cannot proceed.")
            # Exit or raise a critical error because the user's explicit instruction cannot be met.
            sys.exit(1)
            
        # --- Load Player and Opponent Model ---
        # The selective loading function handles architecture mismatches.
        self.logger.info("Loading player model...")
        self._load_selective_checkpoint(self.model, checkpoint_path)
        
        self.logger.info("Loading opponent model...")
        self._load_selective_checkpoint(self.best_opponent_model, checkpoint_path)

        # Attempt to load optimizer/scheduler from the main checkpoint file
        # This might fail if the checkpoint is old, which is handled gracefully.
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'optimizer_state_dict' in checkpoint and 'scheduler_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                # Do NOT load iteration from here, as we are starting fresh from a specific point
                self.logger.info("Loaded optimizer and scheduler state from checkpoint.")
            else:
                self.logger.warning("Optimizer/scheduler state not in checkpoint. Using fresh state.")
        except Exception as e:
            self.logger.warning(f"Could not load optimizer/scheduler state from {checkpoint_path}: {e}. Using fresh state.")

        self.best_opponent_model.to(self.device)
        self.best_opponent_model.eval()

    def _save_progress(self):
        progress_data = {
            'iteration': self.iteration,
            'total_games_played': self.total_games_played,
            'replay_buffer_size': len(self.replay_buffer),
        }
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=4)
        except IOError as e:
            self.logger.error(f"Failed to save progress to {self.progress_file}: {e}")
        
    def _load_progress(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                self.iteration = progress_data.get('iteration', self.iteration)
                self.total_games_played = progress_data.get('total_games_played', self.total_games_played)
                self.logger.info(f"Loaded progress from {self.progress_file}. Resuming from iteration {self.iteration + 1}.")
            except (IOError, json.JSONDecodeError) as e:
                self.logger.error(f"Failed to load progress from {self.progress_file}: {e}. Starting with default values.")

# --------------------------- Inference queue helpers --------------------------

def create_remote_evaluator(req_q, res_q, device='cpu'):
    """Factory that returns an evaluator(board_tensor) callable using queues."""
    import os, torch, numpy as np
    def _eval(board_tensor: torch.Tensor):
        uid = os.urandom(16)
        req_q.put((uid, board_tensor.squeeze(0).cpu().numpy()))
        while True:
            try:
                resp_uid, policy_np, value = res_q.get(timeout=120)
            except Exception:
                raise RuntimeError("Inference server timeout after 120s; check GPU server logs")
            if resp_uid == uid:
                policy = torch.from_numpy(np.asarray(policy_np)).float()
                val = torch.tensor(value, dtype=torch.float32)
                return val, policy
    return _eval

if __name__ == "__main__":
    # This block allows the script to be run directly.
    # It sets up logging and starts the training process.
    
    # --- Logging Setup ---
    # Create a dedicated log file for this run
    log_dir = "logs/alpha_zero_training"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"run_{time.strftime('%Y%m%d-%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout) # Keep logging to console as well
        ]
    )
    
    # Set the start method for multiprocessing
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass # Already set

    trainer = AlphaZeroTrainer(config_path="configs/config.v2.yaml")
    trainer.run() 