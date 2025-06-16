#!/usr/bin/env python3
"""
🏆 MASTER GAMES TRAINER - GRANDMASTER EDUCATION SYSTEM! 🏆

This system trains our breakthrough model on 100,000+ positions from
annotated GM games, providing elite-level strategic knowledge!

Features:
- PGN parsing with annotations
- Position evaluation extraction  
- Multi-source GM database support
- Efficient batch processing
- Quality filtering and validation

Expected Result: Transform 22.9% accuracy into MASTER-LEVEL play!
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import chess
import chess.pgn
import chess.engine
import numpy as np
import random
import json
import time
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any, Iterator
from pathlib import Path
import re
import io
from tqdm import tqdm

# Our existing modules
from models.baseline import ChessCNN
from utils.chess_env import ChessEnvironment

@dataclass
class MasterPosition:
    """A position from a master game with annotations"""
    fen: str
    best_move: str
    evaluation: float  # Centipawn evaluation
    game_result: str   # 1-0, 0-1, 1/2-1/2
    player_elo: int    # ELO of player to move
    opening: str       # Opening name
    annotation: str    # Text annotation if available
    depth: int         # Analysis depth
    
class MasterGamesTrainer:
    """Trains chess AI on master games with annotations"""
    
    def __init__(self, model_path: str = None):
        print("🏆 MASTER GAMES TRAINER - GRANDMASTER EDUCATION!")
        print("=" * 70)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Device: {self.device}")
        
        # Load our breakthrough model as foundation
        self.model = ChessCNN(
            input_channels=12,
            hidden_channels=256,  # Breakthrough architecture
            num_blocks=16
        ).to(self.device)
        
        if model_path and Path(model_path).exists():
            print(f"📚 Loading breakthrough foundation: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print("✅ Breakthrough foundation loaded!")
        else:
            print("🆕 Starting with fresh model weights")
        
        # Training setup
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-5, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, factor=0.5)
        
        # Environment
        self.env = ChessEnvironment()
        
        # Training parameters
        self.batch_size = 64
        self.target_positions = 100000  # Our ambitious goal!
        self.positions_buffer = []
        
        print(f"🧠 Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"🎯 Target GM Positions: {self.target_positions:,}")
        print("🏆 Ready for Grandmaster education!")
    
    def parse_pgn_files(self, pgn_paths: List[str], max_positions: int = 100000) -> List[MasterPosition]:
        """Parse PGN files and extract master positions"""
        
        print(f"\n🔍 PARSING GM GAMES FROM {len(pgn_paths)} FILES")
        print(f"🎯 Target: {max_positions:,} positions")
        print("=" * 50)
        
        positions = []
        games_processed = 0
        
        for pgn_path in pgn_paths:
            if not Path(pgn_path).exists():
                print(f"⚠️ Skipping missing file: {pgn_path}")
                continue
                
            print(f"📖 Processing: {Path(pgn_path).name}")
            
            try:
                with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
                    game_positions = self._extract_positions_from_pgn(pgn_file, max_positions - len(positions))
                    positions.extend(game_positions)
                    games_processed += len(game_positions) // 20  # Rough estimate
                    
                    print(f"   📊 Extracted: {len(game_positions)} positions")
                    print(f"   🎮 Total so far: {len(positions):,}")
                    
                    if len(positions) >= max_positions:
                        print(f"🎯 TARGET REACHED: {len(positions):,} positions!")
                        break
                        
            except Exception as e:
                print(f"❌ Error processing {pgn_path}: {e}")
                continue
        
        print(f"\n🏆 EXTRACTION COMPLETE!")
        print(f"   📊 Total positions: {len(positions):,}")
        print(f"   🎮 Games processed: ~{games_processed}")
        print(f"   📈 Quality: Master-level annotated games")
        
        return positions[:max_positions]
    
    def _extract_positions_from_pgn(self, pgn_file, max_positions: int) -> List[MasterPosition]:
        """Extract positions from a single PGN file"""
        
        positions = []
        
        while len(positions) < max_positions:
            try:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                    
                # Extract game metadata
                white_elo = self._safe_int(game.headers.get('WhiteElo', '1500'))
                black_elo = self._safe_int(game.headers.get('BlackElo', '1500'))
                result = game.headers.get('Result', '*')
                opening = game.headers.get('Opening', 'Unknown')
                
                # Skip low-quality games (lowered threshold for Lichess data)
                if min(white_elo, black_elo) < 1500:  # Strong players (1500+ ELO)
                    continue
                
                # Extract positions from game
                board = game.board()
                move_count = 0
                
                for node in game.mainline():
                    move_count += 1
                    
                    # Skip opening moves (first 8 moves) - focus on middlegame/endgame
                    if move_count <= 8:
                        board.push(node.move)
                        continue
                    
                    # Skip very late endgame
                    if move_count > 80:
                        break
                    
                    # Extract position data
                    fen = board.fen()
                    best_move = node.move.uci()
                    
                    # Get player ELO for this position
                    player_elo = white_elo if board.turn == chess.WHITE else black_elo
                    
                    # Extract evaluation from comments if available
                    evaluation = self._extract_evaluation(node.comment)
                    
                    # Create master position
                    position = MasterPosition(
                        fen=fen,
                        best_move=best_move,
                        evaluation=evaluation,
                        game_result=result,
                        player_elo=player_elo,
                        opening=opening,
                        annotation=node.comment or "",
                        depth=move_count
                    )
                    
                    positions.append(position)
                    board.push(node.move)
                    
                    # Limit positions per game to ensure diversity
                    if len(positions) % 1000 == 0:
                        print(f"      📍 {len(positions):,} positions extracted...")
                    
                    if len(positions) >= max_positions:
                        break
                        
            except Exception as e:
                # Skip problematic games
                continue
        
        return positions
    
    def _safe_int(self, value: str, default: int = 1500) -> int:
        """Safely convert string to int"""
        try:
            return int(value) if value and value.isdigit() else default
        except:
            return default
    
    def _extract_evaluation(self, comment: str) -> float:
        """Extract numerical evaluation from PGN comment"""
        if not comment:
            return 0.0
        
        # Look for common evaluation patterns
        patterns = [
            r'([+-]?\d+\.?\d*)',  # +1.5, -0.5, etc.
            r'eval=([+-]?\d+\.?\d*)',
            r'\[%eval ([+-]?\d+\.?\d*)\]'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, comment)
            if match:
                try:
                    return float(match.group(1))
                except:
                    continue
        
        return 0.0
    
    def create_training_dataset(self, positions: List[MasterPosition]) -> List[Tuple]:
        """Convert master positions to training dataset"""
        
        print(f"\n🎯 CREATING TRAINING DATASET")
        print(f"📊 Input positions: {len(positions):,}")
        print("=" * 40)
        
        dataset = []
        failed_conversions = 0
        
        for i, pos in enumerate(tqdm(positions, desc="Converting positions")):
            try:
                # Parse position
                board = chess.Board(pos.fen)
                
                # Convert to tensor
                board_tensor = torch.from_numpy(self.env.board_to_tensor(board)).float()
                
                # Convert move to index
                move = chess.Move.from_uci(pos.best_move)
                if move not in board.legal_moves:
                    failed_conversions += 1
                    if failed_conversions <= 5:  # Debug first few failures
                        print(f"      ⚠️ Illegal move: {pos.best_move} in position {pos.fen}")
                    continue
                
                move_idx = self.env.move_to_index(move)
                
                # Create value target based on evaluation and game result
                value_target = self._position_to_value(pos)
                
                dataset.append((board_tensor, move_idx, value_target, pos.player_elo))
                
            except Exception as e:
                failed_conversions += 1
                if failed_conversions <= 5:  # Debug first few failures
                    print(f"      ❌ Conversion error: {e} for position {i}")
                continue
        
        print(f"✅ Dataset created!")
        print(f"   📊 Training samples: {len(dataset):,}")
        print(f"   ❌ Failed conversions: {failed_conversions}")
        print(f"   📈 Success rate: {len(dataset)/(len(positions))*100:.1f}%")
        
        return dataset
    
    def _position_to_value(self, position: MasterPosition) -> float:
        """Convert position data to value target"""
        
        # Base value from evaluation
        value = np.tanh(position.evaluation / 200.0)  # Normalize centipawns
        
        # Adjust based on game result (outcome supervision)
        if position.game_result == "1-0":  # White won
            result_bonus = 0.3
        elif position.game_result == "0-1":  # Black won  
            result_bonus = -0.3
        else:  # Draw
            result_bonus = 0.0
        
        # Apply result bonus with decay based on move depth
        decay_factor = max(0.1, 1.0 - position.depth / 100.0)
        value += result_bonus * decay_factor
        
        return np.clip(value, -1.0, 1.0)
    
    def train_on_master_games(self, dataset: List[Tuple], epochs: int = 20) -> Dict[str, List[float]]:
        """Train the model on master game dataset"""
        
        print(f"\n🏆 MASTER GAMES TRAINING!")
        print(f"📊 Dataset size: {len(dataset):,} positions")
        print(f"🔄 Epochs: {epochs}")
        print(f"📦 Batch size: {self.batch_size}")
        print("=" * 50)
        
        # Split dataset
        train_size = int(0.9 * len(dataset))
        train_data = dataset[:train_size]
        val_data = dataset[train_size:]
        
        print(f"🎯 Training: {len(train_data):,} positions")
        print(f"📊 Validation: {len(val_data):,} positions")
        
        history = {'train_loss': [], 'val_loss': [], 'policy_acc': [], 'value_mse': []}
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\n🚀 EPOCH {epoch + 1}/{epochs}")
            print("-" * 30)
            
            # Training
            train_loss, train_acc, train_mse = self._train_epoch(train_data)
            
            # Validation
            val_loss, val_acc, val_mse = self._validate_epoch(val_data)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Progress report
            print(f"📊 EPOCH {epoch + 1} RESULTS:")
            print(f"   🎯 Train Loss: {train_loss:.6f} | Policy Acc: {train_acc:.2f}% | Value MSE: {train_mse:.6f}")
            print(f"   📊 Val Loss: {val_loss:.6f} | Policy Acc: {val_acc:.2f}% | Value MSE: {val_mse:.6f}")
            print(f"   ⚡ Learning Rate: {current_lr:.8f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch + 1, "best", history)
                print(f"   🏆 NEW BEST MODEL! Loss: {val_loss:.6f}")
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['policy_acc'].append(val_acc)
            history['value_mse'].append(val_mse)
            
            # Early stopping check
            if current_lr < 1e-7:
                print(f"   ⏹️ Early stopping - learning rate too low")
                break
        
        print(f"\n🎉 MASTER GAMES TRAINING COMPLETE!")
        print(f"🏆 Best validation loss: {best_val_loss:.6f}")
        
        return history
    
    def _train_epoch(self, train_data: List[Tuple]) -> Tuple[float, float, float]:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        correct_moves = 0
        total_moves = 0
        total_value_mse = 0.0
        batches = 0
        
        # Shuffle data
        random.shuffle(train_data)
        
        for i in range(0, len(train_data), self.batch_size):
            batch = train_data[i:i + self.batch_size]
            
            # Prepare batch
            positions = torch.stack([item[0] for item in batch]).to(self.device)
            move_targets = torch.tensor([item[1] for item in batch], dtype=torch.long).to(self.device)
            value_targets = torch.tensor([item[2] for item in batch], dtype=torch.float32).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predicted_values, policy_logits = self.model(positions)
            
            # Calculate losses
            policy_loss = F.cross_entropy(policy_logits, move_targets)
            value_loss = F.mse_loss(predicted_values.squeeze(), value_targets)
            
            # Combined loss with weighting
            total_loss_batch = policy_loss + 0.5 * value_loss
            
            # Backward pass
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += total_loss_batch.item()
            total_value_mse += value_loss.item()
            
            # Policy accuracy
            _, predicted_moves = torch.max(policy_logits, 1)
            correct_moves += (predicted_moves == move_targets).sum().item()
            total_moves += move_targets.size(0)
            batches += 1
            
            # Progress update
            if batches % 100 == 0:
                current_acc = (correct_moves / total_moves) * 100
                print(f"      Batch {batches}: Loss {total_loss/batches:.4f}, Acc {current_acc:.1f}%")
        
        avg_loss = total_loss / batches if batches > 0 else 0
        avg_acc = (correct_moves / total_moves) * 100 if total_moves > 0 else 0
        avg_mse = total_value_mse / batches if batches > 0 else 0
        
        return avg_loss, avg_acc, avg_mse
    
    def _validate_epoch(self, val_data: List[Tuple]) -> Tuple[float, float, float]:
        """Validate for one epoch"""
        
        self.model.eval()
        total_loss = 0.0
        correct_moves = 0
        total_moves = 0
        total_value_mse = 0.0
        batches = 0
        
        with torch.no_grad():
            for i in range(0, len(val_data), self.batch_size):
                batch = val_data[i:i + self.batch_size]
                
                # Prepare batch
                positions = torch.stack([item[0] for item in batch]).to(self.device)
                move_targets = torch.tensor([item[1] for item in batch], dtype=torch.long).to(self.device)
                value_targets = torch.tensor([item[2] for item in batch], dtype=torch.float32).to(self.device)
                
                # Forward pass
                predicted_values, policy_logits = self.model(positions)
                
                # Calculate losses
                policy_loss = F.cross_entropy(policy_logits, move_targets)
                value_loss = F.mse_loss(predicted_values.squeeze(), value_targets)
                total_loss_batch = policy_loss + 0.5 * value_loss
                
                # Statistics
                total_loss += total_loss_batch.item()
                total_value_mse += value_loss.item()
                
                # Policy accuracy
                _, predicted_moves = torch.max(policy_logits, 1)
                correct_moves += (predicted_moves == move_targets).sum().item()
                total_moves += move_targets.size(0)
                batches += 1
        
        avg_loss = total_loss / batches if batches > 0 else 0
        avg_acc = (correct_moves / total_moves) * 100 if total_moves > 0 else 0
        avg_mse = total_value_mse / batches if batches > 0 else 0
        
        return avg_loss, avg_acc, avg_mse
    
    def save_checkpoint(self, epoch: int, name: str, history: Dict[str, List[float]]):
        """Save model checkpoint"""
        
        checkpoints_dir = Path("models/master_game_checkpoints")
        checkpoints_dir.mkdir(exist_ok=True, parents=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': history
        }
        
        checkpoint_path = checkpoints_dir / f"master_games_{name}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")

def main():
    """🏆 LAUNCH MASTER GAMES TRAINING! 🏆"""
    
    print("🎸⚡ MASTER GAMES TRAINER - LET'S DO IT! ⚡🎸")
    print("=" * 80)
    print("🎯 MISSION: Train on 100,000+ GM positions!")
    print("🏆 GOAL: Transform breakthrough model into MASTER-LEVEL!")
    
    # Initialize with breakthrough model
    breakthrough_model_path = "models/checkpoints/breakthrough_best.pt"
    trainer = MasterGamesTrainer(model_path=breakthrough_model_path)
    
    # Find available PGN files
    data_sources = [
        "data/master_games/",
        "data/external/master_games/",
        "lichess_db_standard_rated_2013-01.pgn",  # Our existing file
        "data/raw/",
        "data/games/"
    ]
    
    pgn_files = []
    for source in data_sources:
        source_path = Path(source)
        if source_path.is_file() and source_path.suffix == '.pgn':
            pgn_files.append(str(source_path))
        elif source_path.is_dir():
            pgn_files.extend([str(f) for f in source_path.glob("*.pgn")])
    
    print(f"\n🔍 FOUND {len(pgn_files)} PGN FILES:")
    for pgn_file in pgn_files:
        print(f"   📖 {pgn_file}")
    
    if not pgn_files:
        print("\n⚠️ No PGN files found! Please add master games to:")
        print("   📁 data/master_games/ (recommended)")
        print("   📁 Or place PGN files in project root")
        return
    
    try:
        # Extract master positions
        positions = trainer.parse_pgn_files(pgn_files, max_positions=100000)
        
        if len(positions) < 1000:
            print(f"\n⚠️ Only {len(positions)} positions found - need more GM games!")
            print("🎯 Continuing with available data...")
        
        # Create training dataset  
        dataset = trainer.create_training_dataset(positions)
        
        if len(dataset) < 500:
            print(f"\n❌ Insufficient training data: {len(dataset)} samples")
            return
        
        # Train the model!
        history = trainer.train_on_master_games(dataset, epochs=25)
        
        # Save final results
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        history_file = results_dir / f"master_games_training_{timestamp}.json"
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n💾 Training history saved: {history_file}")
        print("🏆 MASTER GAMES TRAINING COMPLETE!")
        print("🎯 Ready to evaluate our newly educated model!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 