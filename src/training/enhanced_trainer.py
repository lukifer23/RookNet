"""
Enhanced Training Pipeline for Scaled Chess Models

This module provides enhanced training capabilities for our scaled chess models:
- Multi-source data integration (Stockfish, puzzles, master games)
- Advanced training techniques (gradient clipping, learning rate scheduling)
- Comprehensive logging and checkpointing
- Model validation and performance tracking

Designed for training superhuman chess models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import yaml
import logging
import wandb
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm
import time
from datetime import datetime
import json

from ..models.baseline import create_model
from ..data.enhanced_data_generator import EnhancedDataGenerator
from ..utils.chess_env import ChessEnvironment

logger = logging.getLogger(__name__)


class ChessDataset(Dataset):
    """Enhanced chess dataset supporting multiple data sources."""
    
    def __init__(self, data: List[Dict[str, Any]]):
        """
        Initialize the chess dataset.
        
        Args:
            data: List of training examples from enhanced data generator
        """
        self.data = data
        
        # Convert to tensors for efficient loading
        self.boards = torch.stack([
            torch.FloatTensor(item["board_tensor"]) for item in data
        ])
        
        self.moves = torch.LongTensor([
            item["move_index"] for item in data
        ])
        
        self.evaluations = torch.FloatTensor([
            item["evaluation"] for item in data
        ])
        
        # Enhanced features
        self.game_phases = [item.get("game_phase", "unknown") for item in data]
        self.depths_used = torch.LongTensor([
            item.get("depth_used", 10) for item in data
        ])
        
        logger.info(f"Chess dataset initialized with {len(self.data)} positions")
        logger.info(f"Data sources: {self._analyze_data_sources()}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Get a training example.
        
        Returns:
            board tensor, move target, evaluation target, metadata
        """
        metadata = {
            "game_phase": self.game_phases[idx],
            "depth_used": self.depths_used[idx].item(),
            "is_tactical": self.data[idx].get("is_tactical", False),
            "is_master_game": self.data[idx].get("is_master_game", False),
            "is_opening": self.data[idx].get("is_opening", False)
        }
        
        return (
            self.boards[idx],
            self.moves[idx],
            self.evaluations[idx],
            metadata
        )
    
    def _analyze_data_sources(self) -> Dict[str, int]:
        """Analyze the composition of the dataset."""
        sources = {
            "stockfish_games": 0,
            "tactical_puzzles": 0,
            "master_games": 0,
            "opening_variations": 0
        }
        
        for item in self.data:
            if item.get("is_tactical"):
                sources["tactical_puzzles"] += 1
            elif item.get("is_master_game"):
                sources["master_games"] += 1
            elif item.get("is_opening"):
                sources["opening_variations"] += 1
            else:
                sources["stockfish_games"] += 1
        
        return sources


class EnhancedTrainer:
    """
    Enhanced trainer for scaled chess models with advanced techniques.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the enhanced trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Training parameters
        self.batch_size = config["training"]["batch_size"]
        self.learning_rate = float(config["training"]["learning_rate"])
        self.num_epochs = config["training"]["num_epochs"]
        self.weight_decay = float(config["training"]["weight_decay"])
        
        # Advanced training features
        self.gradient_clip_norm = config["training"].get("gradient_clip_norm", 1.0)
        self.use_mixed_precision = config["training"].get("use_mixed_precision", True)
        self.warmup_steps = config["training"].get("warmup_steps", 1000)
        
        # Model and training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        # Checkpointing
        self.checkpoint_dir = Path("models/checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Logging
        self.experiment_name = config.get("experiment_name", f"enhanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.use_wandb = config.get("use_wandb", False)
        
        logger.info(f"Enhanced trainer initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.use_mixed_precision}")
        logger.info(f"Experiment: {self.experiment_name}")
    
    def setup_model(self) -> None:
        """Setup the scaled model architecture."""
        
        logger.info("Setting up scaled model...")
        
        # Create scaled model
        self.model = create_model(
            self.config["model"]["type"],
            self.config["model"]
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"✅ Model created successfully!")
        logger.info(f"📊 Total Parameters: {total_params:,}")
        logger.info(f"📊 Trainable Parameters: {trainable_params:,}")
        logger.info(f"📊 Model Size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        
        # Setup optimizer with advanced features
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=self.num_epochs * 100,  # Estimate
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Mixed precision scaler
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        logger.info("✅ Training setup complete")
    
    def generate_enhanced_dataset(self) -> ChessDataset:
        """Generate enhanced training dataset."""
        
        logger.info("🚀 Generating enhanced dataset...")
        
        # Initialize enhanced data generator
        generator = EnhancedDataGenerator(self.config["data"])
        
        # Generate comprehensive dataset
        training_data = generator.generate_comprehensive_dataset()
        
        # Create dataset
        dataset = ChessDataset(training_data)
        
        logger.info(f"✅ Enhanced dataset ready: {len(dataset)} positions")
        
        return dataset
    
    def train(self, dataset: Optional[ChessDataset] = None) -> Dict[str, Any]:
        """
        Train the enhanced model.
        
        Args:
            dataset: Optional pre-generated dataset
            
        Returns:
            Training results and metrics
        """
        logger.info("🚀 Starting enhanced training...")
        
        # Setup model if not already done
        if self.model is None:
            self.setup_model()
        
        # Generate dataset if not provided
        if dataset is None:
            dataset = self.generate_enhanced_dataset()
        
        # Create data loader with enhanced features
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        # Initialize WandB if enabled
        if self.use_wandb:
            wandb.init(
                project="chess-ai-scaling",
                name=self.experiment_name,
                config=self.config
            )
        
        # Training metrics
        training_metrics = {
            "epoch_losses": [],
            "epoch_accuracies": [],
            "learning_rates": [],
            "best_loss": float('inf'),
            "best_accuracy": 0.0
        }
        
        # Training loop
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            
            epoch_metrics = self._train_epoch(dataloader, epoch)
            
            # Update training metrics
            training_metrics["epoch_losses"].append(epoch_metrics["loss"])
            training_metrics["epoch_accuracies"].append(epoch_metrics["accuracy"])
            training_metrics["learning_rates"].append(self.optimizer.param_groups[0]['lr'])
            
            # Check for best model
            if epoch_metrics["loss"] < training_metrics["best_loss"]:
                training_metrics["best_loss"] = epoch_metrics["loss"]
                self._save_checkpoint(epoch, is_best=True)
            
            if epoch_metrics["accuracy"] > training_metrics["best_accuracy"]:
                training_metrics["best_accuracy"] = epoch_metrics["accuracy"]
            
            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} | "
                f"Loss: {epoch_metrics['loss']:.4f} | "
                f"Accuracy: {epoch_metrics['accuracy']:.3f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # WandB logging
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": epoch_metrics["loss"],
                    "train_accuracy": epoch_metrics["accuracy"],
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "epoch_time": epoch_time
                })
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch)
        
        # Final checkpoint
        self._save_checkpoint(self.num_epochs - 1, is_final=True)
        
        # Save training results
        self._save_training_results(training_metrics)
        
        logger.info("✅ Enhanced training completed!")
        logger.info(f"Best loss: {training_metrics['best_loss']:.4f}")
        logger.info(f"Best accuracy: {training_metrics['best_accuracy']:.3f}")
        
        return training_metrics
    
    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train a single epoch."""
        
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # Enhanced loss components
        value_loss_fn = nn.MSELoss()
        policy_loss_fn = nn.CrossEntropyLoss()
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (boards, moves, evaluations, metadata) in enumerate(pbar):
            # Move to device
            boards = boards.to(self.device)
            moves = moves.to(self.device)
            evaluations = evaluations.to(self.device).unsqueeze(1)
            
            # Forward pass with mixed precision
            if self.use_mixed_precision and self.scaler:
                with torch.cuda.amp.autocast():
                    value_pred, policy_pred = self.model(boards)
                    
                    # Enhanced loss calculation
                    value_loss = value_loss_fn(value_pred, evaluations / 100.0)  # Normalize evaluations
                    policy_loss = policy_loss_fn(policy_pred, moves)
                    
                    # Weighted combination
                    total_loss_batch = 0.5 * value_loss + 0.5 * policy_loss
            else:
                value_pred, policy_pred = self.model(boards)
                
                # Enhanced loss calculation
                value_loss = value_loss_fn(value_pred, evaluations / 100.0)
                policy_loss = policy_loss_fn(policy_pred, moves)
                
                # Weighted combination
                total_loss_batch = 0.5 * value_loss + 0.5 * policy_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_mixed_precision and self.scaler:
                self.scaler.scale(total_loss_batch).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss_batch.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                
                self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step()
            
            # Calculate accuracy
            with torch.no_grad():
                _, predicted = torch.max(policy_pred, 1)
                correct = (predicted == moves).sum().item()
                total_correct += correct
                total_samples += moves.size(0)
            
            # Update metrics
            total_loss += total_loss_batch.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'Acc': f'{correct/moves.size(0):.3f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = total_correct / total_samples
        
        return {
            "loss": epoch_loss,
            "accuracy": epoch_accuracy
        }
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False, is_final: bool = False) -> None:
        """Save model checkpoint."""
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "config": self.config,
            "experiment_name": self.experiment_name
        }
        
        # Save paths
        if is_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
        elif is_final:
            checkpoint_path = self.checkpoint_dir / f"final_model_{self.experiment_name}.pt"
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_training_results(self, metrics: Dict[str, Any]) -> None:
        """Save training results and metadata."""
        
        results = {
            "experiment_name": self.experiment_name,
            "config": self.config,
            "training_metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "device": str(self.device),
            "total_parameters": sum(p.numel() for p in self.model.parameters())
        }
        
        results_path = Path("logs") / f"training_results_{self.experiment_name}.json"
        results_path.parent.mkdir(exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Training results saved: {results_path}")


def main():
    """Main function for testing the enhanced trainer."""
    
    # Load configuration
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Override for testing
    config["training"]["num_epochs"] = 3
    config["data"]["num_games"] = 50
    config["data"]["positions_per_game"] = 20
    config["experiment_name"] = "enhanced_trainer_test"
    
    # Initialize and run trainer
    trainer = EnhancedTrainer(config)
    results = trainer.train()
    
    print(f"Training completed! Best loss: {results['best_loss']:.4f}")


if __name__ == "__main__":
    main() 