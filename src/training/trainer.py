"""
Training Pipeline for Chess Neural Networks

This module provides training infrastructure for chess models including:
- Data loading and preprocessing
- Model training with various optimizers
- Model evaluation and checkpointing
- Metrics tracking and visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
import yaml
from pathlib import Path
import json
from tqdm import tqdm
import time
from datetime import datetime
import os

from ..models.baseline import create_model
from ..utils.chess_env import ChessEnvironment

logger = logging.getLogger(__name__)


class ChessDataset(Dataset):
    """
    PyTorch Dataset for chess training data.
    """
    
    def __init__(self, data_path: str, transform=None):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to training data NPZ file
            transform: Optional data transforms
        """
        self.data_path = data_path
        self.transform = transform
        
        # Load data from NPZ file
        data = np.load(data_path)
        self.positions = torch.tensor(data['positions'], dtype=torch.float32)
        self.moves = torch.tensor(data['moves'], dtype=torch.long)
        self.evaluations = torch.tensor(data['evaluations'], dtype=torch.float32)
        
        logger.info(f"Loaded {len(self.positions)} training examples from {data_path}")
    
    def __len__(self) -> int:
        return len(self.positions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get training example.
        
        Returns:
            Dictionary with:
            - board: Board tensor (12, 8, 8)
            - move_index: Move index for policy target
            - evaluation: Position evaluation for value target
        """
        
        sample = {
            "board": self.positions[idx],
            "move_index": self.moves[idx],
            "evaluation": self.evaluations[idx],
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class ChessTrainer:
    """
    Main training class for chess neural networks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device(config["hardware"]["device"])
        
        # Setup directories
        self.log_dir = Path(config["logging"]["log_dir"])
        self.save_dir = Path(config["logging"]["save_dir"])
        self.log_dir.mkdir(exist_ok=True)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize model
        self.model = create_model(
            config["model"]["type"], 
            config["model"]
        ).to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Loss functions
        self.value_criterion = nn.MSELoss()
        self.policy_criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_loss = float('inf')
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        
        # ------------------------------------------------------------------
        # Optional experiment trackers
        # ------------------------------------------------------------------
        from torch.utils.tensorboard import SummaryWriter  # local import to avoid heavy dep if unused

        self.tb_writer: Optional[SummaryWriter] = None
        if self.config.get("logging", {}).get("tensorboard", True):
            tb_dir = self.log_dir / "tb"
            tb_dir.mkdir(exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))

        self.wandb_run: Optional[object] = None
        if self.config.get("logging", {}).get("use_wandb", False):
            try:
                import wandb

                wandb_project = self.config["logging"].get("wandb_project", "ChessTrainer")
                wandb_entity = self.config["logging"].get("wandb_entity")
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    config=self.config,
                    dir=str(self.log_dir),
                )
            except Exception as e:  # noqa: BLE001  # Best-effort – don't crash training
                logger.warning(f"W&B init failed: {e}")
        
        logger.info(f"Trainer initialized with device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        optimizer_name = self.config["training"]["optimizer"].lower()
        lr = float(self.config["training"]["learning_rate"])
        weight_decay = float(self.config["training"].get("weight_decay", 0))
        
        if optimizer_name == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_name = self.config["training"].get("scheduler", "none").lower()
        
        if scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config["training"]["num_epochs"]
            )
        elif scheduler_name == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=30, 
                gamma=0.1
            )
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        value_losses = []
        policy_losses = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            board = batch["board"].to(self.device)
            move_index = batch["move_index"].to(self.device)
            evaluation = batch["evaluation"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get model output
            output = self.model(board)
            
            if isinstance(output, tuple):
                # Model returns both value and policy
                value_pred, policy_pred = output
                
                # Calculate losses
                value_loss = self.value_criterion(value_pred.squeeze(), evaluation)
                policy_loss = self.policy_criterion(policy_pred, move_index)
                
                # Combined loss
                total_loss = (
                    self.config["training"].get("value_weight", 0.5) * value_loss +
                    self.config["training"].get("policy_weight", 0.5) * policy_loss
                )
                
                value_losses.append(value_loss.item())
                policy_losses.append(policy_loss.item())
                
            else:
                # Model returns only one output - treat as policy prediction
                total_loss = self.policy_criterion(output, move_index)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if self.config["training"].get("gradient_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config["training"]["gradient_clip"]
                )
            
            self.optimizer.step()
            
            # Update metrics
            epoch_losses.append(total_loss.item())
            self.step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log at intervals
            if self.step % self.config["logging"]["log_interval"] == 0:
                self._log_training_step(total_loss.item())
        
        # Epoch metrics
        metrics = {
            "train_loss": np.mean(epoch_losses),
            "train_value_loss": np.mean(value_losses) if value_losses else 0,
            "train_policy_loss": np.mean(policy_losses) if policy_losses else 0,
        }
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        val_losses = []
        value_losses = []
        policy_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move to device
                board = batch["board"].to(self.device)
                move_index = batch["move_index"].to(self.device)
                evaluation = batch["evaluation"].to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'forward') and len(self.model.forward.__code__.co_varnames) > 2:
                    value_pred, policy_pred = self.model(board)
                    
                    value_loss = self.value_criterion(value_pred.squeeze(), evaluation)
                    policy_loss = self.policy_criterion(policy_pred, move_index)
                    
                    total_loss = (
                        self.config["training"]["diffusion_weight"] * value_loss +
                        self.config["training"]["evaluation_weight"] * policy_loss
                    )
                    
                    value_losses.append(value_loss.item())
                    policy_losses.append(policy_loss.item())
                    
                else:
                    output = self.model(board)
                    if output.size(1) == 1:
                        total_loss = self.value_criterion(output.squeeze(), evaluation)
                    else:
                        total_loss = self.policy_criterion(output, move_index)
                
                val_losses.append(total_loss.item())
        
        metrics = {
            "val_loss": np.mean(val_losses),
            "val_value_loss": np.mean(value_losses) if value_losses else 0,
            "val_policy_loss": np.mean(policy_losses) if policy_losses else 0,
        }
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Main training loop."""
        logger.info("Starting training...")
        
        for epoch in range(self.config["training"]["num_epochs"]):
            self.epoch = epoch
            start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            epoch_time = time.time() - start_time
            
            # Log epoch results
            logger.info(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_loss
            if is_best:
                self.best_loss = val_metrics['val_loss']
            
            if (epoch + 1) % self.config["logging"]["save_interval"] == 0 or is_best:
                self.save_checkpoint(all_metrics, is_best)
            
            # Store metrics
            self.train_losses.append(train_metrics['train_loss'])
            self.val_losses.append(val_metrics['val_loss'])
        
        logger.info("Training completed!")
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_loss": self.best_loss,
            "config": self.config,
            "metrics": metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{self.epoch:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with loss: {self.best_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]
        self.best_loss = checkpoint["best_loss"]
        
        logger.info(f"Loaded checkpoint from epoch {self.epoch}")
    
    def _log_training_step(self, loss: float) -> None:
        """Log training step metrics."""
        if self.tb_writer:
            self.tb_writer.add_scalar("train/loss", loss, self.step)

        if self.wandb_run:
            try:
                import wandb

                wandb.log({"train/loss": loss, "step": self.step}, step=self.step)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"wandb.log error: {e}")

        # Console fallback
        logger.debug(f"Step {self.step}: loss={loss:.4f}")


def main():
    """Main training function."""
    
    # Load configuration
    config_path = "configs/config.v2.yaml"
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            "model": {"type": "cnn"},
            "training": {
                "batch_size": 32,
                "learning_rate": 1e-4,
                "num_epochs": 10,
                "optimizer": "adamw",
                "diffusion_weight": 1.0,
                "evaluation_weight": 0.5
            },
            "hardware": {"device": "cpu"},
            "logging": {
                "log_dir": "logs",
                "save_dir": "models/checkpoints",
                "log_interval": 100,
                "save_interval": 5
            }
        }
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Check for training data
    data_dir = Path("data/processed")
    data_files = list(data_dir.glob("training_data_*.npz"))
    
    if not data_files:
        logger.error("No training data found. Run data generation first.")
        return
    
    # Create datasets
    latest_data_file = max(data_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Using training data: {latest_data_file}")
    
    # ------------------------------------------------------------------
    # Train / validation split – configurable ratio (default 90/10)
    # ------------------------------------------------------------------
    full_dataset = ChessDataset(str(latest_data_file))
    val_ratio = float(config["training"].get("val_split", 0.1))
    val_size = max(1, int(len(full_dataset) * val_ratio))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["hardware"].get("num_workers", 0)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["hardware"].get("num_workers", 0)
    )
    
    # Create trainer and start training
    trainer = ChessTrainer(config)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
