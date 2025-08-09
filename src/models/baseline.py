"""
Baseline Chess Models

This module contains baseline neural network models for chess:
1. Simple MLP for move prediction
2. CNN for board evaluation
3. Basic transformer for sequence modeling

These serve as starting points before implementing the full diffusion model.
"""

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessCNN(nn.Module):
    """
    Convolutional Neural Network for chess board evaluation.
    Takes board tensor as input and outputs position evaluation.
    """

    def __init__(
        self, input_channels: int = 12, hidden_channels: int = 64, num_blocks: int = 4
    ):
        """
        Initialize chess CNN.

        Args:
            input_channels: Number of input channels (12 for piece types)
            hidden_channels: Hidden layer channels
            num_blocks: Number of residual blocks
        """
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        # Initial convolution
        self.input_conv = nn.Conv2d(input_channels, hidden_channels, 3, padding=1)
        self.input_bn = nn.BatchNorm2d(hidden_channels)

        # Residual blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_channels) for _ in range(num_blocks)]
        )

        # Output heads
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),  # Output between -1 and 1
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 4096),  # 64*64 possible moves
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Board tensor of shape (batch_size, 12, 8, 8)

        Returns:
            Tuple of (value, policy) predictions
        """
        # Initial convolution
        x = F.relu(self.input_bn(self.input_conv(x)))

        # Residual blocks
        for block in self.blocks:
            x = block(x)

        # Output heads
        value = self.value_head(x)
        policy = self.policy_head(x)

        return value, policy


class ResidualBlock(nn.Module):
    """Residual block for CNN."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0), :].to(x.device)


class ChessTransformer(nn.Module):
    """
    Transformer model for chess move sequence prediction.
    Can be used as a baseline before implementing diffusion.
    """

    def __init__(
        self,
        vocab_size: int = 4096,  # 64*64 possible moves
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        """
        Initialize chess transformer.

        Args:
            vocab_size: Size of move vocabulary
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embeddings
        self.move_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        # Board encoder (convert 12x8x8 to sequence)
        self.board_encoder = nn.Sequential(
            nn.Conv2d(12, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, d_model, 1),
            nn.Flatten(2),  # (batch, d_model, 64)
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        board: torch.Tensor,
        move_sequence: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            board: Board tensor (batch_size, 12, 8, 8)
            move_sequence: Previous moves (batch_size, seq_len)
            mask: Attention mask

        Returns:
            Move logits (batch_size, seq_len, vocab_size)
        """
        # Encode board position
        board_features = self.board_encoder(board)  # (batch, d_model, 64)
        board_features = board_features.transpose(1, 2)  # (batch, 64, d_model)

        if move_sequence is not None:
            # Encode move sequence
            move_embeddings = self.move_embedding(move_sequence) * math.sqrt(
                self.d_model
            )
            move_embeddings = self.pos_encoding(
                move_embeddings.transpose(0, 1)
            ).transpose(0, 1)

            # Concatenate board and move features
            features = torch.cat([board_features, move_embeddings], dim=1)
        else:
            features = board_features

        # Apply transformer
        output = self.transformer(features, mask=mask)

        # Project to vocabulary
        if move_sequence is not None:
            # Only return move predictions (skip board tokens)
            move_output = output[:, board_features.size(1) :, :]
            logits = self.output_proj(move_output)
        else:
            # For board-only input, predict next move
            logits = self.output_proj(output.mean(dim=1, keepdim=True))

        return logits


class SimpleMovePredictor(nn.Module):
    """
    Simple MLP for move prediction.
    Good baseline for testing training pipeline.
    """

    def __init__(
        self,
        input_size: int = 12 * 8 * 8,  # Flattened board
        hidden_size: int = 1024,
        output_size: int = 4096,  # 64*64 possible moves
        num_layers: int = 3,
    ):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))

        layers.append(nn.Linear(hidden_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Board tensor (batch_size, 12, 8, 8)

        Returns:
            Move logits (batch_size, 4096)
        """
        x = x.view(x.size(0), -1)  # Flatten
        return self.network(x)


class HybridChessModel(nn.Module):
    """
    Hybrid model combining CNN and Transformer.
    Prepares for diffusion model integration.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.config = config

        # CNN backbone for board understanding
        self.cnn_backbone = ChessCNN(
            input_channels=config.get("input_channels", 12),
            hidden_channels=config.get("cnn_channels", 64),
            num_blocks=config.get("cnn_blocks", 4),
        )

        # Transformer for sequence modeling
        self.transformer = ChessTransformer(
            vocab_size=config.get("vocab_size", 4096),
            d_model=config.get("d_model", 512),
            nhead=config.get("nhead", 8),
            num_layers=config.get("num_layers", 6),
        )

        # Feature fusion
        self.feature_fusion = nn.Linear(
            config.get("cnn_channels", 64) * 8 * 8 + config.get("d_model", 512),
            config.get("d_model", 512),
        )

        # Output heads
        self.value_head = nn.Linear(config.get("d_model", 512), 1)
        self.policy_head = nn.Linear(
            config.get("d_model", 512), config.get("vocab_size", 4096)
        )

    def forward(self, board: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass combining CNN and Transformer features.

        Args:
            board: Board tensor (batch_size, 12, 8, 8)

        Returns:
            Tuple of (value, policy) predictions
        """
        # Get CNN features (before final heads)
        cnn_features = self.cnn_backbone.input_conv(board)
        for block in self.cnn_backbone.blocks:
            cnn_features = block(cnn_features)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)

        # Get transformer features
        transformer_logits = self.transformer(board)
        transformer_features = transformer_logits.mean(dim=1)  # Pool sequence

        # Fuse features
        combined_features = torch.cat([cnn_features, transformer_features], dim=1)
        fused_features = self.feature_fusion(combined_features)

        # Output predictions
        value = torch.tanh(self.value_head(fused_features))
        policy = F.log_softmax(self.policy_head(fused_features), dim=1)

        return value, policy


def create_model(model_type: str, config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create different model types.

    Args:
        model_type: Type of model to create
        config: Model configuration

    Returns:
        Initialized model
    """
    if model_type == "cnn":
        # Extract CNN-specific parameters
        cnn_params = {
            "input_channels": config.get("input_channels", 12),
            "hidden_channels": config.get("hidden_channels", 64),
            "num_blocks": config.get("num_blocks", 4),
        }
        return ChessCNN(**cnn_params)
    elif model_type == "transformer":
        # Extract transformer-specific parameters
        transformer_params = {
            "vocab_size": config.get("vocab_size", 4096),
            "d_model": config.get("d_model", 512),
            "nhead": config.get("nhead", 8),
            "num_layers": config.get("num_layers", 6),
            "dim_feedforward": config.get("dim_feedforward", 2048),
            "dropout": config.get("dropout", 0.1),
        }
        return ChessTransformer(**transformer_params)
    elif model_type == "mlp":
        # Extract MLP-specific parameters
        mlp_params = {
            "input_size": config.get("input_size", 12 * 8 * 8),
            "hidden_size": config.get("hidden_size", 1024),
            "output_size": config.get("output_size", 4096),
            "num_layers": config.get("num_layers", 3),
        }
        return SimpleMovePredictor(**mlp_params)
    elif model_type == "hybrid":
        return HybridChessModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Example usage
if __name__ == "__main__":
    # Test models
    batch_size = 4
    board = torch.randn(batch_size, 12, 8, 8)  # Random board

    print("Testing baseline models...")

    # Test CNN
    cnn = ChessCNN()
    value, policy = cnn(board)
    print(f"CNN - Value shape: {value.shape}, Policy shape: {policy.shape}")

    # Test Transformer
    transformer = ChessTransformer()
    logits = transformer(board)
    print(f"Transformer - Output shape: {logits.shape}")

    # Test MLP
    mlp = SimpleMovePredictor()
    logits = mlp(board)
    print(f"MLP - Output shape: {logits.shape}")

    # Test Hybrid
    config = {
        "input_channels": 12,
        "cnn_channels": 64,
        "cnn_blocks": 4,
        "vocab_size": 4096,
        "d_model": 512,
        "nhead": 8,
        "num_layers": 6,
    }
    hybrid = HybridChessModel(config)
    value, policy = hybrid(board)
    print(f"Hybrid - Value shape: {value.shape}, Policy shape: {policy.shape}")

    print("All models tested successfully! 🚀")
