"""
🚀 Chess Transformer Architecture
Hybrid CNN + Multi-Head Attention for Superhuman Chess Performance

Target: 100M parameters, 2400+ ELO rating
Architecture: ResNet backbone + Transformer layers + Strategic heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

class ChessResidualBlock(nn.Module):
    """Enhanced residual block with optional attention"""
    
    def __init__(self, channels: int, use_attention: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Optional spatial attention
        self.use_attention = use_attention
        if use_attention:
            self.attention = SpatialAttention(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.use_attention:
            out = self.attention(out)
        
        out = out + residual  # avoid in-place operation for torch.compile compatibility
        return F.relu(out).contiguous()

class SpatialAttention(nn.Module):
    """Spatial attention for chess board positions"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv_query = nn.Conv2d(channels, channels // 8, 1)
        self.conv_key = nn.Conv2d(channels, channels // 8, 1)
        self.conv_value = nn.Conv2d(channels, channels, 1)
        self.scale = (channels // 8) ** -0.5
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate Q, K, V
        q = self.conv_query(x).reshape(B, -1, H * W).transpose(1, 2)  # B, HW, C//8
        k = self.conv_key(x).reshape(B, -1, H * W)                    # B, C//8, HW
        v = self.conv_value(x).reshape(B, -1, H * W).transpose(1, 2)  # B, HW, C
        
        # Attention computation
        attn = torch.softmax(torch.bmm(q, k) * self.scale, dim=-1)  # B, HW, HW
        out = torch.bmm(attn, v).transpose(1, 2).contiguous().reshape(B, C, H, W)   # B, C, H, W
        
        return (x + out).contiguous()

class PositionalEncoding(nn.Module):
    """Chess-specific positional encoding for 8x8 board"""
    
    def __init__(self, d_model: int):
        super().__init__()
        pe = torch.zeros(64, d_model)
        
        # Create positional encodings for each square
        for pos in range(64):
            row, col = pos // 8, pos % 8
            for i in range(0, d_model, 4):
                pe[pos, i] = math.sin(row / (10000 ** (i / d_model)))
                pe[pos, i + 1] = math.cos(row / (10000 ** (i / d_model)))
                if i + 2 < d_model:
                    pe[pos, i + 2] = math.sin(col / (10000 ** ((i + 2) / d_model)))
                if i + 3 < d_model:
                    pe[pos, i + 3] = math.cos(col / (10000 ** ((i + 3) / d_model)))
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        # x shape: B, L, D where L=64 (chess squares)
        return (x + self.pe[:, :x.size(1)]).contiguous()

class ChessTransformerEncoder(nn.Module):
    """Transformer encoder specialized for chess strategic planning"""
    
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 4):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Chess-specific attention masks for piece interactions
        self.register_buffer('chess_mask', self._create_chess_mask())
    
    def _create_chess_mask(self) -> torch.Tensor:
        """Create attention mask for chess piece interactions"""
        # Allow all squares to attend to all squares initially
        # Can be refined for specific piece movement patterns
        return torch.zeros(64, 64)
    
    def forward(self, x):
        # x shape: B, C, H, W -> B, L, D
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2).contiguous()  # B, 64, C
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer
        x = self.transformer(x.contiguous(), mask=self.chess_mask)
        
        # Reshape back to spatial format
        x = x.transpose(1, 2).contiguous().reshape(B, self.d_model, H, W)
        return x

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # Ensure contiguous tensor after residual connection
        out = self.relu((out + identity).contiguous())
        return out

class ChessTransformer(nn.Module):
    """
    🏆 Hybrid CNN + Transformer Chess AI
    Target Performance: 2400+ ELO, Beat Stockfish depth 12+
    """
    
    def __init__(self, 
                 input_channels: int = 12,
                 cnn_channels: int = 512,
                 cnn_blocks: int = 24,
                 transformer_layers: int = 4,
                 attention_heads: int = 8,
                 policy_head_output_size: int = 4096):
        super().__init__()
        
        # Store architecture params
        self.cnn_channels = cnn_channels
        self.cnn_blocks = cnn_blocks
        
        # Initial convolution
        self.input_conv = nn.Conv2d(input_channels, cnn_channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(cnn_channels)
        
        # CNN backbone (local pattern recognition)
        self.cnn_blocks_list = nn.ModuleList([
            ChessResidualBlock(
                cnn_channels, 
                use_attention=(i >= cnn_blocks - 4)  # Add attention to last 4 blocks
            )
            for i in range(cnn_blocks)
        ])
        
        # Transformer encoder (strategic planning)
        self.transformer = ChessTransformerEncoder(
            d_model=cnn_channels,
            nhead=attention_heads,
            num_layers=transformer_layers
        )
        
        # Strategic feature fusion
        self.fusion_conv = nn.Conv2d(cnn_channels * 2, cnn_channels, 1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(cnn_channels)
        
        # Output heads
        self.value_head = nn.Sequential(
            nn.Conv2d(cnn_channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(cnn_channels, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, policy_head_output_size)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for optimal training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: CNN + Transformer + Strategic fusion
        
        Args:
            x: Input tensor [B, 12, 8, 8] (chess position)
            
        Returns:
            value: Position evaluation [-1, 1]
            policy: Move probabilities [B, 4096]
        """
        # Initial convolution
        x = F.relu(self.input_bn(self.input_conv(x)))
        
        # CNN backbone processing
        cnn_features = x
        for block in self.cnn_blocks_list:
            cnn_features = block(cnn_features)
        
        # Transformer strategic processing
        # Clone the tensor to break the memory layout dependency before the transformer.
        transformer_features = self.transformer(cnn_features.clone())
        
        # Feature fusion (CNN + Transformer)
        fused = torch.cat([cnn_features, transformer_features], dim=1).contiguous()
        fused = F.relu(self.fusion_bn(self.fusion_conv(fused)))
        
        # Generate outputs
        value = self.value_head(fused)
        policy = self.policy_head(fused)
        
        return value, policy
    
    def get_model_size(self) -> str:
        """Get human-readable model size"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return f"Total: {total_params:,} | Trainable: {trainable_params:,} | Size: ~{total_params/1e6:.1f}M"

def create_chess_transformer(config: str = "superhuman") -> ChessTransformer:
    """
    Factory function for creating chess transformer models
    
    Args:
        config: Model configuration
            - "superhuman": 100M param model for beating Stockfish
            - "large": 50M param model for strong play
            - "medium": 25M param model for development
    """
    configs = {
        "superhuman": {
            "cnn_channels": 512,
            "cnn_blocks": 24,
            "transformer_layers": 4,
            "attention_heads": 8
        },
        "large": {
            "cnn_channels": 384,
            "cnn_blocks": 20,
            "transformer_layers": 3,
            "attention_heads": 6
        },
        "medium": {
            "cnn_channels": 256,
            "cnn_blocks": 16,
            "transformer_layers": 2,
            "attention_heads": 4
        }
    }
    
    if config not in configs:
        raise ValueError(f"Unknown config: {config}. Choose from {list(configs.keys())}")
    
    model = ChessTransformer(**configs[config])
    print(f"🚀 Created {config} ChessTransformer: {model.get_model_size()}")
    
    return model

if __name__ == "__main__":
    # Test the architecture
    model = create_chess_transformer("superhuman")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 12, 8, 8)
    
    with torch.no_grad():
        value, policy = model(x)
        
    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Value output: {value.shape}")
    print(f"✅ Policy output: {policy.shape}")
    print(f"🏆 Model ready for superhuman chess training!") 